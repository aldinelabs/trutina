"""Experimental RK3588 Linear offload with CPU-controlled adaptive execution.

Weights use symmetric INT8 scales per output channel; activations use scales per
token. Gated DeltaNet, attention, norms, residuals and trained exits stay in torch.
No RKLLM graph or layer scheduler is involved.
"""
from __future__ import annotations

import ctypes
import threading
import weakref
from pathlib import Path

import numpy as np
import torch
from torch import nn


def quantize_weight(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weight = np.asarray(weight, dtype=np.float32)
    if weight.ndim != 2 or not np.isfinite(weight).all():
        raise ValueError("Expected finite 2D Linear weights")
    maximum = np.max(np.abs(weight), axis=1)
    scales = np.where(maximum > 0, np.maximum(maximum / 127, np.finfo(np.float32).tiny), 1).astype(np.float32)
    q = np.clip(np.rint(weight / scales[:, None]), -127, 127).astype(np.int8)
    return np.ascontiguousarray(q.T), np.ascontiguousarray(scales)


def _release(lib, pool, handles):
    for handle in list(handles.values()):
        lib.trutina_mm_free(handle)
    handles.clear()
    lib.trutina_mm_close(pool)


class MatmulRuntime:
    def __init__(self, library: str | Path, chunk: int = 64, cores: int = 1):
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        lib = self.lib
        ptr = ctypes.c_void_p
        fp = ctypes.POINTER(ctypes.c_float)
        lib.trutina_mm_open.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ptr)]
        lib.trutina_mm_open.restype = ctypes.c_int
        lib.trutina_mm_linear.argtypes = [ptr, ctypes.POINTER(ctypes.c_int8), fp,
                                         ctypes.c_int, ctypes.c_int, ctypes.POINTER(ptr)]
        lib.trutina_mm_linear.restype = ctypes.c_int
        lib.trutina_mm_forward.argtypes = [ptr, fp, ctypes.c_size_t, fp]
        lib.trutina_mm_forward.restype = ctypes.c_int
        lib.trutina_mm_error.argtypes = [ptr]
        lib.trutina_mm_error.restype = ctypes.c_char_p
        lib.trutina_mm_calls.argtypes = [ptr]
        lib.trutina_mm_calls.restype = ctypes.c_uint64
        lib.trutina_mm_stats.argtypes = [ptr, ctypes.POINTER(ctypes.c_uint64)]
        lib.trutina_mm_stats.restype = ctypes.c_int
        lib.trutina_mm_free.argtypes = [ptr]
        lib.trutina_mm_free.restype = ctypes.c_int
        lib.trutina_mm_close.argtypes = [ptr]
        lib.trutina_mm_close.restype = ctypes.c_int
        self.pool = ptr()
        self.lock = threading.RLock()
        self.handles: dict[int, ctypes.c_void_p] = {}
        self.shapes: dict[int, tuple[int, int, int, int]] = {}
        if lib.trutina_mm_open(chunk, cores, ctypes.byref(self.pool)):
            raise ValueError("Invalid NPU chunk size or core mask")
        self.finalizer = weakref.finalize(self, _release, lib, self.pool, self.handles)

    def _check(self, rc: int):
        if rc:
            raise RuntimeError(self.lib.trutina_mm_error(self.pool).decode(errors="replace"))

    def linear(self, weight: np.ndarray) -> int:
        original_n, original_k = weight.shape
        k_pad = (original_k + 31) // 32 * 32
        n_pad = (original_n + 31) // 32 * 32
        if (original_n, original_k) != (n_pad, k_pad):
            weight = np.pad(weight, ((0, n_pad-original_n), (0, k_pad-original_k)))
        q, scales = quantize_weight(weight)
        k, n = q.shape
        handle = ctypes.c_void_p()
        with self.lock:
            if not self.finalizer.alive:
                raise RuntimeError("NPU runtime is closed")
            self._check(self.lib.trutina_mm_linear(
                self.pool, q.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), k, n, ctypes.byref(handle)))
            self.handles[handle.value] = handle
            self.shapes[handle.value] = (original_k, original_n, k, n)
        return handle.value

    def forward(self, handle: int, x: np.ndarray, n: int) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("Expected a matrix of token activations")
        with self.lock:
            if handle not in self.handles or not self.finalizer.alive:
                raise RuntimeError("NPU Linear is closed")
            original_k, original_n, k, padded_n = self.shapes[handle]
            if x.shape[1] != original_k or n != original_n:
                raise ValueError("NPU Linear input/output width mismatch")
            if k != original_k:
                x = np.pad(x, ((0, 0), (0, k-original_k)))
            out = np.empty((len(x), padded_n), dtype=np.float32)
            self._check(self.lib.trutina_mm_forward(
                self.handles[handle], x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(x),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))
        return out if n == padded_n else np.ascontiguousarray(out[:, :n])

    @property
    def calls(self) -> int:
        with self.lock:
            return int(self.lib.trutina_mm_calls(self.pool)) if self.finalizer.alive else 0

    def close(self):
        with self.lock:
            self.finalizer()

    @property
    def stats(self) -> dict[str, float]:
        with self.lock:
            if not self.finalizer.alive:
                return {}
            result = (ctypes.c_uint64 * 5)()
            self._check(self.lib.trutina_mm_stats(self.pool, result))
            return {name: result[i+1] / 1e9 for i, name in
                    enumerate(("bind_seconds", "quant_seconds", "run_seconds", "output_seconds"))}


class NpuLinear(nn.Module):
    def __init__(self, linear: nn.Linear, runtime: MatmulRuntime):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.runtime = runtime
        self.handle = runtime.linear(linear.weight.detach().float().cpu().numpy())
        self.register_buffer("bias", linear.bias.detach().float().cpu().clone()
                             if linear.bias is not None else None)

    def forward(self, x):
        if x.device.type != "cpu" or x.shape[-1] != self.in_features:
            raise ValueError("NPU Linear requires CPU activations of the expected width")
        values = x.detach().float().contiguous().reshape(-1, self.in_features).numpy()
        out = torch.from_numpy(self.runtime.forward(self.handle, values, self.out_features))
        out = out.reshape(*x.shape[:-1], self.out_features)
        return out + self.bias if self.bias is not None else out


def offload_torso(torso, runtime: MatmulRuntime) -> int:
    from trutina.cpu_quant import release_cpu_memory

    def replace(module):
        count = 0
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, NpuLinear(child, runtime))
                count += 1
            else:
                count += replace(child)
        return count

    count = 0
    try:
        for layer in torso.layers:
            layer.float()
            count += replace(layer)
            release_cpu_memory()
        torso.embed_tokens.float()
        torso.norm.float()
    except Exception:
        runtime.close()
        raise
    return count
