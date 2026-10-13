"""RK3588 NPU inference using RKLLM 1.3.1; calibrated full-depth text decisions.

RKLLM exposes final logits, not intermediate-layer execution. Adaptive requests use
the final exit (32); shallow requests are rejected rather than silently changing depth.
All typed decisions, the wide tournament, prompt format and yes/no mapping are shared
with the release. No torch or MLX dependency is needed on the board.
"""
from __future__ import annotations

import ctypes
import dataclasses
import json
import threading
from pathlib import Path

import numpy as np

from trutina import wide
from trutina.api import RequestError, handle
from trutina.readout import FORMAT_VERSION, decisive, full_text, prompt, wide_labels
from trutina.schema import Request


class RKLLMContextLimit(RequestError):
    """The board export's context cap is smaller than this prompt."""


class RKLLMRuntime:
    def __init__(self, model: Path, library: Path, max_context: int):
        self.lib = ctypes.CDLL(str(library.resolve()))
        ptr = ctypes.POINTER(ctypes.c_int32)
        self.lib.trutina_rkllm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.trutina_rkllm_open.restype = ctypes.c_int
        self.lib.trutina_rkllm_logits.argtypes = [ctypes.c_void_p, ptr, ctypes.c_size_t, ptr,
                                                ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        self.lib.trutina_rkllm_logits.restype = ctypes.c_int
        self.lib.trutina_rkllm_error.argtypes = [ctypes.c_void_p]
        self.lib.trutina_rkllm_error.restype = ctypes.c_char_p
        self.lib.trutina_rkllm_close.argtypes = [ctypes.c_void_p]
        self.lib.trutina_rkllm_close.restype = None
        self.handle = ctypes.c_void_p()
        self.lock = threading.Lock()
        rc = self.lib.trutina_rkllm_open(str(model.resolve()).encode(), max_context,
                                        ctypes.byref(self.handle))
        if rc:
            raise RuntimeError(f"RKLLM initialization failed ({rc}); check video/render permissions "
                               "and driver/runtime versions")

    def logits(self, tokens: list[int], labels: list[int]) -> np.ndarray:
        ids = np.ascontiguousarray(tokens, dtype=np.int32)
        labs = np.ascontiguousarray(labels, dtype=np.int32)
        out = np.empty(len(labels), dtype=np.float32)
        with self.lock:
            if not self.handle.value:
                raise RuntimeError("RKLLM runtime is closed")
            rc = self.lib.trutina_rkllm_logits(
                self.handle, ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ids.size,
                labs.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), labs.size,
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            if rc:
                message = self.lib.trutina_rkllm_error(self.handle).decode(errors="replace")
                raise RuntimeError(message or f"RKLLM inference failed ({rc})")
        return out

    def close(self):
        with self.lock:
            if self.handle.value:
                self.lib.trutina_rkllm_close(self.handle)
                self.handle = ctypes.c_void_p()


class RKLLMTrutina:
    def __init__(self, runtime, tok, config: dict, scale: float, max_context: int = 4096):
        if config["format_version"] != FORMAT_VERSION:
            raise ValueError("Unsupported Trutina prompt format")
        if not np.isfinite(scale) or scale <= 0 or max_context < 2:
            raise ValueError("Invalid RKLLM scale or context limit")
        self.runtime, self.tok, self.config = runtime, tok, config
        self.scale, self.max_context = scale, max_context
        self.labels = wide_labels(tok)[1][:wide.PASS]
        self.noul = "decisive" if "noul_decisive" in config else "calibrated"
        self.last_exit = config["exits"][-1]
        self.lock = threading.Lock()

    @classmethod
    def load(cls, path: str, library: str = "build/rkllm/libtrutina_rkllm.so",
             max_context: int | None = None):
        from transformers import AutoTokenizer

        local = Path(path)
        manifest = json.loads((local / "rkllm.json").read_text())
        if manifest["format_version"] != 1 or manifest["runtime_version"] != "1.3.1":
            raise ValueError("RKLLM bundle requires the pinned 1.3.1 bridge/runtime")
        limit = manifest["max_context"] if max_context is None else max_context
        if limit > manifest["max_context"] or limit < 2:
            raise ValueError("Context limit exceeds the exported RKLLM model")
        config = json.loads((local / "trutina.json").read_text())
        model = local / manifest["model_file"]
        if not model.is_file():
            raise FileNotFoundError(model)
        tok = AutoTokenizer.from_pretrained(local, local_files_only=True)
        runtime = RKLLMRuntime(model, Path(library), limit)
        try:
            return cls(runtime, tok, config, manifest["final_logit_scale"], limit)
        except Exception:
            runtime.close()
            raise

    def decide(self, body: dict) -> dict:
        return handle(body, self)

    def __call__(self, req: Request) -> dict[str, np.ndarray]:
        with self.lock:
            return self._decide(req)

    def _decide(self, req: Request) -> dict[str, np.ndarray]:
        if req.depth == "shallow":
            raise RequestError("RKLLM cannot expose layer 12; use the torch CPU engine for shallow")
        if not isinstance(req.state, str) and any(not isinstance(p, str) for p in req.state):
            raise RequestError("RKLLM engine is text-only; use the torch vision engine for images")
        result = {}
        for q in req.questions:
            k = len(q.resolved_options())
            if q.kind == "choice" and k > wide.PASS:
                parts = []
                for group in wide.groups(k):
                    sub = dataclasses.replace(q, options=tuple(q.options[i] for i in group),
                                              descriptions=tuple(q.descriptions[i] for i in group)
                                              if q.descriptions else ())
                    parts.append((group, self._score(req.state, sub)))
                final = wide.finalists(parts)
                sub = dataclasses.replace(q, options=tuple(q.options[i] for i in final),
                                          descriptions=tuple(q.descriptions[i] for i in final)
                                          if q.descriptions else ())
                p = wide.combine(k, parts, final, self._score(req.state, sub))
            else:
                p = self._score(req.state, q)
            if q.kind == "noul" and self.noul == "decisive":
                mapping = self.config["noul_decisive"]
                y = decisive(float(p[1]), mapping["T"], mapping["b"])
                p = np.array([1-y, y])
            result[q.id] = p
        return result

    def _score(self, state, q) -> np.ndarray:
        text = full_text(self.tok, prompt(state, q, "sandwich", "letter"), True, "letter")
        tokens = self.tok.encode(text, add_special_tokens=False)
        # Reserve one runtime output position even though only prefill logits are requested.
        if len(tokens) >= self.max_context:
            raise RKLLMContextLimit(f"{len(tokens)} prompt tokens exceed RKLLM limit {self.max_context-1}")
        k = len(q.resolved_options())
        z = np.asarray(self.runtime.logits(tokens, self.labels[:k]), dtype=np.float64)
        if z.shape != (k,) or not np.isfinite(z).all():
            raise RuntimeError("RKLLM returned invalid decision logits")
        z *= self.scale
        p = np.exp(z-z.max())
        return p / p.sum()

    def close(self):
        self.runtime.close()
