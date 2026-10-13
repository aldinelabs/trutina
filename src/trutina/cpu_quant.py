"""Bounded-memory dynamic INT8 conversion for the ARM CPU reference backend.

Load BF16 first, then convert one decoder layer at a time. Embeddings, norms,
convolution and Gated DeltaNet state math remain floating point. Calibration
temperatures remain those of the release; quantized accuracy must be re-evaluated.
"""
from __future__ import annotations

import ctypes
import gc

import torch
from torch import nn


def quantize_torso(torso):
    if "qnnpack" not in torch.backends.quantized.supported_engines:
        raise RuntimeError("This PyTorch build has no QNNPACK CPU quantization backend")
    torch.backends.quantized.engine = "qnnpack"
    for layer in torso.layers:
        layer.float()
        # Dynamic Linear kernels take FP32 activations; the rest of this layer is
        # FP32 too. Quantize each layer independently to bound conversion memory.
        torch.ao.quantization.quantize_dynamic(layer, {nn.Linear}, dtype=torch.qint8, inplace=True)
        release_cpu_memory()
    torso.embed_tokens.float()
    torso.norm.float()


def release_cpu_memory():
    """Return freed conversion buffers to glibc instead of retaining them as RSS."""
    gc.collect()
    try:
        trim = ctypes.CDLL(None).malloc_trim
    except AttributeError:  # non-glibc platforms can still use the CPU backend
        return
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    trim(0)
