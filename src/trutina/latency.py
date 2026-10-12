"""Warm, single-request latency measurement (the Decision Index protocol: one process,
one request at a time; median and p95 reported)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class LatencyStats:
    n: int
    p50_ms: float
    p95_ms: float
    mean_ms: float


def measure(fn: Callable[[], mx.array], n: int = 50, warmup: int = 5) -> LatencyStats:
    for _ in range(warmup):
        mx.eval(fn())
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        mx.eval(fn())  # MLX is lazy: force evaluation inside the timed region
        times.append((time.perf_counter() - t0) * 1000)
    a = np.array(times)
    return LatencyStats(n, float(np.median(a)), float(np.percentile(a, 95)), float(a.mean()))
