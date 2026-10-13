"""One ROCK 5 endpoint with original depth/vision semantics and bounded residency.

Deep text uses RKLLM. Shallow, adaptive, and images use the CPU multi-exit engine.
Only one model is resident; switching backends incurs loading/quantization.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from trutina.api import handle
from trutina.rkllm_engine import RKLLMContextLimit
from trutina.schema import Request


class Rock5Trutina:
    def __init__(self, config: dict, cpu_factory, npu_factory):
        self.config = config
        self.factories = {"cpu": cpu_factory, "npu": npu_factory}
        self.noul = "decisive" if "noul_decisive" in config else "calibrated"
        self.last_exit = None
        self.last_backend = None
        self.engine = None
        self.lock = threading.Lock()

    @classmethod
    def load(cls, cpu_model: str, npu_model: str,
             library: str = "build/rkllm/libtrutina_rkllm.so", threads: int = 4,
             max_context: int | None = None):
        cpu, npu = Path(cpu_model), Path(npu_model)
        config = json.loads((cpu / "trutina.json").read_text())
        npu_config = json.loads((npu / "trutina.json").read_text())
        if config != npu_config:
            raise ValueError("CPU and NPU bundles must use the same Trutina release/calibration")
        if threads < 1:
            raise ValueError("CPU thread count must be positive")

        def cpu_factory():
            import torch

            from trutina.torch_engine import TorchTrutina

            torch.set_num_threads(threads)
            return TorchTrutina.load(str(cpu), device="cpu", cpu_int8=True,
                                     vision=True, batch_tokens=512)

        def npu_factory():
            from trutina.rkllm_engine import RKLLMTrutina

            return RKLLMTrutina.load(str(npu), library=library, max_context=max_context)

        return cls(config, cpu_factory, npu_factory)

    def decide(self, body: dict) -> dict:
        return handle(body, self)

    def __call__(self, req: Request):
        images = not isinstance(req.state, str) and any(not isinstance(p, str) for p in req.state)
        backend = "npu" if req.depth == "deep" and not images else "cpu"
        with self.lock:
            self._select(backend)
            try:
                result = self.engine(req)
            except RKLLMContextLimit:
                # Preserve CPU's longer-context behavior without treating a genuine
                # NPU runtime error as a reason to silently change backends.
                self._select("cpu")
                result = self.engine(req)
            self.last_exit = self.engine.last_exit
            return result

    def _select(self, backend):
        if self.engine is None or self.last_backend != backend:
            self._close()
            self.engine = self.factories[backend]()
            self.last_backend = backend
        self.engine.noul = self.noul

    def _close(self):
        if self.engine is not None:
            self.engine.close()
            self.engine = None
        self.last_backend = None
        self.last_exit = None

    def close(self):
        with self.lock:
            self._close()
