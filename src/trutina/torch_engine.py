"""trutina-4b on CUDA with PyTorch + transformers (text only; full depth).

  from trutina.torch_engine import TorchTrutina
  model = TorchTrutina.load("aldinelabs/trutina-4b", device="cuda")
  model.decide({"state": "...", "questions": {...}})

Same prompt, readout and calibration as the MLX engine (`trutina.engine`) at depth "deep" without
rotations: the final layer only, no early exits. Questions are length-sorted into padded batches
under a token budget; a batch that runs out of GPU memory is split in half.
Install the CUDA kernels for speed: flash-linear-attention and causal-conv1d.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from trutina import wide
from trutina.api import RequestError, handle
from trutina.readout import FORMAT_VERSION, full_text, prompt, wide_labels
from trutina.schema import Question, Request, State


class TorchTrutina:
    def __init__(self, torso, tok, config: dict, prior_logscale: float, device: str, batch_tokens: int):
        self.torso, self.tok, self.config, self.device = torso, tok, config, device
        self.scale = float(np.exp(prior_logscale)) / config["temperature"][str(config["exits"][-1])]
        ids = torch.tensor(wide_labels(tok)[1][: wide.PASS], device=device)
        self.rows = torso.embed_tokens.weight[ids].float()  # tied LM head: the option letters' rows
        self.batch_tokens = batch_tokens

    @classmethod
    def load(cls, path: str = "aldinelabs/trutina-4b", device: str = "cuda", dtype=torch.bfloat16,
             batch_tokens: int = 32768) -> TorchTrutina:
        from safetensors import safe_open
        from transformers import AutoModelForCausalLM, AutoTokenizer

        local = Path(path)
        if not local.exists():
            from huggingface_hub import snapshot_download

            local = Path(snapshot_download(path))
        config = json.loads((local / "trutina.json").read_text())
        if config["format_version"] != FORMAT_VERSION:
            raise ValueError(f"release format {config['format_version']}, this package reads {FORMAT_VERSION}")
        lm = AutoModelForCausalLM.from_pretrained(str(local), dtype=dtype, attn_implementation="sdpa")
        torso = getattr(lm.model, "language_model", lm.model).to(device).eval()
        with safe_open(str(local / "readout.safetensors"), "np") as f:
            prior = float(f.get_tensor("prior_logscale"))
        return cls(torso, AutoTokenizer.from_pretrained(str(local)), config, prior, device, batch_tokens)

    def decide(self, body: dict) -> dict:
        return handle(body, self)

    def __call__(self, req: Request) -> dict[str, np.ndarray]:
        if not isinstance(req.state, str) and any(not isinstance(p, str) for p in req.state):
            raise RequestError("the torch engine is text-only; use the MLX engine for images")
        jobs: list[tuple[int, list[int] | None]] = []
        qs: list[Question] = []
        for qi, q in enumerate(req.questions):
            k = len(q.resolved_options())
            for g in wide.groups(k) if q.kind == "choice" and k > wide.PASS else [None]:
                jobs.append((qi, g))
                qs.append(q if g is None else _subset(q, g))
        probs = self._forward(req.state, qs)
        res, parts = {}, {}
        for (qi, g), p in zip(jobs, probs):
            if g is None:
                res[req.questions[qi].id] = p
            else:
                parts.setdefault(qi, []).append((g, p))
        if parts:
            fins = {qi: wide.finalists(ps) for qi, ps in parts.items()}
            finals = self._forward(req.state, [_subset(req.questions[qi], f) for qi, f in fins.items()])
            for (qi, f), pf in zip(fins.items(), finals):
                q = req.questions[qi]
                res[q.id] = wide.combine(len(q.options), parts[qi], f, pf)
        return res

    def encode(self, state: State, q: Question) -> list[int]:
        text = full_text(self.tok, prompt(state, q, "sandwich", "letter"), True, "letter")
        return self.tok.encode(text, add_special_tokens=False)

    @torch.no_grad()
    def _forward(self, state: State, qs: Sequence[Question]) -> list[np.ndarray]:
        seqs = [self.encode(state, q) for q in qs]
        ks = [len(q.resolved_options()) for q in qs]
        out: dict[int, np.ndarray] = {}
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        chunk: list[int] = []
        for i in [*order, None]:
            if i is not None and (not chunk or len(seqs[i]) * (len(chunk) + 1) <= self.batch_tokens):
                chunk.append(i)
                continue
            for j, p in zip(chunk, self._batch([seqs[j] for j in chunk], [ks[j] for j in chunk])):
                out[j] = p
            chunk = [i] if i is not None else []
        return [out[i] for i in range(len(seqs))]

    def _batch(self, seqs: list[list[int]], ks: list[int]) -> list[np.ndarray]:
        try:
            T = max(map(len, seqs))
            ids = torch.tensor([s + [0] * (T - len(s)) for s in seqs], device=self.device)
            mask = torch.tensor([[1] * len(s) + [0] * (T - len(s)) for s in seqs], device=self.device)
            h = self.torso(input_ids=ids, attention_mask=mask).last_hidden_state
            last = torch.tensor([len(s) - 1 for s in seqs], device=self.device)
            logits = h[torch.arange(len(seqs), device=self.device), last].float() @ self.rows.T * self.scale
            return [torch.softmax(logits[r, :k], -1).cpu().double().numpy() for r, k in enumerate(ks)]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(seqs) == 1:
                raise RequestError(f"a {len(seqs[0])}-token question does not fit in GPU memory") from None
            m = len(seqs) // 2
            return self._batch(seqs[:m], ks[:m]) + self._batch(seqs[m:], ks[m:])


def _subset(q: Question, idx: Sequence[int]) -> Question:
    d = q.resolved_descriptions()
    return dataclasses.replace(q, options=tuple(q.options[i] for i in idx), descriptions=tuple(d[i] for i in idx))
