"""trutina-4b on CUDA with PyTorch + transformers (text only).

  from trutina.torch_engine import TorchTrutina
  model = TorchTrutina.load("aldinelabs/trutina-4b", device="cuda")
  model.decide({"state": "...", "questions": {...}})

Same prompt, readout, exits and calibration as the MLX engine (`trutina.engine`). Depth: `shallow`
answers at layer 12, `adaptive` walks the exits (12, 16, 20, 24) and stops at the first whose top
probability reaches the threshold for every question of the batch, continuing from that layer's
hidden states otherwise; `deep` runs all 32 layers (without the MLX engine's option rotations).
Questions are length-sorted into padded batches under a token budget; a batch that runs out of GPU
memory is split in half.
Install the CUDA kernels for speed: flash-linear-attention and causal-conv1d.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from trutina import wide
from trutina.api import RequestError, handle
from trutina.readout import FORMAT_VERSION, full_text, prompt, wide_labels
from trutina.schema import Question, Request, State


class TorchTrutina:
    def __init__(self, torso, tok, config: dict, readout: dict, device: str, batch_tokens: int):
        self.torso, self.tok, self.config, self.device = torso, tok, config, device
        self.exits = config["exits"][:-1]
        temps = [config["temperature"][str(e)] for e in config["exits"]]
        scales = [*np.exp(readout["lens_logscale"]), np.exp(readout["prior_logscale"])]
        self.scales = [float(sc) / t for sc, t in zip(scales, temps)]  # logit scale / temperature, per exit
        self.lens = []
        for i in range(len(self.exits)):
            lin = torch.nn.Linear(readout[f"lens.{i}.weight"].shape[1], readout[f"lens.{i}.weight"].shape[0])
            lin.weight.data = torch.from_numpy(readout[f"lens.{i}.weight"])
            lin.bias.data = torch.from_numpy(readout[f"lens.{i}.bias"])
            self.lens.append(lin.to(device).float().eval())
        self.threshold = config["adaptive_threshold"]
        ids = torch.tensor(wide_labels(tok)[1][: wide.PASS], device=device)
        self.rows = torso.embed_tokens.weight[ids].float()  # tied LM head: the option letters' rows
        self.batch_tokens = batch_tokens
        self.depth = "adaptive"
        self.last_exit: int | None = None

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
            readout = {k: f.get_tensor(k).astype(np.float32) for k in f.keys()}  # noqa: SIM118 (safe_open)
        return cls(torso, AutoTokenizer.from_pretrained(str(local)), config, readout, device, batch_tokens)

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
        self.depth = req.depth
        probs = self._forward(req.state, qs)
        res, parts = {}, {}
        for (qi, g), p in zip(jobs, probs):
            if g is None:
                res[req.questions[qi].id] = p
            else:
                parts.setdefault(qi, []).append((g, p))
        if parts:
            fins = {qi: wide.finalists(ps) for qi, ps in parts.items()}
            self.depth = "deep"
            finals = self._forward(req.state, [_subset(req.questions[qi], f) for qi, f in fins.items()])
            for (qi, f), pf in zip(fins.items(), finals):
                q = req.questions[qi]
                res[q.id] = wide.combine(len(q.options), parts[qi], f, pf)
        return res

    def encode(self, state: State, q: Question) -> list[int]:
        text = full_text(self.tok, prompt(state, q, "sandwich", "letter"), True, "letter")
        return self.tok.encode(text, add_special_tokens=False)

    @torch.inference_mode()
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
            return self._read(seqs, ks)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(seqs) == 1:
                raise RequestError(f"a {len(seqs[0])}-token question does not fit in GPU memory") from None
            m = len(seqs) // 2
            return self._batch(seqs[:m], ks[:m]) + self._batch(seqs[m:], ks[m:])

    def _read(self, seqs: list[list[int]], ks: list[int]) -> list[np.ndarray]:
        """Run the layers segment by segment (exit to exit), reading the exits the depth asks for."""
        T, dev, torso = max(map(len, seqs)), self.device, self.torso
        ids = torch.tensor([s + [0] * (T - len(s)) for s in seqs], device=dev)
        mask = torch.tensor([[1] * len(s) + [0] * (T - len(s)) for s in seqs], device=dev)
        rows = torch.arange(len(seqs), device=dev)
        last = torch.tensor([len(s) - 1 for s in seqs], device=dev)
        h, layer_kw = self._prepare(ids, mask)

        def probs(logits: torch.Tensor) -> list[np.ndarray]:
            return [torch.softmax(logits[r, :k], -1).cpu().double().numpy() for r, k in enumerate(ks)]

        plan = {"deep": [], "shallow": [(0, 0.0)]}.get(self.depth, [(i, self.threshold) for i in range(len(self.exits))])
        start = 0
        for i, thr in plan:
            h = self._layers(h, start, self.exits[i], layer_kw)
            start = self.exits[i]
            x = h[rows, last].float()
            z = torso.norm(x + self.lens[i](x)).float()
            p = probs(z @ self.rows.T * self.scales[i])
            if all(q.max() >= thr for q in p):
                self.last_exit = start
                return p
        h = self._layers(h, start, len(torso.layers), layer_kw)
        self.last_exit = len(torso.layers)
        return probs(torso.norm(h[rows, last]).float() @ self.rows.T * self.scales[-1])

    def _prepare(self, ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Embeddings, masks and rotary embeddings, as Qwen3_5TextModel.forward builds them."""
        torso = self.torso
        mod = sys.modules[type(torso).__module__]
        h = torso.embed_tokens(ids)
        pos = torch.arange(ids.shape[1], device=ids.device).view(1, 1, -1).expand(4, ids.shape[0], -1)
        text_pos, pos = pos[0], pos[1:]
        kw = {"config": torso.config, "inputs_embeds": h, "attention_mask": mask, "past_key_values": None,
              "position_ids": text_pos}
        masks = {"full_attention": mod.create_causal_mask(**kw),
                 "linear_attention": mod.create_recurrent_attention_mask(**kw)}
        return h, {"masks": masks, "position_embeddings": torso.rotary_emb(h, pos), "position_ids": text_pos}

    def _layers(self, h: torch.Tensor, start: int, stop: int, kw: dict) -> torch.Tensor:
        types = self.torso.config.layer_types
        for i in range(start, stop):
            h = self.torso.layers[i](h, position_embeddings=kw["position_embeddings"],
                                     attention_mask=kw["masks"][types[i]], position_ids=kw["position_ids"],
                                     past_key_values=None, use_cache=False)
        return h


def _subset(q: Question, idx: Sequence[int]) -> Question:
    d = q.resolved_descriptions()
    return dataclasses.replace(q, options=tuple(q.options[i] for i in idx), descriptions=tuple(d[i] for i in idx))
