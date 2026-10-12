"""Multi-exit decision model: torso + per-exit layer mix + one shared joint head.

Training runs the torso once up to the last exit and returns logits for every exit.
Inference (`adaptive`) runs exit by exit and stops as soon as every question in the batch
is confident enough, continuing from the exit's hidden state rather than recomputing.

`depth` (request parameter) maps to:
  shallow  -> stop at the first exit, always
  adaptive -> stop at the first exit whose max probability clears its threshold
  deep     -> run every layer

ADAPTIVE_THRESHOLDS is tuned for exits (12, 16, 24): answer at 12 when its max probability
reaches 0.9, otherwise at 16. Exit 16 is as accurate as 24 on held-out data, so adaptive
never goes deeper (v1: same LocalLLaMA accuracy as exit 24 at ~16 layers on average).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx import nn

from trutina.backbone import forward_hidden
from trutina.head import JointHead, ScalarMix
from trutina.prompt import Encoded, Tokenizer, encode_question
from trutina.schema import Request


@dataclass
class DecisionBatch:
    tokens: mx.array       # [B, T] right-padded
    answer_pos: mx.array   # [B]
    option_pos: mx.array   # [B, K]
    option_mask: mx.array  # [B, K] bool
    is_score: mx.array     # [B] bool
    source: mx.array | None = None  # [B] label-source ids (training only; inference = 0)

    @classmethod
    def build(cls, items: Sequence[Encoded], kinds: Sequence[str], pad_id: int = 0,
              sources: Sequence[int] | None = None) -> DecisionBatch:
        T = max(len(e.tokens) for e in items)
        K = max(len(e.option_pos) for e in items)
        return cls(
            tokens=mx.array([e.tokens + [pad_id] * (T - len(e.tokens)) for e in items]),
            answer_pos=mx.array([e.answer_pos for e in items]),
            option_pos=mx.array([e.option_pos + [0] * (K - len(e.option_pos)) for e in items]),
            option_mask=mx.array([[True] * len(e.option_pos) + [False] * (K - len(e.option_pos))
                                  for e in items]),
            is_score=mx.array([k == "score" for k in kinds]),
            source=mx.array(list(sources), dtype=mx.int32) if sources is not None else None,
        )


class MultiExitDecider(nn.Module):
    def __init__(self, torso: nn.Module, hidden: int, exits: Sequence[int] = (16, 24, 32),
                 mix_every: int = 4, head_dim: int = 512, head_layers: int = 2, head_heads: int = 8,
                 read_state: bool = False, n_sources: int = 0, ordinal_score: bool = False):
        super().__init__()
        self.torso = torso
        self.exits = tuple(exits)
        self.mix_layers = tuple(sorted(set(range(mix_every, self.exits[-1] + 1, mix_every)) | set(exits)))
        self.mixes = [ScalarMix(len(self._below(e))) for e in self.exits]
        self.head = JointHead(hidden, head_dim, head_layers, head_heads, n_exits=len(self.exits),
                              read_state=read_state, n_sources=n_sources, ordinal_score=ordinal_score)

    def _below(self, e: int) -> list[int]:
        return [l for l in self.mix_layers if l <= e]

    def exit_logits(self, i: int, hs: dict[int, mx.array], b: DecisionBatch) -> mx.array:
        feats = self.mixes[i]([hs[l] for l in self._below(self.exits[i])])
        h_ans = mx.take_along_axis(feats, b.answer_pos[:, None, None], axis=1).squeeze(1)
        h_opt = mx.take_along_axis(feats, b.option_pos[:, :, None], axis=1)
        mem_mask = mx.arange(feats.shape[1])[None] <= b.answer_pos[:, None]
        return self.head(h_ans, h_opt, b.option_mask, i, feats, mem_mask, b.source, b.is_score)

    def __call__(self, b: DecisionBatch) -> list[mx.array]:
        """Logits [B, K] for every exit (training / evaluation)."""
        _, hs = forward_hidden(self.torso, b.tokens, upto=self.exits[-1], collect=self.mix_layers)
        return [self.exit_logits(i, hs, b) for i in range(len(self.exits))]

    def adaptive(self, b: DecisionBatch, thresholds: Sequence[float] | None,
                 max_exit: int | None = None) -> tuple[mx.array, int]:
        """Run exit by exit; stop when all questions clear `thresholds[i]` (or at max_exit)."""
        last = len(self.exits) - 1 if max_exit is None else max_exit
        hs: dict[int, mx.array] = {}
        start = 0
        for i, e in enumerate(self.exits[: last + 1]):
            _, new = forward_hidden(self.torso, b.tokens, upto=e, start=start,
                                    h0=hs.get(start), collect=[l for l in self.mix_layers if l > start])
            hs.update(new)
            start = e
            logits = self.exit_logits(i, hs, b)
            if i == last:
                return logits, i
            if thresholds is not None:
                conf = mx.softmax(logits, axis=-1).max(-1)
                if bool((conf >= thresholds[i]).all()):
                    return logits, i
        raise AssertionError("unreachable")


ADAPTIVE_THRESHOLDS = (0.9, 0.0)


class DeciderEngine:
    """Adapts a MultiExitDecider to the API engine interface (one sequence per question)."""

    def __init__(self, model: MultiExitDecider, tok: Tokenizer,
                 thresholds: Sequence[float] = ADAPTIVE_THRESHOLDS):
        self.model, self.tok, self.thresholds = model, tok, tuple(thresholds)
        self.last_exit: int | None = None

    def __call__(self, req: Request) -> dict[str, np.ndarray]:
        items = [encode_question(self.tok, req.state, q) for q in req.questions]
        batch = DecisionBatch.build(items, [q.kind for q in req.questions])
        if req.depth == "shallow":
            logits, self.last_exit = self.model.adaptive(batch, None, max_exit=0)
        elif req.depth == "deep":
            logits, self.last_exit = self.model.adaptive(batch, None)
        else:
            logits, self.last_exit = self.model.adaptive(batch, self.thresholds)
        probs = np.array(mx.softmax(logits, axis=-1))
        return {q.id: probs[i, : len(items[i].option_pos)] for i, q in enumerate(req.questions)}
