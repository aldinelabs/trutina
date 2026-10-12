"""Pointer-head decision model.

A pretrained decoder torso (LM head discarded) produces hidden states in ONE forward
pass; a tiny head scores each option by comparing the hidden state at `<answer>` with
the hidden state at the option's last token. No autoregressive decoding, so latency
is a single prefill regardless of the number of options.

This is the architecture Strands Decider 2B uses (Qwen3.5-2B-Base torso, rank-16
LoRA, ~1M-param head in fp32).
"""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx
from mlx import nn

from trutina.prompt import Encoded

# torso(tokens[B, T]) -> hidden[B, T, H]
Torso = Callable[[mx.array], mx.array]


class PointerHead(nn.Module):
    def __init__(self, hidden_size: int, head_dim: int = 256):
        super().__init__()
        self.q = nn.Linear(hidden_size, head_dim, bias=False)
        self.k = nn.Linear(hidden_size, head_dim, bias=False)
        self.scale = head_dim**-0.5

    def __call__(self, h_answer: mx.array, h_options: mx.array, option_mask: mx.array) -> mx.array:
        """h_answer [B, H], h_options [B, K, H], option_mask [B, K] bool -> logits [B, K]."""
        q = self.q(h_answer.astype(mx.float32))
        k = self.k(h_options.astype(mx.float32))
        logits = (k @ q[:, :, None]).squeeze(-1) * self.scale
        return mx.where(option_mask, logits, mx.array(-1e9, dtype=mx.float32))


class Decider(nn.Module):
    def __init__(self, torso: nn.Module, hidden_size: int, head_dim: int = 256):
        super().__init__()
        self.torso = torso
        self.head = PointerHead(hidden_size, head_dim)

    def __call__(self, batch: Batch) -> mx.array:
        h = self.torso(batch.tokens)  # [B, T, H]
        h_answer = mx.take_along_axis(h, batch.answer_pos[:, None, None], axis=1).squeeze(1)
        h_options = mx.take_along_axis(h, batch.option_pos[:, :, None], axis=1)
        return self.head(h_answer, h_options, batch.option_mask)


class Batch:
    """Right-padded batch of encoded questions."""

    def __init__(self, items: list[Encoded], pad_id: int = 0):
        T = max(len(e.tokens) for e in items)
        K = max(len(e.option_pos) for e in items)
        self.tokens = mx.array([e.tokens + [pad_id] * (T - len(e.tokens)) for e in items])
        self.answer_pos = mx.array([e.answer_pos for e in items])
        self.option_pos = mx.array([e.option_pos + [0] * (K - len(e.option_pos)) for e in items])
        self.option_mask = mx.array(
            [[True] * len(e.option_pos) + [False] * (K - len(e.option_pos)) for e in items]
        )


def loss_fn(model: Decider, batch: Batch, targets: mx.array) -> mx.array:
    """Cross-entropy against hard labels [B] (int) or soft teacher distributions [B, K]."""
    logits = model(batch)
    if targets.ndim == 1:
        return nn.losses.cross_entropy(logits, targets, reduction="mean")
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return -(targets * mx.where(batch.option_mask, logp, 0.0)).sum(-1).mean()


def mlx_lm_torso(lm: nn.Module) -> nn.Module:
    """Return the module of an mlx-lm causal LM that yields final-normed hidden states.

    Most mlx-lm text models expose it as `.model`; multimodal wrappers nest it under
    `.language_model.model`. Verify for each new architecture (e.g. Qwen3.5 hybrid).
    """
    for path in ("model", "language_model.model"):
        mod = lm
        try:
            for attr in path.split("."):
                mod = getattr(mod, attr)
        except AttributeError:
            continue
        return mod
    raise ValueError(f"cannot find hidden-state torso on {type(lm).__name__}")
