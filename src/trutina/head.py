"""Decision head shared by all early exits.

- `ScalarMix`: learned softmax-weighted mix of (RMS-normalised) hidden states from several
  layers; one per exit, so each exit reads the layers below it.
- `JointHead`: a small transformer over [answer, option_1..option_K] with *no positional
  encoding*, so permuting the options permutes the logits (no option-order bias). Options
  attend to each other, which a per-option dot product cannot do. An exit embedding tells
  the shared head which depth its features come from.

Optional (v1):
- `read_state`: the answer and option slots cross-attend to the exit's features of every
  prompt token, so the head can look evidence up in the state instead of relying only on
  what the backbone summarised at the option / answer positions.
- `n_sources`: a learned embedding of the label source (teacher / dataset) added to the
  answer slot; id 0 = unknown, which is what inference uses. Lets one model absorb
  teacher-specific calibration without averaging it into every prediction.
- `ordinal_score`: for score questions, logits = log P(level) of a discretised logistic
  (location / scale from the answer slot) + the free logits, so the distribution starts
  unimodal and neighbouring levels share strength.
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn


def rms(h: mx.array, eps: float = 1e-6) -> mx.array:
    h = h.astype(mx.float32)
    return h * mx.rsqrt((h * h).mean(-1, keepdims=True) + eps)


class ScalarMix(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.weights = mx.zeros((n,))
        self.gamma = mx.ones((1,))

    def __call__(self, hs: list[mx.array]) -> mx.array:
        a = mx.softmax(self.weights)
        return self.gamma * sum(a[i] * rms(h) for i, h in enumerate(hs))


def ordinal_log_probs(mu: mx.array, scale: mx.array, mask: mx.array) -> mx.array:
    """log P(level k) for a logistic on [0, K-1] cut at k + 0.5; K = valid levels per row.
    mu, scale [B]; mask [B, K] -> [B, K] (padding = -1e9)."""
    K = mask.shape[1]
    k = mx.arange(K, dtype=mx.float32)[None]
    last = (mask.sum(-1, keepdims=True) - 1).astype(mx.float32)
    cdf = mx.where(k < last, mx.sigmoid((k + 0.5 - mu[:, None]) / scale[:, None]), 1.0)
    prev = mx.concatenate([mx.zeros((mask.shape[0], 1)), cdf[:, :-1]], axis=1)
    return mx.where(mask, mx.log(mx.maximum(cdf - prev, 1e-6)), -1e9)


class JointHead(nn.Module):
    def __init__(self, hidden: int, dim: int = 512, layers: int = 2, heads: int = 8, n_exits: int = 3,
                 read_state: bool = False, n_sources: int = 0, ordinal_score: bool = False):
        super().__init__()
        if read_state:
            self.mem_proj = nn.Linear(hidden, dim)
            self.read_ln = nn.LayerNorm(dim)
            self.read = nn.MultiHeadAttention(dim, heads)
        if n_sources:
            self.source = nn.Embedding(n_sources, dim)
        if ordinal_score:
            self.ordinal = nn.Linear(dim, 2)  # location logit, scale
        self.proj = nn.Linear(hidden, dim)
        self.role = nn.Embedding(2, dim)  # 0 = answer/query slot, 1 = option
        self.exit = nn.Embedding(n_exits, dim)
        self.encoder = nn.TransformerEncoder(layers, dim, heads, mlp_dims=2 * dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.bias = nn.Linear(dim, 1)
        self.scale = dim**-0.5

    def __call__(self, h_ans: mx.array, h_opt: mx.array, opt_mask: mx.array, exit_idx: int,
                 mem: mx.array | None = None, mem_mask: mx.array | None = None,
                 source: mx.array | None = None, is_score: mx.array | None = None) -> mx.array:
        """h_ans [B, H], h_opt [B, K, H], opt_mask [B, K] bool -> logits [B, K] (padding = -1e9).
        mem [B, T, H] / mem_mask [B, T]: token features to read (`read_state`); source [B]
        int ids (`n_sources`); is_score [B] bool (`ordinal_score`)."""
        B, K, _ = h_opt.shape
        x = self.proj(mx.concatenate([h_ans[:, None], h_opt], axis=1))
        roles = mx.concatenate([mx.zeros((B, 1), mx.int32), mx.ones((B, K), mx.int32)], axis=1)
        x = x + self.role(roles) + self.exit(mx.array(exit_idx))
        if "source" in self:
            src = self.source(source if source is not None else mx.zeros((B,), mx.int32))
            x = x + mx.concatenate([src[:, None], mx.zeros((B, K, x.shape[-1]))], axis=1)
        if "read" in self:
            m = self.mem_proj(mem)
            x = x + self.read(self.read_ln(x), m, m, mx.where(mem_mask, 0.0, -1e9)[:, None, None, :])
        keep = mx.concatenate([mx.ones((B, 1), mx.bool_), opt_mask], axis=1)
        attn_mask = mx.where(keep, 0.0, -1e9)[:, None, None, :]
        y = self.encoder(x, attn_mask)
        a, o = y[:, 0], y[:, 1:]
        logits = (o @ self.q(a)[:, :, None]).squeeze(-1) * self.scale + self.bias(o).squeeze(-1)
        if "ordinal" in self and is_score is not None:
            loc, sc = self.ordinal(a).astype(mx.float32).split(2, axis=-1)
            last = (opt_mask.sum(-1) - 1).astype(mx.float32)
            ordl = ordinal_log_probs(mx.sigmoid(loc[:, 0]) * last, nn.softplus(sc[:, 0]) + 0.05, opt_mask)
            logits = mx.where(is_score[:, None], logits + ordl, logits)
        return mx.where(opt_mask, logits, -1e9)
