"""Decision losses over padded option logits.

All take logits [B, K], a bool option mask [B, K] and target distributions [B, K] (one-hot
for hard labels, teacher probabilities for distillation):

- `soft_ce`: cross-entropy against (optionally label-smoothed) targets; equals KL up to a
  constant for soft targets.
- `brier`:   squared error of the probability vector (proper scoring rule, calibration).
- `rps`:     ranked probability score on cumulative distributions; used for `score`
             questions so that near-miss levels cost less than far ones.
"""

from __future__ import annotations

import mlx.core as mx


def masked_log_softmax(logits: mx.array, mask: mx.array) -> mx.array:
    logits = mx.where(mask, logits, -1e9)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def smooth(targets: mx.array, mask: mx.array, eps: float, soft_too: bool = False) -> mx.array:
    """Label smoothing for one-hot rows only (unless `soft_too`): teacher distributions
    already carry their uncertainty, and smoothing them pushes the fit towards uniform."""
    k = mask.sum(-1, keepdims=True)
    e = eps if soft_too else mx.where(targets.max(-1, keepdims=True) > 1 - 1e-6, eps, 0.0)
    return mx.where(mask, (1 - e) * targets + e / k, 0.0)


def soft_ce(logits: mx.array, targets: mx.array, mask: mx.array, eps: float = 0.0,
            smooth_soft: bool = False) -> mx.array:
    t = smooth(targets, mask, eps, smooth_soft) if eps else targets
    return -(t * mx.where(mask, masked_log_softmax(logits, mask), 0.0)).sum(-1)


def brier(logits: mx.array, targets: mx.array, mask: mx.array) -> mx.array:
    p = mx.exp(masked_log_softmax(logits, mask))
    return mx.where(mask, (p - targets) ** 2, 0.0).sum(-1)


def rps(logits: mx.array, targets: mx.array, mask: mx.array) -> mx.array:
    p = mx.exp(masked_log_softmax(logits, mask))
    k = mx.maximum(mask.sum(-1) - 1, 1)
    diff = mx.cumsum(p, axis=-1) - mx.cumsum(targets, axis=-1)
    return mx.where(mask, diff**2, 0.0).sum(-1) / k


def decision_loss(
    logits: mx.array, targets: mx.array, mask: mx.array, is_score: mx.array,
    eps: float = 0.05, w_brier: float = 0.5, w_rps: float = 1.0, smooth_soft: bool = False,
) -> mx.array:
    """Per-item loss [B]: CE (label-smoothed on hard labels) + Brier, plus RPS on score questions."""
    loss = soft_ce(logits, targets, mask, eps, smooth_soft) + w_brier * brier(logits, targets, mask)
    return loss + w_rps * mx.where(is_score, rps(logits, targets, mask), 0.0)
