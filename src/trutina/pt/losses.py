"""Torch mirror of `trutina.losses` (see there for the definitions)."""

from __future__ import annotations

import torch


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(torch.where(mask, logits, -1e9), -1)


def smooth(targets: torch.Tensor, mask: torch.Tensor, eps: float, soft_too: bool = False) -> torch.Tensor:
    k = mask.sum(-1, keepdim=True)
    e = eps if soft_too else torch.where(targets.amax(-1, keepdim=True) > 1 - 1e-6, eps, 0.0)
    return torch.where(mask, (1 - e) * targets + e / k, 0.0)


def soft_ce(logits, targets, mask, eps: float = 0.0, smooth_soft: bool = False) -> torch.Tensor:
    t = smooth(targets, mask, eps, smooth_soft) if eps else targets
    return -(t * torch.where(mask, masked_log_softmax(logits, mask), 0.0)).sum(-1)


def brier(logits, targets, mask) -> torch.Tensor:
    p = masked_log_softmax(logits, mask).exp()
    return torch.where(mask, (p - targets) ** 2, 0.0).sum(-1)


def rps(logits, targets, mask) -> torch.Tensor:
    p = masked_log_softmax(logits, mask).exp()
    k = (mask.sum(-1) - 1).clamp(min=1)
    diff = p.cumsum(-1) - targets.cumsum(-1)
    return torch.where(mask, diff**2, 0.0).sum(-1) / k


def decision_loss(logits, targets, mask, is_score, eps: float = 0.05, w_brier: float = 0.5,
                  w_rps: float = 1.0, smooth_soft: bool = False) -> torch.Tensor:
    loss = soft_ce(logits, targets, mask, eps, smooth_soft) + w_brier * brier(logits, targets, mask)
    return loss + w_rps * torch.where(is_score, rps(logits, targets, mask), 0.0)
