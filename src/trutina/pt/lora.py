"""LoRA with the same parameterisation and names as mlx-lm's `LoRALinear`.

y = x W^T + scale * (x @ lora_a) @ lora_b,  lora_a [in, r], lora_b [r, out].
"""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 16, scale: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.linear = linear
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        bound = 1 / math.sqrt(linear.in_features)
        dev = linear.weight.device
        self.lora_a = nn.Parameter(torch.empty(linear.in_features, r, device=dev).uniform_(-bound, bound))
        self.lora_b = nn.Parameter(torch.zeros(r, linear.out_features, device=dev))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute in the activations' dtype: casting the small adapters is far cheaper than
        # casting every activation to the fp32 master weights
        z = (self.dropout(x) @ self.lora_a.to(x.dtype)) @ self.lora_b.to(x.dtype)
        return self.linear(x) + self.scale * z


def add_lora(layers: nn.ModuleList, rank: int = 16, scale: float = 2.0, dropout: float = 0.0) -> None:
    """Wrap every nn.Linear inside `layers` (already frozen) with LoRA."""
    for layer in layers:
        targets = [(name, m) for name, m in layer.named_modules() if isinstance(m, nn.Linear)]
        for name, m in targets:
            parent_name, _, child = name.rpartition(".")
            parent = layer.get_submodule(parent_name) if parent_name else layer
            setattr(parent, child, LoRALinear(m, rank, scale, dropout))
