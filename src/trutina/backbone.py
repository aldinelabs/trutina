"""Layer-level access to an mlx-lm Qwen3.5 text torso.

Adds what the stock forward doesn't expose:
- hidden states of selected intermediate layers (for multi-layer readout / early exit),
- stopping after layer `upto` (shallow exit),
- forking a batch-1 prefix cache into B branches (prefix-fork encoding).

Qwen3.5 interleaves GatedDeltaNet layers (ArraysCache: [conv_state, ssm_state]) with
full attention layers (KVCache) 3:1. Both keep batch on axis 0, so a fork is a repeat.
"""

from __future__ import annotations

from collections.abc import Iterable

import mlx.core as mx
from mlx import nn
from mlx_lm import load as _load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache


def load_backbone(path: str) -> tuple[nn.Module, object]:
    """Returns (text_model, tokenizer); text_model has embed_tokens / layers / norm."""
    lm, tok = _load(path)
    return lm.model, tok


def make_cache(torso: nn.Module) -> list:
    return [ArraysCache(size=2) if l.is_linear else KVCache() for l in torso.layers]


def forward_hidden(
    torso: nn.Module,
    tokens: mx.array,
    cache: list | None = None,
    upto: int | None = None,
    collect: Iterable[int] = (),
    start: int = 0,
    h0: mx.array | None = None,
    embeds: mx.array | None = None,
    rope=None,
) -> tuple[mx.array, dict[int, mx.array]]:
    """Run layers [start, upto) on tokens [B, T].

    With `start > 0`, `h0` must be the raw hidden after `start` layers (as returned in
    `collect`); this continues an early-exited pass without recomputing earlier layers.
    `embeds` replaces the token embeddings (images filled in); `rope` (e.g.
    `trutina.mlx_vision.MRoPE`) replaces the full-attention layers' rope for this call.

    Returns (final-normed hidden of the last layer run, {layer_idx: raw hidden after
    that layer} for idx in `collect`; layer_idx is 1-based, i.e. "after k layers").
    """
    stop = upto or len(torso.layers)
    if cache is None:
        cache = [None] * len(torso.layers)
    if start:
        assert h0 is not None, "continuing from a layer needs its hidden state h0"
        h = h0
    else:
        h = embeds if embeds is not None else torso.embed_tokens(tokens)
    fa_mask = create_attention_mask(h, cache[torso.fa_idx])
    ssm_mask = create_ssm_mask(h, cache[torso.ssm_idx])
    want = set(collect)
    out: dict[int, mx.array] = {}
    saved = []
    if rope is not None:
        for layer in torso.layers:
            if not layer.is_linear:
                saved.append((layer.self_attn, layer.self_attn.rope))
                layer.self_attn.rope = rope
    try:
        for i in range(start + 1, stop + 1):
            layer, c = torso.layers[i - 1], cache[i - 1]
            h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=c)
            if i in want:
                out[i] = h
    finally:
        for attn, r in saved:
            attn.rope = r
    return torso.norm(h), out


def fork_cache(cache: list, n: int) -> list:
    """Copy a batch-1 cache into n independent branches (batch axis 0)."""
    forked = []
    for c in cache:
        if isinstance(c, KVCache):
            f = KVCache()
            f.keys = mx.repeat(c.keys[..., : c.offset, :], n, axis=0)
            f.values = mx.repeat(c.values[..., : c.offset, :], n, axis=0)
            f.offset = c.offset
        elif isinstance(c, ArraysCache):
            f = ArraysCache(size=len(c.cache))
            f.cache = [None if a is None else mx.repeat(a, n, axis=0) for a in c.cache]
        else:
            raise TypeError(f"unsupported cache type {type(c).__name__}")
        forked.append(f)
    return forked
