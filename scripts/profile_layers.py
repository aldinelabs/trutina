"""Where does per-layer time go? Times each component of a Qwen3.5 layer in isolation.

  uv run python scripts/profile_layers.py --model Qwen/Qwen3.5-4B-Base
"""

from __future__ import annotations

import argparse

import mlx.core as mx
from mlx.utils import tree_flatten

from trutina.backbone import forward_hidden, load_backbone
from trutina.latency import measure


def nbytes(module) -> int:
    return sum(v.nbytes for _, v in tree_flatten(module.parameters()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    a = ap.parse_args()
    torso, _ = load_backbone(a.model)
    lin = next(l for l in torso.layers if l.is_linear)
    att = next(l for l in torso.layers if not l.is_linear)
    H = torso.embed_tokens.weight.shape[1]

    print(f"bytes/layer: linear {nbytes(lin) / 1e6:.0f} MB, attn {nbytes(att) / 1e6:.0f} MB "
          f"(at 307 GB/s: {nbytes(lin) / 307e6:.2f} / {nbytes(att) / 307e6:.2f} ms)")
    print(f"{'N':>5} {'component':<28} {'p50 ms':>8}")
    for n in (1, 8, 64, 200):
        for name, fn in components(torso, lin, att, H, n).items():
            print(f"{n:>5} {name:<28} {measure(fn, n=30, warmup=5).p50_ms:>8.3f}")


def components(torso, lin, att, H: int, n: int) -> dict:
    x = mx.random.normal((1, n, H)).astype(mx.bfloat16)
    mask = "causal" if n > 1 else None
    ids = mx.zeros((1, n), dtype=mx.int32)

    def first4():
        h = x
        for layer in torso.layers[:4]:
            h = layer(h, mask=None if layer.is_linear else mask)
        return h

    return {
        "noop (add)": lambda: x + 1,
        "linear layer (full)": lambda: lin(x),
        "  gated-delta mixer": lambda: lin.linear_attn(lin.input_layernorm(x)),
        "  mlp": lambda: lin.mlp(lin.post_attention_layernorm(x)),
        "attn layer (full)": lambda: att(x, mask=mask),
        "  attention mixer": lambda: att.self_attn(att.input_layernorm(x), mask=mask),
        "4 layers in one graph": first4,
        "whole torso": lambda: forward_hidden(torso, ids)[0],
    }


if __name__ == "__main__":
    main()
