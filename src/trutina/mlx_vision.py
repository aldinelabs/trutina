"""Qwen3.5's vision tower and multimodal RoPE in MLX (port of transformers' modeling_qwen3_5).

- `VisionTower`: patch embedding (the Conv3d as a matmul over `trutina.vision.pixel_patches` rows),
  learned 48x48 position table resampled bilinearly (align_corners) to each image's grid, 24 ViT
  blocks with axial 2D RoPE and bidirectional attention within each image, then the 2x2 patch merger
  to the text width. Patches arrive in merge-block order, as `pixel_patches` produces them.
- `mrope_positions`: (3, T) text positions as in `Qwen3_5Model.get_rope_index`: text tokens advance all
  three axes together; an image's tokens take (t, row, col) offsets from the current position, and
  text after it resumes at that position + max(rows, cols).
- `MRoPE`: drop-in for the full-attention layers' `nn.RoPE` with interleaved sections [11, 11, 10]
  (rotary dims 0, 3, 6, ... follow t, 1, 4, ... rows, 2, 5, ... cols). With every axis equal it is
  exactly the text-only rope.

Weights load from the Hugging Face checkpoint's `model.visual.*` tensors (mlx-lm drops them).
"""

from __future__ import annotations

import glob
import json
import math
from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn

from trutina.vision import IMAGE_PAD, MERGE

VISION_PREFIX = "model.visual."


class VisionAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, spans: Sequence[tuple[int, int]]) -> mx.array:
        n, d = x.shape
        q, k, v = self.qkv(x).reshape(n, 3, self.heads, -1).transpose(1, 2, 0, 3)  # each [heads, n, hd]
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        scale = (d // self.heads) ** -0.5
        out = [mx.fast.scaled_dot_product_attention(q[None, :, a:b], k[None, :, a:b], v[None, :, a:b], scale=scale)[0]
               for a, b in spans]  # each image attends only to itself
        return self.proj(mx.concatenate(out, axis=1).transpose(1, 0, 2).reshape(n, d))


class VisionMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.linear_fc1 = nn.Linear(dim, hidden)
        self.linear_fc2 = nn.Linear(hidden, dim)

    def __call__(self, x):
        return self.linear_fc2(nn.gelu_approx(self.linear_fc1(x)))  # gelu_pytorch_tanh


class VisionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, hidden: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = VisionAttention(dim, heads)
        self.mlp = VisionMLP(dim, hidden)

    def __call__(self, x, cos, sin, spans):
        x = x + self.attn(self.norm1(x), cos, sin, spans)
        return x + self.mlp(self.norm2(x))


class PatchMerger(nn.Module):
    def __init__(self, dim: int, out: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.linear_fc1 = nn.Linear(dim * MERGE * MERGE, dim * MERGE * MERGE)
        self.linear_fc2 = nn.Linear(dim * MERGE * MERGE, out)

    def __call__(self, x):
        x = self.norm(x).reshape(-1, self.linear_fc1.weight.shape[1])
        return self.linear_fc2(nn.gelu(self.linear_fc1(x)))  # exact GELU


class PatchEmbed(nn.Module):
    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, dim)  # the Conv3d kernel flattened as (C, T, P, P)

    def __call__(self, x):
        return self.proj(x)


class VisionTower(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        dim, heads = cfg["hidden_size"], cfg["num_heads"]
        p, t = cfg["patch_size"], cfg["temporal_patch_size"]
        self.side = math.isqrt(cfg["num_position_embeddings"])
        self.theta = cfg.get("rope_parameters", {}).get("rope_theta", 10000.0)
        self.head_dim = dim // heads
        self.patch_embed = PatchEmbed(cfg["in_channels"] * t * p * p, dim)
        self.pos_embed = nn.Embedding(cfg["num_position_embeddings"], dim)
        self.blocks = [VisionBlock(dim, heads, cfg["intermediate_size"]) for _ in range(cfg["depth"])]
        self.merger = PatchMerger(dim, cfg["out_hidden_size"])

    def __call__(self, patches: mx.array, grids: Sequence[tuple[int, int, int]]) -> mx.array:
        """patches [sum(t*h*w), C*T*P*P] (images in order, merge-block order) -> [sum(t*h*w)/4, out]."""
        idx, w = interpolation_taps(grids, self.side)
        x = self.patch_embed(patches.astype(self.patch_embed.proj.weight.dtype))
        pos = (self.pos_embed(mx.array(idx)) * mx.array(w)[..., None]).sum(1)
        x = x + pos.astype(x.dtype)
        cos, sin = vision_rope(grids, self.head_dim, self.theta)
        spans, a = [], 0
        for t, h, ww in grids:
            for _ in range(t):
                spans.append((a, a + h * ww))
                a += h * ww
        for blk in self.blocks:
            x = blk(x, cos, sin, spans)
        return self.merger(x)


def rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate-half rope in float32 over the full last axis; cos/sin [n, hd] broadcast over heads."""
    xf = x.astype(mx.float32)
    h = xf.shape[-1] // 2
    rot = mx.concatenate([-xf[..., h:], xf[..., :h]], axis=-1)
    return (xf * cos + rot * sin).astype(x.dtype)


def block_order(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """(row, col) of every patch of an h x w grid in 2x2 merge-block order."""
    r, c = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    shape = (h // MERGE, MERGE, w // MERGE, MERGE)
    return r.reshape(shape).transpose(0, 2, 1, 3).ravel(), c.reshape(shape).transpose(0, 2, 1, 3).ravel()


def vision_rope(grids, head_dim: int, theta: float) -> tuple[mx.array, mx.array]:
    spatial = head_dim // 2
    inv = 1.0 / (theta ** (np.arange(0, spatial, 2, dtype=np.float64) / spatial))
    pos = np.concatenate([np.tile(np.stack(block_order(h, w), -1), (t, 1)) for t, h, w in grids])  # [n, 2]
    f = pos[:, :, None] * inv  # [n, 2, spatial/2]
    f = np.concatenate([f[:, 0], f[:, 1]], -1)
    f = np.concatenate([f, f], -1).astype(np.float32)  # [n, head_dim]
    return mx.array(np.cos(f)), mx.array(np.sin(f))


def interpolation_taps(grids, side: int) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear (align_corners=True) taps into the side x side position table, merge-block order."""
    idx, wts = [], []
    for t, h, w in grids:
        rows, cols = block_order(h, w)

        def axis(v, n):
            s = v * (side - 1) / (n - 1) if n > 1 else np.zeros_like(v, dtype=np.float64)
            lo = np.clip(np.floor(s).astype(int), 0, side - 1)
            hi = np.clip(lo + 1, 0, side - 1)
            return lo, hi, s - lo

        r0, r1, fr = axis(rows, h)
        c0, c1, fc = axis(cols, w)
        i = np.stack([r0 * side + c0, r0 * side + c1, r1 * side + c0, r1 * side + c1], -1)
        wt = np.stack([(1 - fr) * (1 - fc), (1 - fr) * fc, fr * (1 - fc), fr * fc], -1)
        idx.append(np.tile(i, (t, 1)))
        wts.append(np.tile(wt, (t, 1)))
    return np.concatenate(idx), np.concatenate(wts).astype(np.float32)


def mrope_positions(tokens: Sequence[int], grids: Sequence[tuple[int, int, int]]) -> np.ndarray:
    """(3, len(tokens)) positions; each run of IMAGE_PAD tokens is one image, in `grids` order."""
    out = np.zeros((3, len(tokens)), dtype=np.int64)
    pos, i, g = 0, 0, iter(grids)
    while i < len(tokens):
        if tokens[i] != IMAGE_PAD:
            out[:, i] = pos
            pos += 1
            i += 1
            continue
        t, h, w = next(g)
        gh, gw = h // MERGE, w // MERGE
        n = t * gh * gw
        tt, rr, cc = np.meshgrid(np.arange(t), np.arange(gh), np.arange(gw), indexing="ij")
        out[:, i : i + n] = np.stack([tt.ravel(), rr.ravel(), cc.ravel()]) + pos
        pos += max(gh, gw)
        i += n
    return out


class MRoPE:
    """Replaces a full-attention layer's `rope` for one forward: positions [3, B, T] fixed up front."""

    def __init__(self, positions: np.ndarray, dims: int, base: float, sections=(11, 11, 10)):
        inv = 1.0 / (base ** (np.arange(0, dims, 2, dtype=np.float64) / dims))  # [dims/2]
        f = positions[..., None].astype(np.float64) * inv  # [3, B, T, dims/2]
        thw = f[0].copy()
        for axis, offset in ((1, 1), (2, 2)):
            sl = slice(offset, sections[axis] * 3, 3)
            thw[..., sl] = f[axis][..., sl]
        thw = np.concatenate([thw, thw], -1).astype(np.float32)  # [B, T, dims]
        self.cos = mx.array(np.cos(thw))[:, None]  # [B, 1, T, dims]
        self.sin = mx.array(np.sin(thw))[:, None]
        self.dims = dims

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        if offset:
            raise ValueError("multimodal rope is for a single uncached forward")
        L = x.shape[2]
        r, rest = x[..., : self.dims], x[..., self.dims :]
        r = rope(r, self.cos[:, :, :L], self.sin[:, :, :L])
        return mx.concatenate([r, rest], axis=-1)


def snapshot_dir(model_path: str) -> Path:
    if Path(model_path).exists():
        return Path(model_path)
    hits = glob.glob(str(Path.home() / f".cache/huggingface/hub/models--{model_path.replace('/', '--')}/snapshots/*"))
    if not hits:
        raise FileNotFoundError(f"no local snapshot of {model_path}")
    return Path(max(hits))


def load_vision_tower(model_path: str, dtype=mx.float32) -> VisionTower:
    """float32 by default: in bf16 a few tokens per image drift far from the reference (cosine down to
    ~0.4 against transformers in float32; in float32 >= 0.97, mean 0.9999), and the tower is small."""
    d = snapshot_dir(model_path)
    cfg = json.loads((d / "config.json").read_text())["vision_config"]
    tower = VisionTower(cfg)
    weights = {}
    for f in sorted(d.glob("*.safetensors")):
        for k, v in mx.load(str(f)).items():
            if k.startswith(VISION_PREFIX):
                k = k[len(VISION_PREFIX):]
                if k == "patch_embed.proj.weight":  # Conv3d [out, C, T, P, P] -> Linear [out, C*T*P*P]
                    v = v.reshape(v.shape[0], -1)
                weights[k] = v.astype(dtype)
    tower.load_weights(list(weights.items()), strict=True)
    mx.eval(tower.parameters())
    tower.eval()
    return tower


__all__ = ["MRoPE", "VisionTower", "load_vision_tower", "mrope_positions"]
