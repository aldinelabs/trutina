"""Chunked gated delta rule for *training* Qwen3.5 in MLX.

mlx-lm's fast GatedDeltaNet is a Metal kernel without a vjp; in training mode it falls
back to a per-token Python loop whose backward pass keeps a [B, H, Dv, Dk] fp32 state
for every token (~16 GB per layer at B=8, T=1000). This is the chunkwise (WY / UT
transform) form used by flash-linear-attention: within a chunk of C tokens everything is
matmuls, and only one state per chunk is kept, so memory drops ~C-fold and autodiff
works through plain ops.

The unit lower-triangular system (I + A) X = R of each chunk is inverted by block
divide and conquer (log2 C levels of small matmuls).

`enable_chunked_training()` routes mlx-lm's Qwen3.5 training-mode calls here; inference
(eval mode) keeps the Metal kernel.
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn
from mlx_lm.models import qwen3_5

_original_update = qwen3_5.gated_delta_update


def _diag_blocks(x: mx.array, size: int) -> mx.array:
    """[..., C, C] -> its C/size diagonal blocks [..., C/size, size, size]."""
    n = x.shape[-1] // size
    x = x.reshape(*x.shape[:-2], n, size, n, size)
    return mx.moveaxis(mx.diagonal(x, axis1=-4, axis2=-2), -1, -3)


def _block_diag(blocks: mx.array) -> mx.array:
    """[..., n, s, s] -> block-diagonal [..., n*s, n*s]."""
    n, s = blocks.shape[-3], blocks.shape[-1]
    eye = mx.eye(n, dtype=blocks.dtype)
    full = blocks[..., :, :, None, :] * eye[:, None, :, None]   # [..., n, s, n, s]
    return full.reshape(*blocks.shape[:-3], n * s, n * s)


def unit_lower_inverse(a: mx.array) -> mx.array:
    """Inverse of I + a for strictly lower-triangular a [..., C, C] (C a power of two).

    Divide and conquer: inv([[A, 0], [B, D]]) = [[A^-1, 0], [-D^-1 B A^-1, D^-1]], merging
    diagonal blocks of size 1, 2, 4, ... C/2. Every intermediate is a true sub-inverse, so
    it is as stable as forward substitution (a Neumann series is not: its terms grow like
    binomial coefficients when neighbouring keys are similar).
    """
    c = a.shape[-1]
    lower = a + mx.eye(c, dtype=a.dtype)
    inv = mx.broadcast_to(mx.ones((1, 1), a.dtype), (*a.shape[:-2], c, 1, 1))  # 1x1 blocks
    s = 1
    while s < c:
        n = c // (2 * s)
        pairs = inv.reshape(*inv.shape[:-3], n, 2, s, s)
        ai, di = pairs[..., 0, :, :], pairs[..., 1, :, :]
        b = _diag_blocks(lower, 2 * s)[..., s:, :s]
        top = mx.concatenate([ai, mx.zeros_like(ai)], axis=-1)
        bottom = mx.concatenate([-(di @ b @ ai), di], axis=-1)
        inv = mx.concatenate([top, bottom], axis=-2)              # [..., n, 2s, 2s]
        s *= 2
    return inv[..., 0, :, :]


def chunked_gated_delta(q: mx.array, k: mx.array, v: mx.array, log_g: mx.array, beta: mx.array,
                        state: mx.array | None = None, chunk: int = 64) -> tuple[mx.array, mx.array]:
    """Gated delta rule over a whole sequence.

    q, k [B, T, Hk, Dk] (q already carries the readout scale), v [B, T, Hv, Dv],
    log_g, beta [B, T, Hv], state [B, Hv, Dv, Dk] (mlx-lm layout).
    Returns y [B, T, Hv, Dv] in q's dtype and the final state (fp32).
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    out_dtype = q.dtype
    if (r := Hv // Hk) > 1:
        q, k = mx.repeat(q, r, -2), mx.repeat(k, r, -2)
    pad = (-T) % chunk
    N = (T + pad) // chunk

    def blocks(x):  # [B, T, H, ...] -> [B, H, N, C, ...] fp32, zero-padded
        x = x.astype(mx.float32)
        if pad:
            x = mx.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2))
        x = mx.moveaxis(x, 2, 1)
        return x.reshape(B, Hv, N, chunk, *x.shape[3:])

    q, k, v, log_g, beta = (blocks(x) for x in (q, k, v, log_g, beta))
    k_beta, v_beta = k * beta[..., None], v * beta[..., None]
    cum = mx.cumsum(log_g, axis=-1)                                   # [B, H, N, C]
    lower = mx.tril(mx.ones((chunk, chunk), dtype=mx.bool_))
    decay = mx.exp(mx.where(lower, cum[..., :, None] - cum[..., None, :], -mx.inf))
    kt = mx.swapaxes(k, -1, -2)

    strict = mx.where(mx.tril(lower, -1), (k_beta @ kt) * decay, 0.0)
    inv = unit_lower_inverse(strict)
    u = inv @ v_beta                                                  # [B, H, N, C, Dv]
    w = inv @ (k_beta * mx.exp(cum)[..., None])                       # [B, H, N, C, Dk]
    attn = (q @ kt) * decay                                           # [B, H, N, C, C]
    q_dec = q * mx.exp(cum)[..., None]
    k_dec = k * mx.exp(cum[..., -1:] - cum)[..., None]
    chunk_decay = mx.exp(cum[..., -1])[..., None, None]               # [B, H, N, 1, 1]

    s = (mx.zeros((B, Hv, Dk, Dv), mx.float32) if state is None
         else mx.swapaxes(state.astype(mx.float32), -1, -2))          # [B, H, Dk, Dv]
    outs = []
    for i in range(N):
        v_new = u[:, :, i] - w[:, :, i] @ s
        outs.append(q_dec[:, :, i] @ s + attn[:, :, i] @ v_new)
        s = s * chunk_decay[:, :, i] + mx.swapaxes(k_dec[:, :, i], -1, -2) @ v_new
    y = mx.stack(outs, axis=2).reshape(B, Hv, N * chunk, Dv)[:, :, :T]
    return mx.moveaxis(y, 1, 2).astype(out_dtype), mx.swapaxes(s, -1, -2)


def gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, *,
                       use_kernel: bool = True, lower_bound=None, allow_neg_eigval: bool = False):
    """Drop-in for mlx-lm's `gated_delta_update`: chunked form for unmasked training calls."""
    if use_kernel or mask is not None or lower_bound is not None:
        return _original_update(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=use_kernel,
                                lower_bound=lower_bound, allow_neg_eigval=allow_neg_eigval)
    beta = mx.sigmoid(b) * (2.0 if allow_neg_eigval else 1.0)
    log_g = -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)
    return chunked_gated_delta(q, k, v, log_g, beta, state)


def enable_chunked_training() -> None:
    qwen3_5.gated_delta_update = gated_delta_update
