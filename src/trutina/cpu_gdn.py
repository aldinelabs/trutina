"""Faster CPU prefill for Qwen3.5's Gated DeltaNet fallback in transformers.

Without the fla / causal-conv1d CUDA kernels, transformers runs the reference torch
implementation. On the ROCK 5B these parts dominate a short decision prompt:

- the chunked delta rule pads the sequence to a multiple of 64, so a 70-token
  prompt computes two chunks, mostly padding. Here the chunk size follows the
  prompt: the fewest chunks of at most 128 tokens, rounded up to 8.
- the depthwise causal conv runs through mkldnn conv1d. Here it is K shifted
  multiply-adds in the contiguous (tokens, channels) layout.
- the reference multiplies by transposed key views; on ARM a batched matmul with
  a non-contiguous operand was 3x slower than with a contiguous copy. Here the
  transposes are materialized once, and both triangular solves share one call.

Both are the same mathematics; outputs differ only by float summation order. On a
quantized path (W8A8 NPU, CPU INT8) that rounding difference can still move decision
probabilities by a few points, so this is opt-in (`fast_gdn=True`, `--fast-gdn`).
`enable()` patches the transformers module in place and is idempotent.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

MAX_CHUNK = 128


def chunk_length(length: int) -> int:
    chunks = math.ceil(length / MAX_CHUNK)
    return math.ceil(math.ceil(length / chunks) / 8) * 8


def causal_conv1d(hidden_states: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
                  activation: str | None = None, **kwargs) -> torch.Tensor:
    """hidden_states [B, C, L], weight [C, K] -> [B, C, L], as transformers' fallback."""
    from transformers.activations import ACT2FN

    width, length = weight.shape[-1], hidden_states.shape[-1]
    x = F.pad(hidden_states.to(weight.dtype).transpose(1, 2), (0, 0, width-1, 0))  # [B, L+K-1, C]
    taps = weight.t()
    out = x[:, :length] * taps[0]
    for j in range(1, width):
        out.addcmul_(x[:, j:j+length], taps[j])
    if bias is not None:
        out += bias
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype).transpose(1, 2)


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


def chunk_gated_delta_rule(query, key, value, g, beta, chunk_size: int | None = None,
                           initial_state=None, output_final_state: bool = False,
                           use_qk_l2norm_in_kernel: bool = False, **kwargs):
    """transformers' torch_chunk_gated_delta_rule with contiguous operands.

    query/key [B, L, H, Dk], value [B, L, H, Dv], g/beta [B, L, H] ->
    ([B, L, H, Dv], final state [B, H, Dk, Dv] or None).
    """
    dtype = query.dtype
    batch, length, heads, dk = key.shape
    dv = value.shape[-1]
    size = chunk_size or chunk_length(length)
    query, key, value, beta, g = (x.transpose(1, 2).to(torch.float32).contiguous()
                                  for x in (query, key, value, beta, g))
    if use_qk_l2norm_in_kernel:
        query, key = _l2norm(query), _l2norm(key)
    query = query * dk ** -0.5
    pad = (size - length % size) % size
    query, key, value = (F.pad(x, (0, 0, 0, pad)) for x in (query, key, value))
    beta, g = (F.pad(x, (0, pad)) for x in (beta, g))
    chunks = (length + pad) // size
    query, key, value = (x.reshape(batch, heads, chunks, size, -1) for x in (query, key, value))
    beta = beta.reshape(batch, heads, chunks, size, 1)
    decay = g.reshape(batch, heads, chunks, size).cumsum(dim=-1)
    upper = torch.ones(size, size, dtype=torch.bool).triu(1)
    pairwise = (decay.unsqueeze(-1) - decay.unsqueeze(-2)).masked_fill(upper, float("-inf")).exp()
    key_t = key.transpose(-1, -2).contiguous()
    k_beta = key * beta
    system = (k_beta @ key_t) * pairwise
    intra = (query @ key_t) * pairwise
    rhs = torch.cat([value * beta, k_beta * decay.exp().unsqueeze(-1)], dim=-1)
    solved = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
    new_values, k_cumdecay = solved[..., :dv], solved[..., dv:].contiguous()
    state = (torch.zeros(batch, heads, dk, dv) if initial_state is None
             else initial_state.to(torch.float32).clone())
    query = query * decay.exp().unsqueeze(-1)
    # Keys scaled by the decay from each position to its chunk end, stored transposed.
    key_t = (key * (decay[..., -1:] - decay).exp().unsqueeze(-1)).transpose(-1, -2).contiguous()
    chunk_decay = decay[..., -1].exp()[..., None, None]
    out = torch.empty(batch, heads, chunks, size, dv)
    for i in range(chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ state
        out[:, :, i] = query[:, :, i] @ state + intra[:, :, i] @ v_new
        state = state * chunk_decay[:, :, i] + key_t[:, :, i] @ v_new
    out = out.reshape(batch, heads, -1, dv)[:, :, :length].transpose(1, 2).contiguous().to(dtype)
    return out, (state if output_final_state else None)


def enable():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    if getattr(m, "_trutina_cpu_gdn", False):
        return
    m.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
    m.causal_conv1d_fn = causal_conv1d
    m._trutina_cpu_gdn = True
