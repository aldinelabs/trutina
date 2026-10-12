"""Torch mirror of `trutina.head` + `trutina.decider` (training forward only).

The torso is a Hugging Face `Qwen3_5TextModel`; layers past the last exit are dropped,
and hidden states of the mixed layers are captured with forward hooks. With `vision` (the
multimodal `Qwen3_5Model` wrapping the torso), image placeholders are filled with the frozen
vision tower's features and positions follow Qwen3.5's multimodal RoPE.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from trutina.prompt import Encoded
from trutina.vision import IMAGE_PAD, pixel_patches


def rms(h: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    h = h.float()
    return h * torch.rsqrt((h * h).mean(-1, keepdim=True) + eps)


class ScalarMix(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n))
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, hs: list[torch.Tensor]) -> torch.Tensor:
        a = torch.softmax(self.weights, 0)
        return self.gamma * sum(a[i] * rms(h) for i, h in enumerate(hs))


class MultiHeadAttention(nn.Module):  # mlx.nn.MultiHeadAttention (no biases)
    def __init__(self, dims: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.query_proj = nn.Linear(dims, dims, bias=False)
        self.key_proj = nn.Linear(dims, dims, bias=False)
        self.value_proj = nn.Linear(dims, dims, bias=False)
        self.out_proj = nn.Linear(dims, dims, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, kv: torch.Tensor | None = None) -> torch.Tensor:
        kv = x if kv is None else kv
        B, S, D = x.shape
        q = self.query_proj(x).view(B, S, self.num_heads, -1).transpose(1, 2)
        k, v = (p(kv).view(B, kv.shape[1], self.num_heads, -1).transpose(1, 2)
                for p in (self.key_proj, self.value_proj))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out_proj(y.transpose(1, 2).reshape(B, S, D))


class TransformerEncoderLayer(nn.Module):  # mlx.nn.TransformerEncoderLayer, norm_first, relu
    def __init__(self, dims: int, num_heads: int, mlp_dims: int):
        super().__init__()
        self.attention = MultiHeadAttention(dims, num_heads)
        self.ln1 = nn.LayerNorm(dims, eps=1e-5)
        self.ln2 = nn.LayerNorm(dims, eps=1e-5)
        self.linear1 = nn.Linear(dims, mlp_dims)
        self.linear2 = nn.Linear(mlp_dims, dims)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x), mask)
        return x + self.linear2(F.relu(self.linear1(self.ln2(x))))


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, dims: int, num_heads: int, mlp_dims: int):
        super().__init__()
        self.layers = nn.ModuleList(TransformerEncoderLayer(dims, num_heads, mlp_dims)
                                    for _ in range(num_layers))
        self.ln = nn.LayerNorm(dims, eps=1e-5)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.ln(x)


def ordinal_log_probs(mu: torch.Tensor, scale: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """See `trutina.head.ordinal_log_probs`."""
    K = mask.shape[1]
    k = torch.arange(K, dtype=torch.float32, device=mask.device)[None]
    last = (mask.sum(-1, keepdim=True) - 1).float()
    cdf = torch.where(k < last, torch.sigmoid((k + 0.5 - mu[:, None]) / scale[:, None]), 1.0)
    prev = torch.cat([torch.zeros_like(cdf[:, :1]), cdf[:, :-1]], 1)
    return torch.where(mask, torch.log((cdf - prev).clamp(min=1e-6)), -1e9)


class JointHead(nn.Module):
    def __init__(self, hidden: int, dim: int = 512, layers: int = 2, heads: int = 8, n_exits: int = 3,
                 read_state: bool = False, n_sources: int = 0, ordinal_score: bool = False):
        super().__init__()
        if read_state:
            self.mem_proj = nn.Linear(hidden, dim)
            self.read_ln = nn.LayerNorm(dim, eps=1e-5)
            self.read = MultiHeadAttention(dim, heads)
        if n_sources:
            self.source = nn.Embedding(n_sources, dim)
        if ordinal_score:
            self.ordinal = nn.Linear(dim, 2)
        self.proj = nn.Linear(hidden, dim)
        self.role = nn.Embedding(2, dim)
        self.exit = nn.Embedding(n_exits, dim)
        self.encoder = TransformerEncoder(layers, dim, heads, mlp_dims=2 * dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.bias = nn.Linear(dim, 1)
        self.scale = dim**-0.5

    def forward(self, h_ans: torch.Tensor, h_opt: torch.Tensor, opt_mask: torch.Tensor,
                exit_idx: int, mem: torch.Tensor | None = None, mem_mask: torch.Tensor | None = None,
                source: torch.Tensor | None = None, is_score: torch.Tensor | None = None) -> torch.Tensor:
        B, K, _ = h_opt.shape
        x = self.proj(torch.cat([h_ans[:, None], h_opt], 1))
        roles = torch.ones(B, K + 1, dtype=torch.long, device=x.device)
        roles[:, 0] = 0
        x = x + self.role(roles) + self.exit.weight[exit_idx]
        if hasattr(self, "source"):
            src = self.source(source if source is not None else torch.zeros(B, dtype=torch.long, device=x.device))
            x = torch.cat([x[:, :1] + src[:, None], x[:, 1:]], 1)
        if hasattr(self, "read"):
            m = self.mem_proj(mem)
            x = x + self.read(self.read_ln(x), torch.where(mem_mask, 0.0, -1e9)[:, None, None, :], m)
        keep = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device), opt_mask], 1)
        attn_mask = torch.where(keep, 0.0, -1e9)[:, None, None, :]
        y = self.encoder(x, attn_mask)
        a, o = y[:, 0], y[:, 1:]
        logits = (o @ self.q(a)[:, :, None]).squeeze(-1) * self.scale + self.bias(o).squeeze(-1)
        if hasattr(self, "ordinal") and is_score is not None:
            loc, sc = self.ordinal(a).float().unbind(-1)
            last = (opt_mask.sum(-1) - 1).float()
            ordl = ordinal_log_probs(torch.sigmoid(loc) * last, F.softplus(sc) + 0.05, opt_mask)
            logits = torch.where(is_score[:, None], logits + ordl, logits)
        return torch.where(opt_mask, logits, -1e9)


@dataclass
class DecisionBatch:
    tokens: torch.Tensor       # [B, T] right-padded
    answer_pos: torch.Tensor   # [B]
    option_pos: torch.Tensor   # [B, K]
    option_mask: torch.Tensor  # [B, K] bool
    is_score: torch.Tensor     # [B] bool
    source: torch.Tensor | None = None          # [B] label-source ids (training only)
    pixel_values: torch.Tensor | None = None    # [patches, 3*2*16*16], images in row order
    image_grid_thw: torch.Tensor | None = None  # [images, 3]

    @classmethod
    def build(cls, items: Sequence[Encoded], kinds: Sequence[str], pad_id: int = 0,
              device: torch.device | str | None = None,
              buckets: Sequence[int] | None = None, sources: Sequence[int] | None = None) -> DecisionBatch:
        """`buckets`: pad to the smallest listed length that fits, so GPU kernels (compiled and
        tuned per shape, ~20 s each on first sight) only ever see a handful of shapes."""
        T = max(len(e.tokens) for e in items)
        if buckets:
            T = next((n for n in sorted(buckets) if n >= T), T)
        K = max(len(e.option_pos) for e in items)

        def t(x, **kw):
            return torch.tensor(x, device=device, **kw)

        grids = [g for e in items for g in e.grids]
        pixels = None
        if grids:
            patches = []
            for e in items:
                for ref, g in zip(e.images, e.grids):
                    with ref.open() as img:
                        patches.append(pixel_patches(img, g))
            pixels = torch.from_numpy(np.concatenate(patches)).to(device)

        return cls(
            source=t(list(sources)) if sources is not None else None,
            pixel_values=pixels,
            image_grid_thw=t(grids) if grids else None,
            tokens=t([e.tokens + [pad_id] * (T - len(e.tokens)) for e in items]),
            answer_pos=t([e.answer_pos for e in items]),
            option_pos=t([e.option_pos + [0] * (K - len(e.option_pos)) for e in items]),
            option_mask=t([[True] * len(e.option_pos) + [False] * (K - len(e.option_pos))
                           for e in items]),
            is_score=t([k == "score" for k in kinds]),
        )


class MultiExitDecider(nn.Module):
    def __init__(self, torso: nn.Module, hidden: int, exits: Sequence[int] = (16, 24, 32),
                 mix_every: int = 4, head_dim: int = 512, head_layers: int = 2, head_heads: int = 8,
                 vision: nn.Module | None = None, read_state: bool = False, n_sources: int = 0,
                 ordinal_score: bool = False):
        super().__init__()
        self._vision = [vision] if vision is not None else []  # not a submodule: frozen, not exported
        self.exits = tuple(exits)
        torso.layers = torso.layers[: self.exits[-1]]  # never run past the last exit
        self.torso = torso
        self.mix_layers = tuple(sorted(set(range(mix_every, self.exits[-1] + 1, mix_every)) | set(exits)))
        self.mixes = nn.ModuleList(ScalarMix(len(self._below(e))) for e in self.exits)
        self.head = JointHead(hidden, head_dim, head_layers, head_heads, n_exits=len(self.exits),
                              read_state=read_state, n_sources=n_sources, ordinal_score=ordinal_score)
        self._captured: dict[int, torch.Tensor] = {}
        for l in self.mix_layers:
            torso.layers[l - 1].register_forward_hook(self._capture(l))

    def _capture(self, l: int):
        def hook(_module, _args, out):
            self._captured[l] = out[0] if isinstance(out, tuple) else out
        return hook

    def _below(self, e: int) -> list[int]:
        return [l for l in self.mix_layers if l <= e]

    def hidden(self, b: DecisionBatch) -> dict[int, torch.Tensor]:
        """Hidden states [B, T, H] after each mixed layer (layer count -> tensor)."""
        self._captured = {}
        if b.pixel_values is None:
            self.torso(input_ids=b.tokens, use_cache=False)
        else:
            self.torso(inputs_embeds=self._embed_images(b), position_ids=self._mrope_positions(b),
                       use_cache=False)
        hs, self._captured = self._captured, {}
        return hs

    def exit_features(self, i: int, hs: dict[int, torch.Tensor]) -> torch.Tensor:
        return self.mixes[i]([hs[l] for l in self._below(self.exits[i])])

    def exit_logits(self, i: int, feats: torch.Tensor, b: DecisionBatch) -> torch.Tensor:
        idx = torch.arange(feats.shape[0], device=feats.device)
        h_ans = feats[idx, b.answer_pos]
        h_opt = feats[idx[:, None], b.option_pos]
        mem_mask = torch.arange(feats.shape[1], device=feats.device)[None] <= b.answer_pos[:, None]
        return self.head(h_ans, h_opt, b.option_mask, i, feats, mem_mask, b.source, b.is_score)

    def forward(self, b: DecisionBatch) -> list[torch.Tensor]:
        """Logits [B, K] (fp32) for every exit."""
        hs = self.hidden(b)
        return [self.exit_logits(i, self.exit_features(i, hs), b) for i in range(len(self.exits))]

    @property
    def vision(self) -> nn.Module:
        if not self._vision:
            raise ValueError("batch has images but the model was built without a vision tower")
        return self._vision[0]

    def _embed_images(self, b: DecisionBatch) -> torch.Tensor:
        emb = self.torso.embed_tokens(b.tokens)
        with torch.no_grad():  # frozen tower
            feats = self.vision.get_image_features(b.pixel_values, b.image_grid_thw, return_dict=True)
        feats = torch.cat(feats.pooler_output).to(emb.dtype)
        mask = (b.tokens == IMAGE_PAD)[..., None]
        if int(mask.sum()) != feats.shape[0]:
            raise ValueError(f"{int(mask.sum())} image tokens but {feats.shape[0]} image features")
        return emb.masked_scatter(mask, feats)

    def _mrope_positions(self, b: DecisionBatch) -> torch.Tensor:
        types = (b.tokens == IMAGE_PAD).int()
        pos, _ = self.vision.get_rope_index(b.tokens, mm_token_type_ids=types,
                                            image_grid_thw=b.image_grid_thw)
        return pos  # [3, B, T]; right padding only extends the text positions

    def trainable_state(self) -> dict[str, torch.Tensor]:
        """Trainable tensors under MLX-compatible names (adapters + mixes + head)."""
        return {k: p.detach().float().cpu().contiguous()
                for k, p in self.named_parameters() if p.requires_grad}
