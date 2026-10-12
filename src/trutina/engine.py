"""trutina-4b on Apple silicon (MLX): load a release and answer typed decision requests.

  from trutina import Trutina
  model = Trutina.load("aldinelabs/trutina-4b")       # a Hugging Face repo id or a local directory
  model.decide({"state": "...", "questions": {"q": {"type": "noul", "instructions": "..."}}})

A release directory (scripts/export_release.py) is a Qwen3.5-4B checkpoint with the fine-tune merged
in, plus `readout.safetensors` (tuned-lens exits) and `trutina.json` (prompt format, calibration).

Every question is one sequence: chat template with thinking off, the options labelled (A)..(Z) before
and after the input ("sandwich"), and the answer read from the LM-head rows of the option letters at
"Best answer: (". The questions of a request run as one right-padded batch. Choice questions with more
than 26 options go through a two-round tournament (`trutina.wide`).

Depth:
  shallow   answer at the first lens exit (layer 12)
  adaptive  walk the lens exits (12, 16, 20, 24); stop at the first whose max probability reaches the
            calibrated threshold for every question in the batch; otherwise the full 32 layers
  deep      the full 32 layers, averaged over up to 4 cyclic rotations of each choice question's options
Probabilities are temperature-calibrated per exit. `noul="decisive"` additionally pushes leaning yes/no
probabilities out of a 0.2-0.8 abstention band (for scorers that count such answers as abstentions,
e.g. JevBench v1.5); the default reports the calibrated probability. Images (OpenAI-style `input_image` parts in the
state) go through the Qwen3.5 vision tower, loaded on the first image.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx_lm import load as mlx_load

from trutina import wide
from trutina.api import handle
from trutina.backbone import forward_hidden
from trutina.readout import FORMAT_VERSION, encode_ids, full_text, is_wide, prompt, wide_labels
from trutina.schema import Question, Request, State
from trutina.vision import IMAGE_PAD, ImageRef, image_grid, n_image_tokens, pixel_patches

DEEP_ROTATIONS = 4


class ReadoutModel(nn.Module):
    """The torso plus the readout: final logits from the (tied) LM-head rows of the option letters,
    and one tuned-lens transform per early exit."""

    def __init__(self, torso: nn.Module, lens_exits: Sequence[int]):
        super().__init__()
        self.torso = torso
        hidden = torso.embed_tokens.weight.shape[1]
        self.prior_logscale = mx.zeros(())
        self.lens = [nn.Linear(hidden, hidden) for _ in lens_exits]
        self.lens_logscale = mx.zeros((len(lens_exits),))
        self.lens_exits = tuple(lens_exits)

    def label_rows(self, ids: mx.array) -> mx.array:
        return self.torso.embed_tokens.weight[ids].astype(mx.float32)

    def final_logits(self, h_normed: mx.array, w: mx.array) -> mx.array:
        return mx.exp(self.prior_logscale) * (h_normed.astype(mx.float32) @ w.T)

    def lens_logits(self, i: int, h_raw: mx.array, w: mx.array) -> mx.array:
        x = h_raw.astype(mx.float32)
        z = self.torso.norm(x + self.lens[i](x)).astype(mx.float32)
        return mx.exp(self.lens_logscale[i]) * (z @ w.T)


class Trutina:
    def __init__(self, model: ReadoutModel, tok, config: dict, path: str):
        self.model, self.tok, self.config, self.path = model, tok, config, path
        exits = [*model.lens_exits, len(model.torso.layers)]
        if config["exits"] != exits:
            raise ValueError(f"trutina.json is for exits {config['exits']}, the model has {exits}")
        self.temps = [config["temperature"][str(e)] for e in exits]
        self.threshold = config["adaptive_threshold"]
        self.label_ids = mx.array(wide_labels(tok)[1][: wide.PASS])
        self.noul = "calibrated"  # or "decisive"
        self.vision = None
        self.last_exit: int | None = None  # layer that answered the last batch (for adaptive stats)

    @classmethod
    def load(cls, path: str = "aldinelabs/trutina-4b") -> Trutina:
        local = Path(path)
        if not local.exists():
            from huggingface_hub import snapshot_download

            local = Path(snapshot_download(path))
        config = json.loads((local / "trutina.json").read_text())
        if config["format_version"] != FORMAT_VERSION:
            raise ValueError(f"release format {config['format_version']}, this package reads {FORMAT_VERSION}")
        lm, tok = mlx_load(str(local))
        model = ReadoutModel(lm.model, config["exits"][:-1])
        model.load_weights(str(local / "readout.safetensors"), strict=False)
        mx.eval(model.parameters())
        model.eval()
        return cls(model, tok, config, str(local))

    # ---- public API ----
    def decide(self, body: dict) -> dict:
        """A `/v1/systemone` request body in, the response body out (raises `trutina.api.RequestError`)."""
        return handle(body, self)

    def __call__(self, req: Request) -> dict[str, np.ndarray]:
        """{question id: probabilities in option order}."""
        rows: list[tuple[int, Question, list[int], int]] = []  # (question, shown question, option map, round)
        tourney: dict[int, list[list[int]]] = {}
        for qi, q in enumerate(req.questions):
            k = len(q.resolved_options())
            if q.kind == "choice" and k > wide.PASS:
                tourney[qi] = wide.groups(k)
                rows += [(qi, _subset(q, g), g, 1) for g in tourney[qi]]
                continue
            shifts = range(min(DEEP_ROTATIONS, k)) if req.depth == "deep" and q.kind == "choice" else [0]
            for s in shifts:  # cyclic rotations: every option visits several positions
                perm = [(j + s) % k for j in range(k)]
                rows.append((qi, q if s == 0 else _subset(q, perm), perm, 0))
        probs = self._run([r[1] for r in rows], req.state, req.depth)
        out: dict[int, list[np.ndarray]] = {}
        parts: dict[int, list[tuple[list[int], np.ndarray]]] = {}
        for (qi, _, perm, rnd), p in zip(rows, probs):
            if rnd:
                parts.setdefault(qi, []).append((perm, p))
                continue
            back = np.zeros(len(perm))
            back[perm] = p  # shown option j is original option perm[j]
            out.setdefault(qi, []).append(back)
        res = {req.questions[qi].id: np.mean(v, 0) for qi, v in out.items()}
        if self.noul == "decisive":
            t, b = self.config["noul_decisive"]["T"], self.config["noul_decisive"]["b"]
            for q in req.questions:
                if q.kind == "noul":
                    y = decisive(float(res[q.id][1]), t, b)
                    res[q.id] = np.array([1 - y, y])
        if parts:  # the tournament final, all wide questions in one batch
            fins = {qi: wide.finalists(ps) for qi, ps in parts.items()}
            finals = self._run([_subset(req.questions[qi], f) for qi, f in fins.items()], req.state, "deep")
            for (qi, f), pf in zip(fins.items(), finals):
                q = req.questions[qi]
                res[q.id] = wide.combine(len(q.options), parts[qi], f, pf)
        return res

    # ---- internals ----
    def encode(self, state: State, q: Question) -> tuple[list[int], list[ImageRef], list[tuple[int, int, int]]]:
        """(tokens, images, grids): the prompt tokens with every image pad expanded to its grid."""
        refs = [] if isinstance(state, str) else [p for p in state if not isinstance(p, str)]
        text = full_text(self.tok, prompt(state, q, "sandwich", "letter", images=bool(refs)), True, "letter")
        ids = encode_ids(self.tok, text)[0] if is_wide(q, "letter") else self.tok.encode(text, add_special_tokens=False)
        if not refs:
            return ids, [], []
        grids = []
        for r in refs:
            with r.open() as img:
                grids.append(image_grid(img.size))
        if ids.count(IMAGE_PAD) != len(refs):
            raise ValueError(f"{len(refs)} images but {ids.count(IMAGE_PAD)} image pads in the prompt")
        out, g = [], iter(grids)
        for t in ids:
            out += [IMAGE_PAD] * n_image_tokens(next(g)) if t == IMAGE_PAD else [t]
        return out, refs, grids

    def _image_inputs(self, tokens: mx.array, encs) -> tuple[mx.array, object]:
        """Embeddings with the image features filled in, and the multimodal rope for this batch."""
        from trutina.mlx_vision import MRoPE, load_vision_tower, mrope_positions

        if self.vision is None:
            self.vision = load_vision_tower(self.path)
        feats, memo = [], {}
        for _, refs, grids in encs:
            for r, g in zip(refs, grids):
                key = (r.source, r.base, g)
                if key not in memo:  # rotations and tournament groups repeat a request's images
                    with r.open() as img:
                        memo[key] = self.vision(mx.array(pixel_patches(img, g)), [g])
                feats.append(memo[key])
        torso = self.model.torso
        emb = torso.embed_tokens(tokens)
        B, T, H = emb.shape
        flat = np.nonzero(np.array(tokens).reshape(-1) == IMAGE_PAD)[0]
        emb = emb.reshape(B * T, H)
        emb[mx.array(flat)] = mx.concatenate(feats).astype(emb.dtype)
        pos = np.zeros((3, B, T), dtype=np.int64)
        for b, (ids, _, grids) in enumerate(encs):
            p = mrope_positions(ids, grids)
            pos[:, b, : len(ids)] = p
            pos[:, b, len(ids):] = p.max() + 1 + np.arange(T - len(ids))  # padding: never read
        rope = next(l.self_attn.rope for l in torso.layers if not l.is_linear)
        return emb.reshape(B, T, H), MRoPE(pos, rope.dims, rope.base)

    def _run(self, qs: Sequence[Question], state: State, depth: str) -> list[np.ndarray]:
        encs = [self.encode(state, q) for q in qs]
        seqs = [e[0] for e in encs]
        ks = [len(q.resolved_options()) for q in qs]
        T = max(map(len, seqs))
        tokens = mx.array([s + [0] * (T - len(s)) for s in seqs])
        embeds, rope = self._image_inputs(tokens, encs) if any(e[1] for e in encs) else (None, None)
        last = mx.array([len(s) - 1 for s in seqs])
        rows = mx.arange(len(seqs))
        K = max(ks)
        w = self.model.label_rows(self.label_ids[:K])
        mask = mx.array([[j < k for j in range(K)] for k in ks])

        def probs(logits: mx.array, e: int) -> np.ndarray:
            return np.array(mx.softmax(mx.where(mask, logits / self.temps[e], -1e9), axis=-1))

        exits = self.model.lens_exits
        plan = {"deep": [], "shallow": [(0, 0.0)]}.get(depth, [(i, self.threshold) for i in range(len(exits))])
        torso, start, h_raw = self.model.torso, 0, None
        for i, thr in plan:
            l = exits[i]
            _, caught = forward_hidden(torso, tokens, upto=l, collect=[l], start=start, h0=h_raw,
                                       embeds=embeds, rope=rope)
            h_raw, start = caught[l], l
            p = probs(self.model.lens_logits(i, h_raw[rows, last], w), i)
            if all(p[r, :k].max() >= thr for r, k in enumerate(ks)):
                self.last_exit = l
                return [p[r, :k] for r, k in enumerate(ks)]
        h, _ = forward_hidden(torso, tokens, start=start, h0=h_raw, embeds=embeds, rope=rope)
        p = probs(self.model.final_logits(h[rows, last], w), len(exits))
        self.last_exit = len(torso.layers)
        return [p[r, :k] for r, k in enumerate(ks)]


def decisive(p_yes: float, t: float, b: float) -> float:
    z = np.log(max(p_yes, 1e-9)) - np.log(max(1 - p_yes, 1e-9))
    return float(1 / (1 + np.exp(-(z - b) / t)))


def _subset(q: Question, idx: Sequence[int]) -> Question:
    d = q.resolved_descriptions()
    return dataclasses.replace(q, options=tuple(q.options[i] for i in idx), descriptions=tuple(d[i] for i in idx))
