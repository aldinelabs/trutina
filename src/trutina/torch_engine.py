"""trutina-4b with PyTorch + transformers on CUDA or ARM CPU.

  from trutina.torch_engine import TorchTrutina
  model = TorchTrutina.load("aldinelabs/trutina-4b", device="cuda")
  model.decide({"state": "...", "questions": {...}})

Same prompt, readout, exits and calibration as the MLX engine (`trutina.engine`). Depth: `shallow`
answers at layer 12, `adaptive` walks the exits (12, 16, 20, 24) and stops at the first whose top
probability reaches the threshold for every question of the batch, continuing from that layer's
hidden states otherwise; `deep` runs all 32 layers (without the MLX engine's option rotations).
`noul="decisive"` (default when the release ships the map) pushes leaning yes/no probabilities out of
the 0.2-0.8 band, as in the MLX engine; `noul="calibrated"` reports the calibrated probability.
Questions are length-sorted into padded batches under a token budget; a batch that runs out of GPU
memory is split in half.
With `cuda_graphs` (default), batches of up to `graph_tokens` (2,048) padded tokens replay CUDA graphs, one per
exit-to-exit segment and (batch, length bucket): short requests are bound by kernel launches, not
FLOPs. Sequences are right-padded to the bucket without an attention mask; the model is causal and
each row is read at its own last token, so the padding never reaches what is read.
Install the CUDA kernels for speed: flash-linear-attention and causal-conv1d.
On ARM CPU, cpu_int8=True quantizes decoder Linear layers with QNNPACK; vision=True
retains the floating-point vision tower. CUDA graphs remain text-only.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from trutina import wide
from trutina.api import RequestError, handle
from trutina.readout import FORMAT_VERSION, decisive, full_text, prompt, wide_labels
from trutina.schema import Question, Request, State


class TorchTrutina:
    def __init__(self, torso, tok, config: dict, readout: dict, device: str, batch_tokens: int,
                 cuda_graphs: bool = True, graph_tokens: int = 2048):
        self.torso, self.tok, self.config, self.device = torso, tok, config, device
        self.exits = config["exits"][:-1]
        temps = [config["temperature"][str(e)] for e in config["exits"]]
        scales = [*np.exp(readout["lens_logscale"]), np.exp(readout["prior_logscale"])]
        self.scales = [float(sc) / t for sc, t in zip(scales, temps)]  # logit scale / temperature, per exit
        self.lens = []
        for i in range(len(self.exits)):
            lin = torch.nn.Linear(readout[f"lens.{i}.weight"].shape[1], readout[f"lens.{i}.weight"].shape[0])
            lin.weight.data = torch.from_numpy(readout[f"lens.{i}.weight"])
            lin.bias.data = torch.from_numpy(readout[f"lens.{i}.bias"])
            self.lens.append(lin.to(device).float().eval())
        self.threshold = config["adaptive_threshold"]
        ids = torch.tensor(wide_labels(tok)[1][: wide.PASS], device=device)
        self.rows = torso.embed_tokens.weight[ids].float()  # tied LM head: the option letters' rows
        self.batch_tokens = batch_tokens
        self.depth = "adaptive"
        self.noul = "decisive" if "noul_decisive" in config else "calibrated"
        self.last_exit: int | None = None
        self.graph_tokens = graph_tokens if cuda_graphs and str(device).startswith("cuda") else 0
        self.graphs: dict[tuple[int, int], _Chain] = {}

    @classmethod
    def load(cls, path: str = "aldinelabs/trutina-4b", device: str = "cuda", dtype=torch.bfloat16,
             batch_tokens: int = 32768, cuda_graphs: bool = True, graph_tokens: int = 2048,
             cpu_int8: bool = False, vision: bool = False,
             npu_matmul: str | None = None) -> TorchTrutina:
        from safetensors import safe_open
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

        if cpu_int8 and torch.device(device).type != "cpu":
            raise ValueError("cpu_int8 requires device='cpu'")
        if npu_matmul and (cpu_int8 or torch.device(device).type != "cpu"):
            raise ValueError("npu_matmul requires device='cpu' and cpu_int8=False")

        local = Path(path)
        if not local.exists():
            from huggingface_hub import snapshot_download

            local = Path(snapshot_download(path))
        config = json.loads((local / "trutina.json").read_text())
        if config["format_version"] != FORMAT_VERSION:
            raise ValueError(f"release format {config['format_version']}, this package reads {FORMAT_VERSION}")
        auto = AutoModelForImageTextToText if vision else AutoModelForCausalLM
        lm = auto.from_pretrained(str(local), dtype=dtype, attn_implementation="sdpa")
        torso = getattr(lm.model, "language_model", lm.model).to(device).eval()
        if cpu_int8:
            from trutina.cpu_quant import quantize_torso

            quantize_torso(torso)
        runtime = None
        if npu_matmul:
            from trutina.npu_matmul import MatmulRuntime, offload_torso

            runtime = MatmulRuntime(npu_matmul)
            offload_torso(torso, runtime)
        if vision:
            lm.model.visual.to(device).float().eval()
        with safe_open(str(local / "readout.safetensors"), "np") as f:
            readout = {k: f.get_tensor(k).astype(np.float32) for k in f.keys()}  # noqa: SIM118 (safe_open)
        engine = cls(torso, AutoTokenizer.from_pretrained(str(local)), config, readout, device, batch_tokens,
                     cuda_graphs, graph_tokens)
        engine.vision = lm.model if vision else None
        engine.npu_runtime = runtime
        if runtime is not None:
            engine.last_backend = "npu-matmul"
        if cpu_int8 or npu_matmul:
            from trutina.cpu_quant import release_cpu_memory

            del lm
            release_cpu_memory()
        return engine

    def decide(self, body: dict) -> dict:
        return handle(body, self)

    def close(self):
        """Release weights before another board backend is loaded."""
        self.graphs.clear()
        self.torso = None
        self.vision = None
        self.rows = None
        self.lens.clear()
        runtime = getattr(self, "npu_runtime", None)
        if runtime is not None:
            runtime.close()
            self.npu_runtime = None
        if torch.device(self.device).type == "cpu":
            from trutina.cpu_quant import release_cpu_memory

            release_cpu_memory()

    def __call__(self, req: Request) -> dict[str, np.ndarray]:
        if (not isinstance(req.state, str) and any(not isinstance(p, str) for p in req.state)
                and getattr(self, "vision", None) is None):
            raise RequestError("the torch engine needs vision=True (server: --vision) for images")
        jobs: list[tuple[int, list[int] | None]] = []
        qs: list[Question] = []
        for qi, q in enumerate(req.questions):
            k = len(q.resolved_options())
            for g in wide.groups(k) if q.kind == "choice" and k > wide.PASS else [None]:
                jobs.append((qi, g))
                qs.append(q if g is None else _subset(q, g))
        self.depth = req.depth
        probs = self._forward(req.state, qs)
        res, parts = {}, {}
        for (qi, g), p in zip(jobs, probs):
            if g is None:
                res[req.questions[qi].id] = p
            else:
                parts.setdefault(qi, []).append((g, p))
        if parts:
            fins = {qi: wide.finalists(ps) for qi, ps in parts.items()}
            self.depth = "deep"
            finals = self._forward(req.state, [_subset(req.questions[qi], f) for qi, f in fins.items()])
            for (qi, f), pf in zip(fins.items(), finals):
                q = req.questions[qi]
                res[q.id] = wide.combine(len(q.options), parts[qi], f, pf)
        if self.noul == "decisive":
            t, b = self.config["noul_decisive"]["T"], self.config["noul_decisive"]["b"]
            for q in req.questions:
                if q.kind == "noul":
                    y = decisive(float(res[q.id][1]), t, b)
                    res[q.id] = np.array([1 - y, y])
        return res

    def encode(self, state: State, q: Question) -> list[int]:
        text = full_text(self.tok, prompt(state, q, "sandwich", "letter"), True, "letter")
        return self.tok.encode(text, add_special_tokens=False)

    @torch.inference_mode()
    def _forward(self, state: State, qs: Sequence[Question]) -> list[np.ndarray]:
        if not isinstance(state, str) and any(not isinstance(p, str) for p in state):
            # Vision is intentionally sequential: keep board memory bounded and avoid
            # replaying text-only CUDA graphs with multimodal RoPE.
            return [self._read_image(state, q) for q in qs]
        seqs = [self.encode(state, q) for q in qs]
        ks = [len(q.resolved_options()) for q in qs]
        out: dict[int, np.ndarray] = {}
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        chunk: list[int] = []
        for i in [*order, None]:
            if i is not None and (not chunk or len(seqs[i]) * (len(chunk) + 1) <= self.batch_tokens):
                chunk.append(i)
                continue
            for j, p in zip(chunk, self._batch([seqs[j] for j in chunk], [ks[j] for j in chunk])):
                out[j] = p
            chunk = [i] if i is not None else []
        return [out[i] for i in range(len(seqs))]

    def _read_image(self, state: State, q: Question) -> np.ndarray:
        from trutina.vision import IMAGE_PAD, image_grid, n_image_tokens, pixel_patches

        refs = [p for p in state if not isinstance(p, str)]
        text = full_text(self.tok, prompt(state, q, "sandwich", "letter", images=True), True, "letter")
        tokens = self.tok.encode(text, add_special_tokens=False)
        grids, patches = [], []
        for ref in refs:
            with ref.open() as img:
                grid = image_grid(img.size)
                grids.append(grid)
                patches.append(pixel_patches(img, grid))
        if tokens.count(IMAGE_PAD) != len(refs):
            raise RequestError("Image count does not match prompt image markers")
        expanded, g = [], iter(grids)
        for t in tokens:
            expanded.extend([IMAGE_PAD] * n_image_tokens(next(g)) if t == IMAGE_PAD else [t])
        ids = torch.tensor([expanded], device=self.device)
        grid = torch.tensor(grids, device=self.device)
        pixels = torch.from_numpy(np.concatenate(patches)).to(self.device)
        feats = self.vision.get_image_features(pixels, grid, return_dict=True)
        feats = torch.cat(feats.pooler_output).to(self.torso.embed_tokens.weight.dtype)
        emb = self.torso.embed_tokens(ids)
        mask = (ids == IMAGE_PAD)[..., None]
        if int(mask.sum()) != feats.shape[0]:
            raise RequestError("Vision encoder returned the wrong number of features")
        emb = emb.masked_scatter(mask, feats)
        pos, _ = self.vision.get_rope_index(ids, mm_token_type_ids=(ids == IMAGE_PAD).int(),
                                           image_grid_thw=grid)
        h, kw = self._prepare(ids, torch.ones_like(ids), emb, pos)
        rows = torch.zeros(1, dtype=torch.long, device=self.device)
        last = torch.tensor([len(expanded)-1], device=self.device)
        return self._read_exits(h, kw, rows, last, [len(q.resolved_options())])[0]

    def _batch(self, seqs: list[list[int]], ks: list[int]) -> list[np.ndarray]:
        try:
            return self._read(seqs, ks)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(seqs) == 1:
                raise RequestError(f"a {len(seqs[0])}-token question does not fit in GPU memory") from None
            m = len(seqs) // 2
            return self._batch(seqs[:m], ks[:m]) + self._batch(seqs[m:], ks[m:])

    def _plan(self) -> list[tuple[int, float]]:
        return {"deep": [], "shallow": [(0, 0.0)]}.get(self.depth, [(i, self.threshold) for i in range(len(self.exits))])

    def _read(self, seqs: list[list[int]], ks: list[int]) -> list[np.ndarray]:
        """Run the layers segment by segment (exit to exit), reading the exits the depth asks for."""
        T = bucket(max(map(len, seqs)))
        if T * len(seqs) <= self.graph_tokens:
            return self._read_graphs(seqs, ks, T)
        T, dev = max(map(len, seqs)), self.device
        ids = torch.tensor([s + [0] * (T - len(s)) for s in seqs], device=dev)
        mask = torch.tensor([[1] * len(s) + [0] * (T - len(s)) for s in seqs], device=dev)
        rows = torch.arange(len(seqs), device=dev)
        last = torch.tensor([len(s) - 1 for s in seqs], device=dev)
        h, layer_kw = self._prepare(ids, mask)
        return self._read_exits(h, layer_kw, rows, last, ks)

    def _read_exits(self, h, layer_kw, rows, last, ks):
        torso = self.torso

        def probs(logits: torch.Tensor) -> list[np.ndarray]:
            return [torch.softmax(logits[r, :k], -1).cpu().double().numpy() for r, k in enumerate(ks)]

        start = 0
        for i, thr in self._plan():
            h = self._layers(h, start, self.exits[i], layer_kw)
            start = self.exits[i]
            x = h[rows, last].float()
            z = torso.norm(x + self.lens[i](x)).float()
            p = probs(z @ self.rows.T * self.scales[i])
            if all(q.max() >= thr for q in p):
                self.last_exit = start
                return p
        h = self._layers(h, start, len(torso.layers), layer_kw)
        self.last_exit = len(torso.layers)
        return probs(torso.norm(h[rows, last]).float() @ self.rows.T * self.scales[-1])

    def _prepare(self, ids: torch.Tensor, mask: torch.Tensor, embeds=None,
                 positions=None) -> tuple[torch.Tensor, dict]:
        """Embeddings, masks and rotary embeddings, as Qwen3_5TextModel.forward builds them."""
        torso = self.torso
        mod = sys.modules[type(torso).__module__]
        h = torso.embed_tokens(ids) if embeds is None else embeds
        text_pos = torch.arange(ids.shape[1], device=ids.device).expand(ids.shape[0], -1)
        pos = text_pos[None].expand(3, -1, -1) if positions is None else positions
        kw = {"config": torso.config, "inputs_embeds": h, "attention_mask": mask, "past_key_values": None,
              "position_ids": text_pos}
        masks = {"full_attention": mod.create_causal_mask(**kw),
                 "linear_attention": mod.create_recurrent_attention_mask(**kw)}
        return h, {"masks": masks, "position_embeddings": torso.rotary_emb(h, pos), "position_ids": text_pos}

    def _read_graphs(self, seqs: list[list[int]], ks: list[int], T: int) -> list[np.ndarray]:
        """`_read` through the captured segment graphs of this (batch, length bucket)."""
        chain = self.graphs.get((len(seqs), T)) or self._capture(len(seqs), T)
        chain.ids.copy_(torch.tensor([s + [0] * (T - len(s)) for s in seqs]))
        chain.last.copy_(torch.tensor([len(s) - 1 for s in seqs]))

        def probs(logits: torch.Tensor) -> list[np.ndarray]:
            z = logits.cpu().double().numpy()
            return [_softmax(z[r, :k]) for r, k in enumerate(ks)]

        chain.graphs[0].replay()
        done = 0  # segments replayed so far
        for i, thr in self._plan():
            chain.graphs[i + 1].replay()
            done = i + 1
            p = probs(chain.logits[i])
            if all(q.max() >= thr for q in p):
                self.last_exit = self.exits[i]
                return p
        for g in chain.graphs[done + 1:]:
            g.replay()
        self.last_exit = len(self.torso.layers)
        return probs(chain.logits[-1])

    def _capture(self, B: int, T: int) -> _Chain:
        """Warm up and capture the prepare step and every exit-to-exit segment for a [B, T] batch."""
        torso, dev = self.torso, self.device
        ids = torch.zeros(B, T, dtype=torch.long, device=dev)
        last = torch.full((B,), T - 1, dtype=torch.long, device=dev)
        rows = torch.arange(B, device=dev)
        bounds = [0, *self.exits, len(torso.layers)]
        # masks and rotary embeddings depend only on the shape when nothing is masked: build them once,
        # outside the graph (mask construction inspects values on the host)
        _, kw = self._prepare(ids, torch.ones_like(ids))
        state: dict = {"kw": kw, "rows": rows}  # the graphs read these addresses: keep them alive

        def prepare():
            state["h"] = torso.embed_tokens(ids)
            return state["h"]

        def segment(j: int):
            h = self._layers(state["h"], bounds[j], bounds[j + 1], state["kw"])
            state["h"] = h
            x = h[rows, last].float()
            z = torso.norm(x + self.lens[j](x)) if j < len(self.exits) else torso.norm(x)
            return z.float() @ self.rows.T * self.scales[j]

        steps = [prepare, *[lambda j=j: segment(j) for j in range(len(bounds) - 1)]]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # warm-up: Triton autotuning and cuBLAS workspaces before capture
            for _ in range(2):
                for f in steps:
                    f()
        torch.cuda.current_stream().wait_stream(side)
        # one memory pool per chain, every segment's output kept alive: replaying one bucket must never
        # write into memory another bucket's graphs read
        pool, graphs, logits, keep = torch.cuda.graph_pool_handle(), [], [], []
        for n, f in enumerate(steps):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                out = f()
            graphs.append(g)
            keep.append(state["h"])
            if n:
                logits.append(out)
        state["keep"] = keep
        chain = _Chain(ids, last, graphs, logits, state)
        self.graphs[(B, T)] = chain
        return chain

    def warm_graphs(self, max_tokens: int | None = None, batch: int = 1) -> int:
        """Capture every length bucket up to `max_tokens` for `batch` rows now, rather than on first use."""
        n = 0
        for T in BUCKETS:
            if T * batch <= min(max_tokens or self.graph_tokens, self.graph_tokens) and (batch, T) not in self.graphs:
                with torch.inference_mode():
                    self._capture(batch, T)
                n += 1
        return n

    def _layers(self, h: torch.Tensor, start: int, stop: int, kw: dict) -> torch.Tensor:
        types = self.torso.config.layer_types
        for i in range(start, stop):
            h = self.torso.layers[i](h, position_embeddings=kw["position_embeddings"],
                                     attention_mask=kw["masks"][types[i]], position_ids=kw["position_ids"],
                                     past_key_values=None, use_cache=False)
        return h


BUCKETS = [*range(64, 512, 64), *range(512, 1024, 128), *range(1024, 2048, 256), *range(2048, 8193, 512)]


def bucket(n: int) -> int:
    """The smallest length bucket that holds `n` tokens (at most ~12% padding); past the table, `n`."""
    return next((b for b in BUCKETS if b >= n), n)


@dataclasses.dataclass
class _Chain:
    ids: torch.Tensor          # static input [B, T]
    last: torch.Tensor         # static input [B]: each row's last token
    graphs: list               # prepare, then one graph per exit-to-exit segment
    logits: list               # each segment's static readout logits [B, PASS]
    state: dict                # keeps the captured hidden states alive


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def _subset(q: Question, idx: Sequence[int]) -> Question:
    d = q.resolved_descriptions()
    return dataclasses.replace(q, options=tuple(q.options[i] for i in idx), descriptions=tuple(d[i] for i in idx))
