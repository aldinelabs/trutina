"""Framework-free training plumbing shared by the MLX and torch loops.

- `TrainConfig` and its CLI flags
- `encode_all` / `step_batches`: pre-tokenised items and an endless, deterministic stream of
  optimizer steps (so a resumed run replays the same data). Each step holds
  `batch_size * grad_accum` questions of similar length, split into micro-batches either
  of `batch_size` questions or, with `batch_tokens`, of at most that many padded tokens
  (long states then get small micro-batches instead of being dropped).
- `RunDir`: the run directory contract the web UI reads::

    config.json     {"train": TrainConfig, ...launch args}
    status.json     {"state": loading|running|stopped|done|failed, "step", "steps", "examples",
                     "pid", "updated"}
    metrics.jsonl   {"type": "train", "step", "loss", "lr", "tok_s", "mem_gb", "time"}
                    {"type": "eval", "step", "exits": {"exit16": {"acc", "loss", ...}}, "time"}
    train.log       human-readable log
    trainable.safetensors   final export (MLX-loadable)
    best.safetensors + best.json   weights with the lowest validation loss so far
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from trutina.data import Example
from trutina.metrics import ece
from trutina.prompt import Encoded, Tokenizer, encode_question

# behaviour of options for runs started before the option existed
LEGACY = {"smooth_soft": True}


@dataclass
class TrainConfig:
    steps: int = 1000
    lr: float = 1e-4
    warmup: int = 50
    batch_size: int = 8
    grad_accum: int = 1
    rank: int = 16
    lora_scale: float = 2.0
    exits: list[int] = field(default_factory=lambda: [16, 24, 32])
    exit_weights: list[float] | None = None
    head_dim: int = 512
    max_tokens: int | None = 2048
    batch_tokens: int | None = None
    grad_checkpointing: bool = False
    eval_every: int = 250
    ckpt_every: int = 100
    seed: int = 0
    label_smoothing: float = 0.05
    smooth_soft: bool = False  # also smooth teacher distributions (pre-v1 behaviour)
    self_distill: float = 0.0  # weight of KL(deepest exit || shallower exits)
    permute_choices: bool = False  # see `augment`
    drop_descriptions: float = 0.0
    read_state: bool = False      # head cross-attends to the state tokens (trutina.head)
    ordinal_score: bool = False   # discretised-logistic score parameterisation
    source_tokens: int = 0        # label-source embedding slots (0 = off; slot 0 = unknown)
    source_dropout: float = 0.2   # share of training questions whose source is hidden
    group_alpha: float = 1.0      # flatten sub-domains within each dataset (see `rebalance`)

    def head_kwargs(self) -> dict:
        return {"read_state": self.read_state, "n_sources": self.source_tokens,
                "ordinal_score": self.ordinal_score}

    def loss_kwargs(self) -> dict:
        return {"eps": self.label_smoothing, "smooth_soft": self.smooth_soft}

    def lr_factor(self, step: int) -> float:
        """Linear warmup then cosine decay to 0; `step` is 0-based."""
        if step < self.warmup:
            return (step + 1) / self.warmup
        return 0.5 * (1 + math.cos(math.pi * (step - self.warmup) / max(1, self.steps - self.warmup)))


def add_config_args(ap: argparse.ArgumentParser) -> None:
    for f in dataclasses.fields(TrainConfig):
        default = f.default if f.default is not dataclasses.MISSING else f.default_factory()
        name = "--" + f.name.replace("_", "-")
        if isinstance(default, bool):
            ap.add_argument(name, action=argparse.BooleanOptionalAction, default=default)
        elif f.name in ("exits", "exit_weights"):
            ap.add_argument(name, type=int if f.name == "exits" else float, nargs="+", default=default)
        else:
            ap.add_argument(name, type=type(default) if default is not None else int, default=default)


def config_from_args(a: argparse.Namespace) -> TrainConfig:
    return TrainConfig(**{f.name: getattr(a, f.name) for f in dataclasses.fields(TrainConfig)})


@dataclass
class Item:
    enc: Encoded
    kind: str
    target: np.ndarray
    source: int = 0
    dataset: str = ""  # for per-dataset validation metrics


def rebalance(examples: Sequence[Example], alpha: float, rng: random.Random) -> list[Example]:
    """Within each dataset (the part of `group` before "/"), resample its sub-domains so their
    shares are ∝ n ** alpha. Dataset totals are kept (the mix across datasets is set when the
    data is built); alpha < 1 flattens towards small sub-domains, duplicating some examples."""
    by_ds: dict[str, dict[str, list[Example]]] = {}
    for e in examples:
        by_ds.setdefault(e.group.split("/")[0], {}).setdefault(e.group, []).append(e)
    out = []
    for groups in by_ds.values():
        size = sum(len(v) for v in groups.values())
        weights = {g: len(v) ** alpha for g, v in groups.items()}
        total = sum(weights.values())
        for g, v in groups.items():
            n = round(size * weights[g] / total)
            out += rng.sample(v, n) if n <= len(v) else v + rng.choices(v, k=n - len(v))
    rng.shuffle(out)
    return out


def augment(examples: Sequence[Example], cfg: TrainConfig) -> list[Example]:
    """Training-time augmentation, fixed per example (seeded by `cfg.seed`):
    - `permute_choices`: shuffle the options of choice questions (and their target), so the
      backbone cannot lean on option position. Score levels keep their order.
    - `drop_descriptions`: with this probability, a choice / score question loses its option
      descriptions (real requests often give bare labels)."""
    rng = random.Random(cfg.seed + 1)
    if cfg.group_alpha != 1.0:
        examples = rebalance(examples, cfg.group_alpha, rng)
    out = []
    for e in examples:
        q, t = e.question, e.target
        if cfg.permute_choices and q.kind == "choice":
            perm = list(range(len(q.options)))
            rng.shuffle(perm)
            q = dataclasses.replace(q, options=tuple(q.options[i] for i in perm),
                                    descriptions=tuple(q.resolved_descriptions()[i] for i in perm))
            t = t[perm]
        if cfg.drop_descriptions and q.kind != "noul" and rng.random() < cfg.drop_descriptions:
            q = dataclasses.replace(q, descriptions=())
        src = "" if cfg.source_tokens and rng.random() < cfg.source_dropout else e.source
        out.append(e if q is e.question and src == e.source else Example(e.state, q, t, src, e.group))
    return out


def source_vocab(examples: Sequence[Example], slots: int) -> dict[str, int]:
    """Stable ids 1.. for the label sources in `examples` (0 = unknown)."""
    names = sorted({e.source for e in examples if e.source})
    if len(names) >= slots:
        raise ValueError(f"{len(names)} label sources need --source-tokens > {len(names)}: {names}")
    return {n: i + 1 for i, n in enumerate(names)}


def encode_all(examples: Sequence[Example], tok: Tokenizer, max_tokens: int | None,
               sources: dict[str, int] | None = None) -> tuple[list[Item], int]:
    """`sources` maps label sources to ids; without it every item gets 0 (unknown), which is
    also what validation and inference use."""
    items, dropped = [], 0
    for e in examples:
        enc = encode_question(tok, e.state, e.question)
        if max_tokens and len(enc.tokens) > max_tokens:
            dropped += 1
            continue
        items.append(Item(enc, e.question.kind, e.target, (sources or {}).get(e.source, 0),
                          e.group.split("/")[0]))
    return items, dropped


def padded_targets(items: Sequence[Item]) -> np.ndarray:
    K = max(len(i.target) for i in items)
    t = np.zeros((len(items), K), dtype=np.float32)
    for r, i in enumerate(items):
        t[r, : len(i.target)] = i.target
    return t


def split_micro(group: Sequence[Item], batch_size: int, batch_tokens: int | None) -> list[list[Item]]:
    """Split a length-sorted group into micro-batches of `batch_size` items or, with
    `batch_tokens`, of at most that many padded tokens (always at least one item)."""
    if not batch_tokens:
        return [list(group[i : i + batch_size]) for i in range(0, len(group), batch_size)]
    out: list[list[Item]] = []
    cur: list[Item] = []
    longest = 0
    for it in group:
        n = max(longest, len(it.enc.tokens))
        if cur and n * (len(cur) + 1) > batch_tokens:
            out.append(cur)
            cur, n = [], len(it.enc.tokens)
        cur.append(it)
        longest = n
    out.append(cur)
    return out


def step_batches(items: Sequence[Item], cfg: TrainConfig, rng: random.Random,
                 window: int = 64) -> Iterator[list[list[Item]]]:
    """Endless optimizer steps, each a list of micro-batches.

    Shuffle, sort by length inside windows of `window` steps, cut into steps of
    `batch_size * grad_accum` questions, shuffle the steps, then split each into micro-batches.
    """
    per_step = cfg.batch_size * cfg.grad_accum
    order = list(range(len(items)))
    while True:
        rng.shuffle(order)
        groups = []
        for s in range(0, len(order), per_step * window):
            w = sorted(order[s : s + per_step * window], key=lambda i: len(items[i].enc.tokens))
            groups += [w[j : j + per_step] for j in range(0, len(w) - per_step + 1, per_step)]
        rng.shuffle(groups)
        for g in groups:
            yield split_micro([items[i] for i in g], cfg.batch_size, cfg.batch_tokens)


def eval_batches(items: Sequence[Item], batch_size: int,
                 batch_tokens: int | None = None) -> Iterator[list[Item]]:
    ordered = sorted(items, key=lambda i: len(i.enc.tokens))
    yield from split_micro(ordered, batch_size, batch_tokens)


class EvalAccumulator:
    """Collects per-exit probabilities / losses and reports acc, loss, Brier, ECE, confidence,
    KL(target || p) (meaningful for teacher distributions), accuracy and ECE on long inputs
    (more than `long_tokens` tokens), which a mostly short validation set would otherwise hide,
    and accuracy per dataset (`by_dataset`)."""

    def __init__(self, exits: Sequence[int], long_tokens: int = 1024):
        self.exits = list(exits)
        self.long_tokens = long_tokens
        self.long: list[bool] = []
        self.probs: list[list[np.ndarray]] = [[] for _ in exits]
        self.losses: list[list[float]] = [[] for _ in exits]
        self.gold: list[int] = []
        self.targets: list[np.ndarray] = []
        self.datasets: list[str] = []

    def add_targets(self, items: Sequence[Item]) -> None:
        self.targets += [i.target for i in items]
        self.datasets += [i.dataset for i in items]
        self.gold += [int(np.argmax(i.target)) for i in items]
        self.long += [len(i.enc.tokens) > self.long_tokens for i in items]

    def add(self, k: int, probs: np.ndarray, losses: np.ndarray, items: Sequence[Item]) -> None:
        self.probs[k] += [probs[r, : len(i.target)] for r, i in enumerate(items)]
        self.losses[k] += losses.tolist()

    def result(self) -> dict:
        out = {}
        for k, e in enumerate(self.exits):
            p = self.probs[k]
            out[f"exit{e}"] = {
                "acc": float(np.mean([np.argmax(x) == g for x, g in zip(p, self.gold)])),
                "loss": float(np.mean(self.losses[k])),
                "brier": float(np.mean([np.sum((x - t) ** 2) for x, t in zip(p, self.targets)])),
                "ece": ece(p, self.gold),
                "mean_conf": float(np.mean([x.max() for x in p])),
                "kl": float(np.mean([np.sum(t * (np.log(np.maximum(t, 1e-12)) - np.log(np.maximum(x, 1e-12))))
                                     for x, t in zip(p, self.targets)])),
            }
            by: dict[str, list[bool]] = {}
            for x, g, ds in zip(p, self.gold, self.datasets):
                if ds:
                    by.setdefault(ds, []).append(bool(np.argmax(x) == g))
            if by:
                out[f"exit{e}"]["by_dataset"] = {ds: {"acc": float(np.mean(v)), "n": len(v)}
                                                 for ds, v in sorted(by.items())}
            longs = [i for i, flag in enumerate(self.long) if flag]
            if longs:
                lp, lg = [p[i] for i in longs], [self.gold[i] for i in longs]
                out[f"exit{e}"].update(
                    long_acc=float(np.mean([np.argmax(x) == g for x, g in zip(lp, lg)])),
                    long_ece=ece(lp, lg), long_n=len(longs))
        return out


class RunDir:
    def __init__(self, out: str | Path, launch: dict, cfg: TrainConfig):
        self.path = Path(out)
        self.path.mkdir(parents=True, exist_ok=True)
        conf = self.path / "config.json"
        stored = json.loads(conf.read_text())["train"] if conf.exists() else None
        if stored is not None:  # options added since the run started keep their old behaviour
            for f in dataclasses.fields(cfg):
                if f.name not in stored:
                    setattr(cfg, f.name, LEGACY.get(f.name, getattr(cfg, f.name)))
        train = dataclasses.asdict(cfg)
        if stored is not None and {k: train.get(k) for k in stored} != stored:
            raise SystemExit(f"{conf} differs from these arguments; use a new --out")
        conf.write_text(json.dumps({**launch, "train": train}, indent=2))
        self.steps = cfg.steps
        self.examples: int | None = None  # training questions, for epoch accounting

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        self._append("train.log", line)

    def record(self, kind: str, step: int, **data) -> None:
        self._append("metrics.jsonl", json.dumps({"type": kind, "step": step, **data, "time": time.time()}))

    def _append(self, name: str, line: str) -> None:
        # open / append / close per line: synced volumes (e.g. Modal) only publish closed files
        with (self.path / name).open("a") as f:
            f.write(line + "\n")

    @staticmethod
    def val_score(res: dict) -> float:
        """Selection score for a validation result: mean decision loss over exits (lower is
        better; covers calibration too, unlike accuracy)."""
        return float(np.mean([v["loss"] for v in res.values()]))

    def is_best(self, res: dict) -> bool:
        best = self.path / "best.json"
        return not best.exists() or self.val_score(res) < json.loads(best.read_text())["score"]

    def mark_best(self, step: int, res: dict) -> None:
        """Call after writing best.safetensors."""
        tmp = self.path / "best.json.tmp"
        tmp.write_text(json.dumps({"step": step, "score": self.val_score(res), "exits": res}))
        os.replace(tmp, self.path / "best.json")
        self.log(f"best so far (val loss {self.val_score(res):.4f}) -> best.safetensors")

    def status(self, state: str, step: int) -> None:
        tmp = self.path / "status.json.tmp"
        tmp.write_text(json.dumps({"state": state, "step": step, "steps": self.steps,
                                   "examples": self.examples, "pid": os.getpid(),
                                   "updated": time.time()}))
        os.replace(tmp, self.path / "status.json")
