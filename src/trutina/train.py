"""Training (MLX): LoRA on the torso + full training of the per-exit mixes and joint head.

Data format: see `trutina.data`; run-directory contract: see `trutina.runs`.

Qwen3.5's GatedDeltaNet layers run through `trutina.gdn` while training (the Metal
kernel has no backward pass). Resumable: every `ckpt_every` steps (and on Ctrl-C / SIGTERM) the trainable weights and
optimizer state go to `ckpt-<step>/`, committed by an atomic `ckpt.json` pointer. The
batch stream is deterministic, so a resumed run continues on the same data.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.trainer import grad_checkpoint

from trutina.data import Example, load_examples, target_vector
from trutina.decider import DecisionBatch, MultiExitDecider
from trutina.gdn import enable_chunked_training
from trutina.losses import decision_loss, masked_log_softmax, soft_ce
from trutina.runs import (
    EvalAccumulator,
    Item,
    RunDir,
    TrainConfig,
    eval_batches,
    padded_targets,
    step_batches,
)

__all__ = ["Example", "add_lora", "evaluate", "load_examples", "load_trainable", "save_trainable",
           "target_vector", "train"]


def add_lora(model: MultiExitDecider, rank: int = 16, scale: float = 2.0, dropout: float = 0.0) -> None:
    """Freeze the torso and wrap every Linear in layers [0, last exit) with LoRA.

    (mlx-lm's `linear_to_lora_layers` adapts the *last* N layers, which is wrong for
    early exits.) Parameter names match `trutina.pt.lora`, so torch-trained adapters load here.
    """
    model.torso.freeze()
    for layer in model.torso.layers[: model.exits[-1]]:
        lora = [(k, LoRALinear.from_base(m, r=rank, scale=scale, dropout=dropout))
                for k, m in layer.named_modules() if isinstance(m, nn.Linear)]
        layer.update_modules(tree_unflatten(lora))


def to_batch(items: Sequence[Item]) -> tuple[DecisionBatch, mx.array]:
    b = DecisionBatch.build([i.enc for i in items], [i.kind for i in items],
                            sources=[i.source for i in items])
    return b, mx.array(padded_targets(items))


def multi_exit_loss(model: MultiExitDecider, b: DecisionBatch, targets: mx.array,
                    exit_weights: Sequence[float], denom: float | None = None,
                    loss_kw: dict | None = None, distill: float = 0.0) -> mx.array:
    """Exit-weighted decision loss summed over the batch / `denom` (default: batch size),
    so micro-batches of different sizes add up to the per-question mean of the step.
    `distill` adds, for every exit but the deepest, KL(deepest || exit) with the deepest
    exit's distribution held fixed (exit self-distillation)."""
    logits = model(b)
    total = sum(w * decision_loss(lg, targets, b.option_mask, b.is_score, **(loss_kw or {})).sum()
                for w, lg in zip(exit_weights, logits))
    if distill:
        teacher = mx.stop_gradient(mx.exp(masked_log_softmax(logits[-1], b.option_mask)))
        total = total + distill * sum(soft_ce(lg, teacher, b.option_mask).sum() for lg in logits[:-1])
    return total / (sum(exit_weights) * (denom or b.tokens.shape[0]))


def evaluate(model: MultiExitDecider, items: Sequence[Item], batch_size: int,
             batch_tokens: int | None = None, loss_kw: dict | None = None) -> dict:
    was_training = model.training
    model.eval()  # fast Metal GatedDeltaNet kernel
    acc = EvalAccumulator(model.exits)
    for chunk in eval_batches(items, batch_size, batch_tokens):
        b, t = to_batch(chunk)
        acc.add_targets(chunk)
        for k, lg in enumerate(model(b)):
            loss = decision_loss(lg, t, b.option_mask, b.is_score, **(loss_kw or {}))
            acc.add(k, np.array(mx.softmax(lg, axis=-1)), np.array(loss), chunk)
    model.train(was_training)
    return acc.result()


def _save_ckpt(model: nn.Module, opt: optim.Optimizer, out: Path, step: int) -> None:
    d = out / f"ckpt-{step}"
    d.mkdir(exist_ok=True)
    mx.save_safetensors(str(d / "trainable.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    mx.save_safetensors(str(d / "optimizer.safetensors"), dict(tree_flatten(opt.state)))
    tmp = out / "ckpt.json.tmp"
    tmp.write_text(json.dumps({"step": step, "dir": d.name}))
    os.replace(tmp, out / "ckpt.json")
    for old in out.glob("ckpt-*"):
        if old != d:
            shutil.rmtree(old)


def _load_ckpt(model: nn.Module, opt: optim.Optimizer, out: Path) -> int:
    pointer = out / "ckpt.json"
    if not pointer.exists():
        return 0
    meta = json.loads(pointer.read_text())
    load_trainable(model, out / meta["dir"] / "trainable.safetensors")
    opt.init(model.trainable_parameters())
    opt.state = tree_unflatten(list(mx.load(str(out / meta["dir"] / "optimizer.safetensors")).items()))
    return meta["step"]


def train(model: MultiExitDecider, train_items: Sequence[Item], val_items: Sequence[Item],
          cfg: TrainConfig, run: RunDir, stop_at: int | None = None) -> bool:
    """Train to `cfg.steps`, resuming from the run directory. Returns True when complete
    (and exported); False when stopped early (`stop_at`, Ctrl-C or SIGTERM)."""
    out = run.path
    run.examples = len(train_items)
    enable_chunked_training()
    model.train()  # differentiable chunked GatedDeltaNet instead of the Metal kernel
    if cfg.grad_checkpointing:  # recompute each decoder layer in the backward pass
        grad_checkpoint(model.torso.layers[0])
    opt = optim.AdamW(learning_rate=cfg.lr, weight_decay=0.01)
    exit_weights = cfg.exit_weights or [1.0] * len(model.exits)
    loss_and_grad = nn.value_and_grad(
        model, lambda m, b, t, n: multi_exit_loss(m, b, t, exit_weights, n, cfg.loss_kwargs(),
                                                  cfg.self_distill))
    data = step_batches(train_items, cfg, random.Random(cfg.seed))

    start = _load_ckpt(model, opt, out)
    if start:
        for _ in range(start):  # replay the deterministic batch stream
            next(data)
        run.log(f"resumed from step {start}")

    completed = start  # optimizer steps applied so far
    run.status("running", completed)
    losses: list[float] = []
    toks, t0 = 0, time.time()
    try:
        for step in range(start, cfg.steps):
            grads, step_loss = None, 0.0
            micro = next(data)
            n_q = sum(len(m) for m in micro)
            for chunk in micro:
                b, t = to_batch(chunk)
                loss, g = loss_and_grad(model, b, t, n_q)
                grads = g if grads is None else tree_map(mx.add, grads, g)
                mx.eval(loss, grads)
                step_loss += loss.item()
                toks += b.tokens.size
            losses.append(step_loss)
            grads, _ = optim.clip_grad_norm(grads, 1.0)
            opt.learning_rate = cfg.lr * cfg.lr_factor(step)
            opt.update(model, grads)
            mx.eval(model.trainable_parameters(), opt.state)
            done = completed = step + 1

            if done % 10 == 0 or done == cfg.steps:
                dt = time.time() - t0
                rec = {"loss": float(np.mean(losses)), "lr": float(opt.learning_rate),
                       "tok_s": toks / dt, "mem_gb": mx.get_peak_memory() / 2**30}
                run.record("train", done, **rec)
                run.log(f"step {done}: loss {rec['loss']:.4f} lr {rec['lr']:.2e} "
                        f"{rec['tok_s']:.0f} tok/s {rec['mem_gb']:.1f}GiB peak")
                losses, toks, t0 = [], 0, time.time()
                run.status("running", done)
            if done % cfg.ckpt_every == 0 or done in (cfg.steps, stop_at):
                _save_ckpt(model, opt, out, done)
            if val_items and (done % cfg.eval_every == 0 or done == cfg.steps):
                res = evaluate(model, val_items, cfg.batch_size * 2,
                               cfg.batch_tokens and cfg.batch_tokens * 2, cfg.loss_kwargs())
                run.record("eval", done, exits=res)
                if run.is_best(res):
                    tmp = str(out / "best.tmp.safetensors")
                    mx.save_safetensors(tmp, dict(tree_flatten(model.trainable_parameters())))
                    os.replace(tmp, out / "best.safetensors")
                    run.mark_best(done, res)
                run.log(f"eval {done}: " + " ".join(
                    f"{k} acc {v['acc']:.3f} ece {v['ece']:.3f}"
                    + (f" long {v['long_acc']:.3f}" if "long_acc" in v else "") for k, v in res.items()))
                t0 = time.time()
            if done == stop_at and done < cfg.steps:
                run.log(f"stopped at step {done}; rerun to resume")
                run.status("stopped", done)
                return False
    except KeyboardInterrupt:
        _save_ckpt(model, opt, out, completed)
        run.log(f"interrupted; checkpoint at step {completed}")
        run.status("stopped", completed)
        return False
    except BaseException:
        run.status("failed", completed)
        raise

    save_trainable(model, out / "trainable.safetensors")
    run.log(f"exported {out / 'trainable.safetensors'}")
    run.status("done", cfg.steps)
    return True


def save_trainable(model: nn.Module, path: str | Path) -> None:
    mx.save_safetensors(str(path), dict(tree_flatten(model.trainable_parameters())))


def load_trainable(model: nn.Module, path: str | Path) -> None:
    """Load adapters + mixes + head (saved here or exported by `trutina.pt`); call after add_lora."""
    weights = mx.load(str(path))
    expected = {k for k, _ in tree_flatten(model.trainable_parameters())}
    if missing := expected - weights.keys():
        raise ValueError(f"{len(missing)} trainable weights missing, e.g. {sorted(missing)[:3]}")
    if extra := weights.keys() - expected:
        raise ValueError(f"{len(extra)} unexpected weights, e.g. {sorted(extra)[:3]}")
    model.load_weights(list(weights.items()), strict=False)
