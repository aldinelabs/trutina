"""Torch training loop for the multi-exit decider (CUDA; resumable for preemptible machines).

Mirrors `trutina.train` (same config, batch stream, run directory and export format):
frozen bf16 torso + LoRA on layers [0, last exit), fp32 mixes and joint head, mean of
per-exit decision losses. Adds optional activation checkpointing.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from trutina.pt.lora import add_lora
from trutina.pt.losses import decision_loss, masked_log_softmax, soft_ce
from trutina.pt.model import DecisionBatch, MultiExitDecider
from trutina.runs import (
    EvalAccumulator,
    Item,
    RunDir,
    TrainConfig,
    eval_batches,
    padded_targets,
    step_batches,
)


class HFTokenizer:
    def __init__(self, tok):
        self.tok = tok

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)


def load_torso(path: str, dtype: torch.dtype = torch.bfloat16, attn: str = "sdpa", vision: bool = False):
    """(Qwen3_5TextModel, tokenizer, vision) from a HF checkpoint. `vision` is the frozen
    multimodal `Qwen3_5Model` (vision tower + torso) when requested, else None; the LM head
    and MTP weights are dropped."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    auto = AutoModelForImageTextToText if vision else AutoModelForCausalLM
    lm = auto.from_pretrained(path, dtype=dtype, attn_implementation=attn)
    inner = lm.model
    torso = getattr(inner, "language_model", inner)
    for p in inner.parameters():
        p.requires_grad_(False)
    return torso, HFTokenizer(AutoTokenizer.from_pretrained(path)), (inner if vision else None)


# padded lengths on CUDA: 1.5x geometric steps (<= ~33% padding, typically ~15%)
LENGTH_BUCKETS = (64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)


def collate(items: Sequence[Item], device) -> tuple[DecisionBatch, torch.Tensor]:
    buckets = LENGTH_BUCKETS if torch.device(device).type == "cuda" else None
    b = DecisionBatch.build([i.enc for i in items], [i.kind for i in items], device=device,
                            buckets=buckets, sources=[i.source for i in items])
    return b, torch.from_numpy(padded_targets(items)).to(device)


def multi_exit_loss(model: MultiExitDecider, b: DecisionBatch, t: torch.Tensor,
                    exit_weights: Sequence[float], denom: float | None = None,
                    loss_kw: dict | None = None, distill: float = 0.0) -> torch.Tensor:
    """See `trutina.train.multi_exit_loss`."""
    logits = model(b)
    total = sum(w * decision_loss(lg, t, b.option_mask, b.is_score, **(loss_kw or {})).sum()
                for w, lg in zip(exit_weights, logits))
    if distill:
        teacher = masked_log_softmax(logits[-1], b.option_mask).exp().detach()
        total = total + distill * sum(soft_ce(lg, teacher, b.option_mask).sum() for lg in logits[:-1])
    return total / (sum(exit_weights) * (denom or b.tokens.shape[0]))


@torch.no_grad()
def evaluate(model: MultiExitDecider, items: Sequence[Item], batch_size: int, device,
             batch_tokens: int | None = None, loss_kw: dict | None = None) -> dict:
    model.eval()
    acc = EvalAccumulator(model.exits)
    for chunk in eval_batches(items, batch_size, batch_tokens):
        b, t = collate(chunk, device)
        acc.add_targets(chunk)
        for k, lg in enumerate(model(b)):
            loss = decision_loss(lg, t, b.option_mask, b.is_score, **(loss_kw or {}))
            acc.add(k, torch.softmax(lg, -1).cpu().numpy(), loss.cpu().numpy(), chunk)
    model.train()
    return acc.result()


def build_model(torso, cfg: TrainConfig, device, vision=None) -> MultiExitDecider:
    if cfg.grad_checkpointing:
        torso.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = MultiExitDecider(torso, hidden=torso.config.hidden_size, exits=cfg.exits,
                             head_dim=cfg.head_dim, vision=vision, **cfg.head_kwargs())
    if vision is not None:
        vision.visual.to(device)
    add_lora(model.torso.layers, rank=cfg.rank, scale=cfg.lora_scale)
    model.to(device)
    for p in model.parameters():  # trainable state in fp32, frozen torso stays bf16
        if p.requires_grad:
            p.data = p.data.float()
    return model


def train(model: MultiExitDecider, train_items: Sequence[Item], val_items: Sequence[Item],
          cfg: TrainConfig, run: RunDir, device, stop_at: int | None = None) -> bool:
    """Train to `cfg.steps`, resuming from run/ckpt.pt. Returns True when complete (and
    exported); False when stopped early (`stop_at`, Ctrl-C or SIGTERM)."""
    out = run.path
    run.examples = len(train_items)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cfg.lr_factor)
    exit_weights = cfg.exit_weights or [1.0] * len(model.exits)
    data = step_batches(train_items, cfg, random.Random(cfg.seed))

    ckpt, start = out / "ckpt.pt", 0
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        res = model.load_state_dict(state["trainable"], strict=False)
        trainable = {k for k, p in model.named_parameters() if p.requires_grad}
        if res.unexpected_keys or trainable & set(res.missing_keys):
            raise RuntimeError(f"checkpoint does not match the model: {res}")
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start = state["step"]
        for _ in range(start):  # replay the deterministic batch stream
            next(data)
        run.log(f"resumed from step {start}")

    def save(step: int) -> None:
        trainable = {k: p.detach() for k, p in model.named_parameters() if p.requires_grad}
        tmp = ckpt.with_suffix(".tmp")
        torch.save({"trainable": trainable, "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step}, tmp)
        os.replace(tmp, ckpt)

    model.train()
    completed = start
    run.status("running", completed)
    losses: list[float] = []
    norms: list[float] = []
    tokens_seen = 0  # padded tokens processed in this session
    toks, t0 = 0, time.time()
    try:
        for step in range(start, cfg.steps):
            micro = next(data)
            n_q, step_loss = sum(len(m) for m in micro), 0.0
            for chunk in micro:
                b, t = collate(chunk, device)
                loss = multi_exit_loss(model, b, t, exit_weights, n_q, cfg.loss_kwargs(), cfg.self_distill)
                loss.backward()
                step_loss += loss.item()
                toks += b.tokens.numel()
                tokens_seen += b.tokens.numel()
            losses.append(step_loss)
            norms.append(float(torch.nn.utils.clip_grad_norm_(params, 1.0)))
            lr = sched.get_last_lr()[0]
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            done = completed = step + 1

            if done % 10 == 0 or done == cfg.steps:
                dt = time.time() - t0
                mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
                rec = {"loss": float(np.mean(losses)), "lr": lr, "tok_s": toks / dt, "mem_gb": mem,
                       "grad_norm": float(np.mean(norms)), "tokens": tokens_seen}
                run.record("train", done, **rec)
                run.log(f"step {done}: loss {rec['loss']:.4f} lr {lr:.2e} "
                        f"{rec['tok_s']:.0f} tok/s {mem:.1f}GiB peak")
                losses, norms, toks, t0 = [], [], 0, time.time()
                run.status("running", done)
            if done % cfg.ckpt_every == 0 or done in (cfg.steps, stop_at):
                save(done)
            if val_items and (done % cfg.eval_every == 0 or done == cfg.steps):
                res = evaluate(model, val_items, cfg.batch_size * 2, device,
                               cfg.batch_tokens and cfg.batch_tokens * 2, cfg.loss_kwargs())
                run.record("eval", done, exits=res)
                if run.is_best(res):
                    export(model, out / "best.tmp.safetensors")
                    os.replace(out / "best.tmp.safetensors", out / "best.safetensors")
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
        save(completed)
        run.log(f"interrupted; checkpoint at step {completed}")
        run.status("stopped", completed)
        return False
    except BaseException:
        run.status("failed", completed)
        raise

    export(model, out / "trainable.safetensors")
    run.log(f"exported {out / 'trainable.safetensors'}")
    run.status("done", cfg.steps)
    return True


def export(model: MultiExitDecider, path: Path) -> None:
    """MLX-loadable weights (`trutina.train.load_trainable`)."""
    save_file(model.trainable_state(), str(path))
