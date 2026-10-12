"""Train the multi-exit decider with MLX on Apple silicon; resumes from --out.

  uv run python scripts/train.py --data train.jsonl --val val.jsonl --out runs/exp2 --steps 4000

Ctrl-C / SIGTERM checkpoints and exits; rerun the same command to resume. Writes the run
directory described in `trutina.runs` (watch it with `scripts/webui.py`).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys

import mlx.core as mx

from trutina.backbone import load_backbone
from trutina.data import load_all
from trutina.decider import MultiExitDecider
from trutina.runs import (
    RunDir,
    add_config_args,
    augment,
    config_from_args,
    encode_all,
    source_vocab,
)
from trutina.train import add_lora, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--data", required=True, nargs="+", help="one or more training JSONL files")
    ap.add_argument("--val", default=None, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stop-at", type=int, default=None, help="end this session after N steps")
    ap.add_argument("--keep-awake", action=argparse.BooleanOptionalAction, default=True,
                    help="on macOS, hold a caffeinate assertion (no idle/system sleep) while training")
    ap.add_argument("--cache-limit-gb", type=float, default=4.0,
                    help="cap MLX's buffer cache so freed GPU memory returns to the system")
    add_config_args(ap)
    a = ap.parse_args()
    cfg = config_from_args(a)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    mx.set_cache_limit(int(a.cache_limit_gb * 2**30))
    if a.keep_awake and sys.platform == "darwin":  # released automatically when we exit
        subprocess.Popen(["caffeinate", "-ims", "-w", str(os.getpid())],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    run = RunDir(a.out, {"framework": "mlx", "model": a.model, "data": a.data, "val": a.val}, cfg)
    run.status("loading", 0)
    torso, tok = load_backbone(a.model)
    model = MultiExitDecider(torso, hidden=torso.embed_tokens.weight.shape[1], exits=cfg.exits,
                             head_dim=cfg.head_dim, **cfg.head_kwargs())
    add_lora(model, rank=cfg.rank, scale=cfg.lora_scale)
    examples = load_all(a.data)
    sources = source_vocab(examples, cfg.source_tokens) if cfg.source_tokens else None
    if sources:
        (run.path / "sources.json").write_text(json.dumps(sources, indent=2))
    train_items, dropped = encode_all(augment(examples, cfg), tok, cfg.max_tokens, sources)
    run.log(f"train: {len(train_items)} questions ({dropped} over {cfg.max_tokens} tokens dropped)")
    val_items = []
    if a.val:
        val_items, dropped = encode_all(load_all(a.val), tok, cfg.max_tokens)
        run.log(f"val: {len(val_items)} questions ({dropped} dropped)")
    train(model, train_items, val_items, cfg, run, stop_at=a.stop_at)


if __name__ == "__main__":
    main()
