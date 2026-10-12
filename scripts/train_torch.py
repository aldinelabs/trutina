"""Train the multi-exit decider with PyTorch on CUDA (e.g. Colab); resumes from --out.

  pip install -e ".[torch]" flash-linear-attention   # fast GatedDeltaNet kernels
  python scripts/train_torch.py --data train.jsonl --val val.jsonl --out runs/exp2 --steps 4000

Same flags and run directory as `scripts/train.py`; `trainable.safetensors` loads into the
MLX model with `trutina.train.load_trainable`.
"""

from __future__ import annotations

import argparse
import json
import signal

import torch

from trutina.data import load_all
from trutina.pt.train import build_model, load_torso, train
from trutina.runs import (
    RunDir,
    add_config_args,
    augment,
    config_from_args,
    encode_all,
    source_vocab,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--data", required=True, nargs="+", help="one or more training JSONL files")
    ap.add_argument("--val", default=None, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--vision", action=argparse.BooleanOptionalAction, default=True,
                    help="load the vision tower (needed when the data has images)")
    ap.add_argument("--stop-at", type=int, default=None, help="end this session after N steps")
    add_config_args(ap)
    a = ap.parse_args()
    cfg = config_from_args(a)
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    gpu = torch.cuda.get_device_name() if a.device.startswith("cuda") and torch.cuda.is_available() else None
    run = RunDir(a.out, {"framework": "torch", "device": a.device, "gpu": gpu, "model": a.model,
                         "data": a.data, "val": a.val}, cfg)
    run.status("loading", 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # cuDNN attention builds a plan per new sequence length (~10 ms CPU each); lengths vary
    # every step here, so use the flash / memory-efficient kernels instead
    torch.backends.cuda.enable_cudnn_sdp(False)
    torso, tok, vision = load_torso(a.model, vision=a.vision)
    model = build_model(torso, cfg, a.device, vision)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run.log(f"device {a.device}, trainable params {n_train / 1e6:.1f}M")
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
    train(model, train_items, val_items, cfg, run, a.device, stop_at=a.stop_at)


if __name__ == "__main__":
    main()
