"""Evaluate a run trained with scripts/train_torch.py on CUDA (same report as scripts/evaluate.py).

  python scripts/evaluate_torch.py --run runs/v1 --data test.jsonl [--weights best]

Per exit: accuracy, Brier, ECE, NLL, score MAE and soft-label metrics by question type,
plus the adaptive-depth sweep. Writes <run>/eval-<data name>-<weights>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from trutina.data import load_all
from trutina.evaluation import adaptive_sweep, summarize
from trutina.pt.train import build_model, collate, load_torso
from trutina.runs import TrainConfig, encode_all, eval_batches


def weights_path(run: Path, which: str) -> Path:
    return {"best": run / "best.safetensors", "final": run / "trainable.safetensors"}.get(which, Path(which))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--name", default=None, help="report name (default: first data file's stem)")
    ap.add_argument("--weights", default="best", help="best | final | path")
    ap.add_argument("--batch-tokens", type=int, default=65536)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    run = Path(a.run)
    config = json.loads((run / "config.json").read_text())
    cfg = TrainConfig(**{**config["train"], "grad_checkpointing": False})
    torch.backends.cuda.enable_cudnn_sdp(False)
    torso, tok, vision = load_torso(config["model"], vision=True)
    model = build_model(torso, cfg, a.device, vision)
    state = load_file(str(weights_path(run, a.weights)))
    res = model.load_state_dict({k: v.to(a.device) for k, v in state.items()}, strict=False)
    trainable = {k for k, p in model.named_parameters() if p.requires_grad}
    if res.unexpected_keys or trainable & set(res.missing_keys):
        raise SystemExit(f"weights do not match the model: {res}")
    model.eval()

    examples = load_all(a.data)
    items, _ = encode_all(examples, tok, None)  # source 0 (unknown), as in deployment
    per_exit: list[list[np.ndarray]] = [[] for _ in model.exits]
    targets, kinds = [], []
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.device == "cuda"):
        for chunk in eval_batches(items, 64, a.batch_tokens):
            b, _ = collate(chunk, a.device)
            for k, lg in enumerate(model(b)):
                p = torch.softmax(lg.float(), -1).cpu().numpy()
                per_exit[k] += [p[r, : len(it.target)] for r, it in enumerate(chunk)]
            targets += [it.target for it in chunk]
            kinds += [it.kind for it in chunk]
    gold = [int(np.argmax(t)) for t in targets]
    exits = list(model.exits)
    report = {"run": str(run), "weights": a.weights, "data": a.data, "questions": len(gold),
              "seconds": time.time() - t0,
              "chance": float(np.mean([1 / len(p) for p in per_exit[0]])),
              "exits": {f"exit{e}": summarize(per_exit[k], targets, kinds) for k, e in enumerate(exits)},
              "adaptive": adaptive_sweep(per_exit, gold, exits)}
    name = a.name or Path(a.data[0]).stem
    out = run / f"eval-{name}-{a.weights if a.weights in ('best', 'final') else 'custom'}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"{len(gold)} questions in {report['seconds']:.0f}s -> {out}")
    for e, r in report["exits"].items():
        print(f"{e}: " + "  ".join(f"{k} acc {v['acc']:.3f} ece {v['ece']:.3f}"
                                   + (f" clear {v['clear_acc']:.3f} kl {v['kl_soft']:.3f}" if "kl_soft" in v else "")
                                   for k, v in r.items()))


if __name__ == "__main__":
    main()
