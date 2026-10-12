"""Evaluate a trained multi-exit decider (MLX) on a labelled JSONL set.

  uv run python scripts/evaluate.py --run runs/exp2-v0 --data test.jsonl [--weights best]

Reports, per exit: accuracy, Brier, ECE, NLL, and score-question MAE (in levels), broken
down by question type; then the adaptive-depth trade-off (accuracy vs. mean exit layer)
for a sweep of confidence thresholds, simulated from the per-exit probabilities. Writes
<run>/eval-<data name>-<weights>.json (and, with --dump-probs, the per-question probabilities
to <run>/probs-<data name>-<weights>.npz for scripts/calibrate.py). Add --zero-shot to also score the frozen backbone
with LM label log-probs (the no-training baseline), before loading adapters.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from trutina.backbone import load_backbone
from trutina.data import load_examples
from trutina.decider import MultiExitDecider
from trutina.engines import LMScoringEngine
from trutina.evaluation import adaptive_sweep, summarize
from trutina.runs import TrainConfig, encode_all, eval_batches
from trutina.train import add_lora, load_trainable, to_batch


def weights_path(run: Path, which: str) -> Path:
    if which == "best":
        return run / "best.safetensors"
    if which == "final":
        return run / "trainable.safetensors"
    if which == "latest":
        return run / json.loads((run / "ckpt.json").read_text())["dir"] / "trainable.safetensors"
    return Path(which)


def pad(rows: list[np.ndarray]) -> np.ndarray:
    """[N, K] with zeros past each question's options (probabilities and targets alike)."""
    out = np.zeros((len(rows), max(len(r) for r in rows)), dtype=np.float32)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default="best", help="best | final | latest | path")
    ap.add_argument("--batch-tokens", type=int, default=16384)
    ap.add_argument("--zero-shot", action="store_true")
    ap.add_argument("--dump-probs", action="store_true")
    a = ap.parse_args()

    run = Path(a.run)
    config = json.loads((run / "config.json").read_text())
    tc = config["train"]
    wpath = weights_path(run, a.weights)
    torso, tok = load_backbone(config["model"])
    examples = load_examples(a.data)
    zero_shot = None
    if a.zero_shot:  # frozen backbone, before any adapters are attached
        engine = LMScoringEngine(torso, tok)
        zs = [engine.score_question(e.state, e.question) for e in examples]
        zero_shot = summarize(zs, [e.target for e in examples], [e.question.kind for e in examples])
    model = MultiExitDecider(torso, hidden=torso.embed_tokens.weight.shape[1], exits=tc["exits"],
                             head_dim=tc["head_dim"], **TrainConfig(**tc).head_kwargs())
    add_lora(model, rank=tc["rank"], scale=tc["lora_scale"])
    load_trainable(model, wpath)
    model.eval()

    items, _ = encode_all(examples, tok, None)
    exits = list(model.exits)
    per_exit: list[list[np.ndarray]] = [[] for _ in exits]
    targets, kinds = [], []
    t0 = time.time()
    for chunk in eval_batches(items, 64, a.batch_tokens):
        b, _ = to_batch(chunk)
        logits = model(b)
        mx.eval(logits)
        for k, lg in enumerate(logits):
            p = np.array(mx.softmax(lg, axis=-1))
            per_exit[k] += [p[r, : len(it.target)] for r, it in enumerate(chunk)]
        targets += [it.target for it in chunk]
        kinds += [it.kind for it in chunk]
    gold = [int(np.argmax(t)) for t in targets]
    elapsed = time.time() - t0

    report = {"run": str(run), "weights": str(wpath), "data": a.data, "questions": len(gold),
              "seconds": elapsed,
              "chance": float(np.mean([1 / len(p) for p in per_exit[0]])),
              "exits": {f"exit{e}": summarize(per_exit[k], targets, kinds) for k, e in enumerate(exits)},
              "adaptive": adaptive_sweep(per_exit, gold, exits)}

    if zero_shot:
        report["zero_shot_lm"] = zero_shot

    name = Path(a.data).stem
    tag = a.weights if a.weights in ("best", "final", "latest") else "custom"
    out = run / f"eval-{name}-{tag}.json"
    out.write_text(json.dumps(report, indent=2))
    if a.dump_probs:
        np.savez(run / f"probs-{name}-{tag}.npz", exits=np.array(exits), kinds=np.array(kinds),
                 n_options=np.array([len(t) for t in targets]),
                 targets=pad(targets), **{f"exit{e}": pad(per_exit[k]) for k, e in enumerate(exits)})
    print(f"{len(gold)} questions in {elapsed:.0f}s -> {out}")
    for e, r in report["exits"].items():
        print(f"{e}: " + "  ".join(f"{k} acc {v['acc']:.3f} ece {v['ece']:.3f}"
                                   + (f" clear {v['clear_acc']:.3f} kl {v['kl_soft']:.3f}" if "kl_soft" in v else "")
                                   for k, v in r.items()))
    print("adaptive:", " ".join(f"t={r['threshold']}: {r['acc']:.3f}@{r['mean_layers']:.1f}L"
                                for r in report["adaptive"]))


if __name__ == "__main__":
    main()
