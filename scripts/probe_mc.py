"""Experiment 0c: does early exit hold on knowledge & reasoning?

Per-layer linear probe for multiple choice (ARC-Challenge, MMLU). Each (question, option)
is encoded as "Question: ...\\nAnswer: <option>"; the last-token hidden state of every option
is centred by its question's mean (so the probe scores options relative to each other),
then a logistic probe is trained on "is this the correct option" and evaluated by argmax.
A zero-shot reference is reported too: length-normalised option log-prob from the LM head.

  uv run python scripts/probe_mc.py --model Qwen/Qwen3.5-4B-Base
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx import nn

from trutina.backbone import forward_hidden, load_backbone


def arc(split: str, n: int | None = None, seed: int = 0):
    rows = list(_arc_rows(split))
    if n and n < len(rows):
        rows = random.Random(seed).sample(rows, n)
    yield from rows


def _arc_rows(split: str):
    from datasets import load_dataset

    for cfg in (["ARC-Easy", "ARC-Challenge"] if split == "train" else ["ARC-Challenge"]):
        for ex in load_dataset("allenai/ai2_arc", cfg)[split]:
            labels = ex["choices"]["label"]
            if ex["answerKey"] in labels:
                yield ex["question"], ex["choices"]["text"], labels.index(ex["answerKey"])


def mmlu(split: str, n: int | None, seed: int = 0):
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all")
    rows = list(ds["validation"]) + list(ds["dev"]) if split == "train" else list(ds["test"])
    if n:
        rows = random.Random(seed).sample(rows, n)
    for ex in rows:
        yield ex["question"], ex["choices"], ex["answer"]


def encode(torso, tok, layers, items, want_lm: bool):
    """Returns per-layer centred option features, labels, question ids, zero-shot preds."""
    feats = {l: [] for l in layers}
    y, qid, zs_correct = [], [], []
    embed = torso.embed_tokens
    t0 = time.perf_counter()
    for qi, (q, opts, ans) in enumerate(items):
        pre = tok.encode(f"Question: {q.strip()}\nAnswer:")
        outs = [tok.encode(" " + str(o).strip()) or [0] for o in opts]
        seqs = [pre + o for o in outs]
        L = max(map(len, seqs))
        batch = mx.array([s + [s[-1]] * (L - len(s)) for s in seqs])
        final, hid = forward_hidden(torso, batch, collect=layers)
        last = mx.array([len(s) - 1 for s in seqs])
        for l in layers:
            h = mx.take_along_axis(hid[l], last[:, None, None], axis=1)[:, 0].astype(mx.float32)
            feats[l].append(np.array(h - h.mean(0, keepdims=True)))
        y += [int(i == ans) for i in range(len(opts))]
        qid += [qi] * len(opts)
        if qi % 250 == 0:
            print(f"    {qi} questions, {time.perf_counter() - t0:.0f}s", flush=True)
        if want_lm:
            # Project only the positions that predict option tokens (vocab is 248k).
            scores = []
            for i, o in enumerate(outs):
                h = final[i, len(pre) - 1 : len(pre) - 1 + len(o)]
                logp = nn.log_softmax(embed.as_linear(h).astype(mx.float32), axis=-1)
                scores.append(float(logp[mx.arange(len(o)), mx.array(o)].mean()))
            zs_correct.append(int(np.argmax(scores) == ans))
    print(f"  encoded {qi + 1} questions in {time.perf_counter() - t0:.0f}s")
    return ({l: np.concatenate(v) for l, v in feats.items()}, np.array(y), np.array(qid),
            float(np.mean(zs_correct)) if want_lm else None)


def probe_mc_acc(xtr, ytr, xte, yte, qte, steps: int = 400) -> float:
    sd = xtr.std(0) + 1e-5
    xtr, xte = mx.array(xtr / sd), mx.array(xte / sd)
    lin = nn.Linear(xtr.shape[1], 1)
    opt = optim.AdamW(1e-3, weight_decay=0.05)
    ytr_m = mx.array(ytr)

    def loss(m):
        return nn.losses.binary_cross_entropy(m(xtr)[:, 0], ytr_m, with_logits=True)

    step = nn.value_and_grad(lin, loss)
    for _ in range(steps):
        _, g = step(lin)
        opt.update(lin, g)
        mx.eval(lin.parameters(), opt.state)
    s = np.array(lin(xte)[:, 0])
    correct = [yte[qte == q][np.argmax(s[qte == q])] for q in np.unique(qte)]
    return float(np.mean(correct))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--mmlu-test", type=int, default=1500)
    ap.add_argument("--mmlu-train", type=int, default=None, help="subsample (default: all 1816)")
    ap.add_argument("--arc-train", type=int, default=None, help="subsample (default: all 3370)")
    ap.add_argument("--arc-test", type=int, default=None, help="subsample (default: all 1172)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    torso, tok = load_backbone(a.model)
    n = len(torso.layers)
    layers = sorted(set(range(8, n + 1, 4)) | {n // 2})
    report = {"model": a.model, "layers": layers}
    for name, tr, te in [
        ("arc_challenge", lambda: arc("train", a.arc_train), lambda: arc("test", a.arc_test)),
        ("mmlu", lambda: mmlu("train", a.mmlu_train), lambda: mmlu("test", a.mmlu_test)),
    ]:
        print(name)
        xtr, ytr, _, _ = encode(torso, tok, layers, tr(), want_lm=False)
        xte, yte, qte, zs = encode(torso, tok, layers, te(), want_lm=True)
        acc = {l: round(probe_mc_acc(xtr[l], ytr, xte[l], yte, qte), 4) for l in layers}
        report[name] = {"n_test_q": len(np.unique(qte)), "zero_shot_lm": round(zs, 4),
                        "probe_acc_by_layer": acc}
        print(json.dumps(report[name]))
    out = Path(f"runs/probe-mc-{a.model.split('/')[-1]}{a.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
