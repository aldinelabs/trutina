"""Backbone probe (roadmap experiment 0) for one candidate torso on this machine.

  A. latency    full / half depth, several input lengths, bf16 vs 4-bit, per-layer overhead
  B. fork       prefix-fork encoding: numerical equivalence vs. sequential, and timing
  C. probe      BoolQ linear probe on every 4th layer's last-token hidden state

Usage:
  uv run python scripts/probe_backbone.py --model Qwen/Qwen3.5-4B-Base
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx import nn

from trutina.backbone import fork_cache, forward_hidden, load_backbone, make_cache
from trutina.latency import measure

STATE = (
    "Ticket #4821 from a enterprise customer: our nightly export job to S3 has failed three "
    "nights in a row with 'AccessDenied' after we rotated the IAM keys on Monday. The job "
    "worked for two years before that. We already re-entered the new access key in the "
    "dashboard and the connection test passes, but the scheduled run still fails. The "
    "customer is on the Business plan, has an SLA of 8 business hours, and mentions that "
    "the export feeds a compliance report due Friday. They ask whether they should create "
    "a brand new integration or wait for support to investigate."
)
QUESTION = "Which team should handle this ticket first?"
OPTIONS = [
    "Billing",
    "Integrations / data pipelines support",
    "Security & identity (IAM, credentials, access policies)",
    "Account management",
]


def ms(stats) -> dict:
    return {"p50_ms": round(stats.p50_ms, 2), "p95_ms": round(stats.p95_ms, 2)}


# ---------------------------------------------------------------- A. latency
def bench_latency(torso, n_layers: int, label: str) -> dict:
    res = {}
    rng = np.random.default_rng(0)
    for n_tok in (8, 200, 1000, 3400):
        toks = mx.array(rng.integers(1000, 100000, size=(1, n_tok)))
        reps = 30 if n_tok <= 200 else 8
        res[f"full_{n_tok}"] = ms(measure(lambda t=toks: forward_hidden(torso, t)[0], n=reps))
        if n_tok in (200, 3400):
            res[f"half_{n_tok}"] = ms(
                measure(lambda t=toks: forward_hidden(torso, t, upto=n_layers // 2)[0], n=reps)
            )
    # Per-layer fixed cost: tiny input, so time is dominated by dispatch + weight reads.
    res["per_layer_ms_at_8tok"] = round(res["full_8"]["p50_ms"] / n_layers, 3)
    toks = mx.array(rng.integers(1000, 100000, size=(1, 200)))
    try:
        fn = mx.compile(lambda t: forward_hidden(torso, t)[0])
        res["compiled_full_200"] = ms(measure(lambda: fn(toks), n=30))
    except (ValueError, RuntimeError) as e:  # compile support varies by layer; record it
        res["compiled_full_200"] = f"failed: {type(e).__name__}: {e}"[:200]
    tps = 3400 / (res["full_3400"]["p50_ms"] / 1000)
    res["prefill_tok_per_s_3400"] = round(tps)
    print(f"[{label}] " + json.dumps(res))
    return res


# ---------------------------------------------------------------- B. fork
def bench_fork(torso, tok, n_layers: int) -> dict:
    prefix = tok.encode(f"<state>\n{STATE}\n</state>\n<question>{QUESTION}</question>\n<option>")
    branches = [tok.encode(f" {o}</option>\n<answer>") for o in OPTIONS]
    mid = n_layers // 2

    def sequential_ref():
        finals, mids = [], []
        for b in branches:
            f, c = forward_hidden(torso, mx.array([prefix + b]), collect=[mid])
            finals.append(f[0, -1])
            mids.append(c[mid][0, -1])
        return mx.stack(finals), mx.stack(mids)

    def forked():
        cache = make_cache(torso)
        forward_hidden(torso, mx.array([prefix]), cache=cache)
        fc = fork_cache(cache, len(branches))
        L = max(len(b) for b in branches)
        padded = mx.array([b + [b[-1]] * (L - len(b)) for b in branches])  # right pad: causal-safe
        f, c = forward_hidden(torso, padded, cache=fc, collect=[mid])
        last = mx.array([len(b) - 1 for b in branches])
        take = lambda h: mx.take_along_axis(h, last[:, None, None], axis=1)[:, 0]
        return take(f), take(c[mid])

    rf, rm = sequential_ref()
    ff, fm = forked()
    mx.eval(rf, rm, ff, fm)

    def rel(a, b):
        a, b = a.astype(mx.float32), b.astype(mx.float32)
        return float(mx.abs(a - b).max() / mx.abs(a).max())

    def cos(a, b):
        a, b = a.astype(mx.float32), b.astype(mx.float32)
        return float(((a * b).sum(-1) / (mx.linalg.norm(a, axis=-1) * mx.linalg.norm(b, axis=-1))).min())

    concat = mx.array([prefix + [t for b in branches for t in b]])
    res = {
        "prefix_tokens": len(prefix),
        "branch_tokens": [len(b) for b in branches],
        "final_max_rel_diff": round(rel(rf, ff), 5),
        "final_min_cosine": round(cos(rf, ff), 6),
        "mid_max_rel_diff": round(rel(rm, fm), 5),
        "mid_min_cosine": round(cos(rm, fm), 6),
        "time_forked": ms(measure(lambda: forked()[0], n=20)),
        "time_concat_one_seq": ms(measure(lambda: forward_hidden(torso, concat)[0], n=20)),
        "time_separate_passes": ms(measure(lambda: sequential_ref()[0], n=10)),
    }
    print("[fork] " + json.dumps(res))
    return res


# ---------------------------------------------------------------- C. probe
def logreg_acc(xtr, ytr, xte, yte, steps: int = 400) -> float:
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-5
    xtr, xte = mx.array((xtr - mu) / sd), mx.array((xte - mu) / sd)
    ytr = mx.array(ytr)
    lin = nn.Linear(xtr.shape[1], 1)
    opt = optim.AdamW(1e-3, weight_decay=0.05)

    def loss(m):
        return nn.losses.binary_cross_entropy(m(xtr)[:, 0], ytr, with_logits=True)

    step = nn.value_and_grad(lin, loss)
    for _ in range(steps):
        _, g = step(lin)
        opt.update(lin, g)
        mx.eval(lin.parameters(), opt.state)
    pred = np.array(lin(xte)[:, 0] > 0)
    return float((pred == yte).mean())


def probe_boolq(torso, tok, n_layers: int, n_train: int, n_test: int, max_tok: int) -> dict:
    from datasets import load_dataset

    ds = load_dataset("google/boolq")
    layers = sorted(set(range(4, n_layers + 1, 4)) | {n_layers // 2})

    def feats(split, n):
        xs = {l: [] for l in layers}
        ys = []
        t0 = time.perf_counter()
        for ex in ds[split].select(range(n)):
            ids = tok.encode(f"{ex['passage']}\nQuestion: {ex['question']}?\nAnswer:")[-max_tok:]
            _, c = forward_hidden(torso, mx.array([ids]), collect=layers)
            for l in layers:
                xs[l].append(c[l][0, -1].astype(mx.float32))
            ys.append(int(ex["answer"]))
        out = {l: np.array(mx.stack(v)) for l, v in xs.items()}
        print(f"  {split}: {n} examples in {time.perf_counter() - t0:.0f}s")
        return out, np.array(ys)

    xtr, ytr = feats("train", n_train)
    xte, yte = feats("validation", n_test)
    acc = {l: round(logreg_acc(xtr[l], ytr, xte[l], yte), 4) for l in layers}
    res = {"majority": round(max(yte.mean(), 1 - yte.mean()), 4), "acc_by_layer": acc}
    print("[probe] " + json.dumps(res))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip", nargs="*", default=[], choices=["latency", "quant", "fork", "probe"])
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--max-tok", type=int, default=512)
    a = ap.parse_args()

    torso, tok = load_backbone(a.model)
    n_layers = len(torso.layers)
    report = {"model": a.model, "n_layers": n_layers, "device": str(mx.default_device())}
    if "latency" not in a.skip:
        report["latency_bf16"] = bench_latency(torso, n_layers, "bf16")
    if "fork" not in a.skip:
        report["fork"] = bench_fork(torso, tok, n_layers)
    if "probe" not in a.skip:
        report["probe_boolq"] = probe_boolq(torso, tok, n_layers, a.n_train, a.n_test, a.max_tok)
    if "quant" not in a.skip:
        for bits in (8, 4):
            torso, _ = load_backbone(a.model)
            nn.quantize(torso, group_size=64, bits=bits)
            report[f"latency_q{bits}"] = bench_latency(torso, n_layers, f"q{bits}")

    out = Path(a.out or f"runs/probe-{a.model.split('/')[-1]}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
