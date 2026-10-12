"""Per-exit temperature scaling, fitted on one probability dump and checked on others.

  uv run python scripts/calibrate.py --run runs/v1 --fit probs-val-best.npz \\
      --check probs-eval_localllama_test-best.npz

Dumps come from `scripts/evaluate.py --dump-probs`. For each exit, the temperature T
minimises the cross-entropy against the (soft) targets on the fit set; probabilities are
then softmax(log p / T). Also fits one temperature per exit and question type. Reports
accuracy, ECE, KL and mean confidence before / after, plus the adaptive-depth sweep on the
calibrated probabilities, and writes <run>/calibration.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trutina.evaluation import adaptive_sweep, summarize

KINDS = ("choice", "score", "noul")


def load(path: Path) -> dict:
    z = np.load(path)
    exits = [int(e) for e in z["exits"]]
    if "n_options" in z:
        n_opt = z["n_options"]
    else:  # older dumps: padded options have exactly zero probability
        n_opt = np.array([int((row > 0).sum()) for row in z[f"exit{exits[0]}"]])
    valid = np.arange(z["targets"].shape[1])[None] < n_opt[:, None]
    return {"exits": exits, "kinds": [str(k) for k in z["kinds"]], "targets": z["targets"], "valid": valid,
            "probs": [z[f"exit{e}"] for e in exits]}


def scale(p: np.ndarray, valid: np.ndarray, t: float) -> np.ndarray:
    z = np.where(valid, np.log(np.maximum(p, 1e-12)) / t, -np.inf)
    z = z - z.max(1, keepdims=True)
    e = np.where(valid, np.exp(z), 0.0)
    return e / e.sum(1, keepdims=True)


def soft_ce(p: np.ndarray, targets: np.ndarray, valid: np.ndarray) -> float:
    return float(-(targets * np.where(valid, np.log(np.maximum(p, 1e-12)), 0.0)).sum(1).mean())


def fit(p: np.ndarray, targets: np.ndarray, valid: np.ndarray) -> float:
    """Temperature minimising soft cross-entropy: coarse log grid, then a finer one."""
    grid = np.exp(np.linspace(np.log(0.25), np.log(4.0), 81))
    best = min(grid, key=lambda t: soft_ce(scale(p, valid, t), targets, valid))
    fine = best * np.exp(np.linspace(-0.05, 0.05, 41))
    return float(min(fine, key=lambda t: soft_ce(scale(p, valid, t), targets, valid)))


def rows(p: np.ndarray, valid: np.ndarray) -> list[np.ndarray]:
    return [r[v] for r, v in zip(p, valid)]


def apply(d: dict, temps: dict) -> list[np.ndarray]:
    """`temps[exit]` is a float or {kind: float}."""
    out = []
    for k, e in enumerate(d["exits"]):
        t = temps[str(e)]
        p = d["probs"][k].copy()
        if isinstance(t, dict):
            for kind, tk in t.items():
                m = np.array([x == kind for x in d["kinds"]])
                p[m] = scale(d["probs"][k][m], d["valid"][m], tk)
        else:
            p = scale(p, d["valid"], t)
        out.append(p)
    return out


def report(d: dict, probs: list[np.ndarray]) -> dict:
    targets = rows(d["targets"], d["valid"])
    res = {}
    for k, e in enumerate(d["exits"]):
        ps = rows(probs[k], d["valid"])
        s = summarize(ps, targets, d["kinds"])["all"]
        res[f"exit{e}"] = {"acc": s["acc"], "ece": s["ece"], "kl": s.get("kl_soft"), "nll": s["nll"],
                           "mean_conf": float(np.mean([x.max() for x in ps]))}
    gold = [int(np.argmax(t)) for t in targets]
    res["adaptive"] = adaptive_sweep([rows(p, d["valid"]) for p in probs], gold, d["exits"])
    return res


def show(name: str, before: dict, after: dict, exits: list[int]) -> None:
    print(f"\n{name}")
    for e in exits:
        b, a = before[f"exit{e}"], after[f"exit{e}"]
        print(f"  exit{e}: acc {b['acc']:.3f} -> {a['acc']:.3f}  ece {b['ece']:.3f} -> {a['ece']:.3f}  "
              f"kl {b['kl']:.3f} -> {a['kl']:.3f}  conf {b['mean_conf']:.3f} -> {a['mean_conf']:.3f}")
    print("  adaptive " + " ".join(f"t={r['threshold']}: {r['acc']:.3f}@{r['mean_layers']:.1f}L"
                                   for r in after["adaptive"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--fit", required=True, help="dump to fit on (file name inside the run, or a path)")
    ap.add_argument("--check", nargs="*", default=[], help="held-out dumps to report on")
    a = ap.parse_args()
    run = Path(a.run)
    path = lambda x: run / x if (run / x).exists() else Path(x)

    d = load(path(a.fit))
    per_exit = {str(e): fit(d["probs"][k], d["targets"], d["valid"]) for k, e in enumerate(d["exits"])}
    per_kind = {}
    for k, e in enumerate(d["exits"]):
        per_kind[str(e)] = {}
        for kind in KINDS:
            m = np.array([x == kind for x in d["kinds"]])
            if m.sum() >= 50:
                per_kind[str(e)][kind] = fit(d["probs"][k][m], d["targets"][m], d["valid"][m])
    print("temperature per exit:", {e: round(t, 3) for e, t in per_exit.items()})
    print("temperature per exit and type:", {e: {k: round(t, 3) for k, t in v.items()} for e, v in per_kind.items()})

    out = {"fit_on": str(path(a.fit)), "per_exit": per_exit, "per_exit_kind": per_kind, "reports": {}}
    for name, dump in [(a.fit, d)] + [(c, load(path(c))) for c in a.check]:
        before = report(dump, dump["probs"])
        after = report(dump, apply(dump, per_exit))
        after_kind = report(dump, apply(dump, per_kind))
        show(f"{name} (per-exit T)", before, after, dump["exits"])
        show(f"{name} (per-exit-and-type T)", before, after_kind, dump["exits"])
        out["reports"][name] = {"before": before, "per_exit": after, "per_exit_kind": after_kind}
    (run / "calibration.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> {run / 'calibration.json'}")


if __name__ == "__main__":
    main()
