"""Framework-free evaluation metrics for decision outputs (shared by the MLX and torch CLIs).

`summarize`: per-question-type accuracy, Brier, ECE, NLL, score MAE and, for soft targets,
KL(target || p), mean target confidence and accuracy on clear-cut questions.
`adaptive_sweep`: accuracy vs mean exit depth when stopping at the first confident exit.
"""

from __future__ import annotations

import numpy as np

from trutina.metrics import ece


def summarize(probs: list[np.ndarray], targets: list[np.ndarray], kinds: list[str],
              clear: float = 0.7) -> dict:
    """Hard metrics against argmax(target); with soft targets (annotator / teacher
    distributions) also KL(target || p), the targets' own ambiguity, and accuracy on the
    questions whose target puts at least `clear` on one option."""
    gold = [int(np.argmax(t)) for t in targets]

    def block(idx):
        p, g = [probs[i] for i in idx], [gold[i] for i in idx]
        t = [targets[i] for i in idx]
        out = {"n": len(idx),
               "acc": float(np.mean([np.argmax(x) == y for x, y in zip(p, g)])),
               "brier": float(np.mean([np.sum((x - np.eye(len(x))[y]) ** 2) for x, y in zip(p, g)])),
               "ece": ece(p, g),
               "nll": float(np.mean([-np.log(max(x[y], 1e-12)) for x, y in zip(p, g)]))}
        if any(tt.max() < 1 for tt in t):
            out["kl_soft"] = float(np.mean([np.sum(tt * (np.log(np.maximum(tt, 1e-12))
                                                         - np.log(np.maximum(x, 1e-12)))) for x, tt in zip(p, t)]))
            out["target_max"] = float(np.mean([tt.max() for tt in t]))
            sure = [j for j, tt in enumerate(t) if tt.max() >= clear]
            out["clear_n"] = len(sure)
            out["clear_acc"] = float(np.mean([np.argmax(p[j]) == g[j] for j in sure])) if sure else None
        sc = [i for i in idx if kinds[i] == "score"]
        if sc:  # expected level vs gold level
            out["score_mae"] = float(np.mean([abs(probs[i] @ np.arange(len(probs[i])) - gold[i]) for i in sc]))
        return out

    res = {"all": block(range(len(gold)))}
    for k in ("choice", "score", "noul"):
        idx = [i for i, kind in enumerate(kinds) if kind == k]
        if idx:
            res[k] = block(idx)
    return res


def adaptive_sweep(per_exit: list[list[np.ndarray]], gold: list[int], exits: list[int]) -> list[dict]:
    """Stop at the first exit whose max probability >= t (last exit always answers)."""
    rows = []
    for t in (0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.01):
        chosen, layers = [], []
        for i in range(len(gold)):
            for k, e in enumerate(exits):
                if per_exit[k][i].max() >= t or k == len(exits) - 1:
                    chosen.append(per_exit[k][i])
                    layers.append(e)
                    break
        rows.append({"threshold": t, "acc": float(np.mean([np.argmax(p) == g for p, g in zip(chosen, gold)])),
                     "mean_layers": float(np.mean(layers)),
                     "share_per_exit": {str(e): float(np.mean([x == e for x in layers])) for e in exits}})
    return rows
