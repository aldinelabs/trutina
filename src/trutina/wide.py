"""Choice questions with more than 26 options: a two-round tournament that only uses the letters A..Z.

The model reads one sequence per group of at most 26 options (balanced, in the given order); the top
26 // groups of every group meet in a final sequence. Every option gets a probability: a finalist
M * p_final, the rest (1 - M) in proportion to their group probability, where M is the mean group
mass of the finalists. Engine-agnostic: the engines call `groups`, then `finalists`, then `combine`.

On MASSIVE intent classification (59 intents, not used in training) this reads 79.3% top-1, against
59.4% for single-token two-letter labels (AA, AB, ...) that the model never saw in training.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

PASS = 26  # options per sequence: the letters the model was trained on


def groups(k: int) -> list[list[int]]:
    n = -(-k // PASS)
    cuts = [round(i * k / n) for i in range(n + 1)]
    return [list(range(cuts[i], cuts[i + 1])) for i in range(n)]


def finalists(parts: Sequence[tuple[list[int], np.ndarray]]) -> list[int]:
    """Original option indices that reach the final, in their original order."""
    per = PASS // len(parts)
    return sorted(i for g, p in parts for i in np.asarray(g)[np.argsort(-np.asarray(p))[:per]].tolist())


def combine(k: int, parts: Sequence[tuple[list[int], np.ndarray]], fin: Sequence[int], p_final: np.ndarray) -> np.ndarray:
    group_p = np.zeros(k)
    for g, p in parts:
        group_p[g] = np.asarray(p) / len(parts)
    is_fin = np.zeros(k, dtype=bool)
    is_fin[list(fin)] = True
    m = float(group_p[is_fin].sum())
    out = np.zeros(k)
    out[list(fin)] = m * np.asarray(p_final)
    rest = group_p[~is_fin]
    out[~is_fin] = (1 - m) * rest / rest.sum() if rest.sum() > 0 else 0.0
    return out / out.sum()
