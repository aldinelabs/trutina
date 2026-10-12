"""Accuracy and calibration metrics, plus temperature scaling."""

from __future__ import annotations

import numpy as np


def accuracy(probs: list[np.ndarray], labels: list[int]) -> float:
    return float(np.mean([int(np.argmax(p) == y) for p, y in zip(probs, labels)]))


def brier(probs: list[np.ndarray], labels: list[int]) -> float:
    return float(np.mean([np.sum((p - np.eye(len(p))[y]) ** 2) for p, y in zip(probs, labels)]))


def ece(probs: list[np.ndarray], labels: list[int], n_bins: int = 15) -> float:
    conf = np.array([p.max() for p in probs])
    correct = np.array([int(np.argmax(p) == y) for p, y in zip(probs, labels)])
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def fit_temperature(logits: list[np.ndarray], labels: list[int]) -> float:
    """Single scalar temperature minimising held-out NLL (grid search; robust and cheap)."""
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 400))

    def nll(t: float) -> float:
        return -float(np.mean([np.log(softmax(l, t)[y] + 1e-12) for l, y in zip(logits, labels)]))

    return float(min(grid, key=nll))
