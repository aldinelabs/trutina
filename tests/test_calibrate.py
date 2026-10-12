import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from calibrate import apply, fit, scale


def test_fit_recovers_a_known_temperature():
    rng = np.random.default_rng(0)
    n, k = 4000, 4
    valid = np.ones((n, k), dtype=bool)
    valid[: n // 2, 3] = False  # half the questions have 3 options
    logits = np.where(valid, rng.normal(0, 2, (n, k)), -np.inf)
    truth = np.exp(logits - logits.max(1, keepdims=True))
    truth /= truth.sum(1, keepdims=True)
    sharp = scale(truth, valid, 0.5)  # an overconfident model: true distribution at T = 0.5
    t = fit(sharp, truth, valid)
    assert abs(t - 2.0) < 0.05
    np.testing.assert_allclose(scale(sharp, valid, t), truth, atol=0.02)
    assert (scale(sharp, valid, t)[~valid] == 0).all()


def test_apply_per_kind_only_touches_that_kind():
    p = np.array([[0.7, 0.3], [0.7, 0.3]])
    d = {"exits": [12], "kinds": ["choice", "noul"], "valid": np.ones((2, 2), bool), "probs": [p]}
    out = apply(d, {"12": {"choice": 2.0}})[0]
    assert out[0, 0] < 0.7 and np.allclose(out[1], p[1])
