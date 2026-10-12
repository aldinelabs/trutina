import mlx.core as mx
import pytest

from trutina.losses import brier, decision_loss, rps, soft_ce


def test_losses_prefer_correct_and_near_misses():
    mask = mx.array([[True] * 4])
    target = mx.array([[0.0, 0.0, 1.0, 0.0]])
    good, near, far = (mx.array([[0.0, 0.0, 5.0, 0.0]]), mx.array([[0.0, 5.0, 0.0, 0.0]]),
                       mx.array([[5.0, 0.0, 0.0, 0.0]]))
    assert soft_ce(good, target, mask).item() < soft_ce(near, target, mask).item()
    assert brier(good, target, mask).item() < brier(near, target, mask).item()
    # RPS is ordinal: predicting level 1 for true level 2 beats predicting level 0.
    assert rps(near, target, mask).item() < rps(far, target, mask).item()
    assert soft_ce(near, target, mask).item() == pytest.approx(soft_ce(far, target, mask).item())


def test_padding_does_not_change_loss():
    t3 = mx.array([[0.0, 1.0, 0.0]])
    l3 = mx.array([[0.3, 1.2, -0.4]])
    m3 = mx.array([[True] * 3])
    t4 = mx.array([[0.0, 1.0, 0.0, 0.0]])
    l4 = mx.array([[0.3, 1.2, -0.4, 9.0]])
    m4 = mx.array([[True, True, True, False]])
    for score in (False, True):
        a = decision_loss(l3, t3, m3, mx.array([score])).item()
        b = decision_loss(l4, t4, m4, mx.array([score])).item()
        assert a == pytest.approx(b, rel=1e-5)


def test_smoothing_leaves_teacher_distributions_alone():
    from trutina.losses import smooth

    mask = mx.array([[True, True, True], [True, True, True]])
    t = mx.array([[0.0, 1.0, 0.0], [0.2, 0.5, 0.3]])
    out = smooth(t, mask, 0.3)
    assert abs(out[0, 1].item() - (0.7 + 0.1)) < 1e-6        # one-hot: smoothed
    assert mx.allclose(out[1], t[1]).item()                     # soft: untouched
    assert abs(smooth(t, mask, 0.3, soft_too=True)[1, 0].item() - (0.14 + 0.1)) < 1e-6
