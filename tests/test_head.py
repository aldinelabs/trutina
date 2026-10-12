import mlx.core as mx
import numpy as np

from trutina.head import JointHead, ScalarMix


def test_joint_head_is_permutation_equivariant():
    mx.random.seed(0)
    head = JointHead(hidden=32, dim=16, layers=2, heads=4)
    a, o = mx.random.normal((2, 32)), mx.random.normal((2, 4, 32))
    mask = mx.array([[True] * 4, [True, True, True, False]])
    perm = [2, 0, 1, 3]  # keep the padded slot of row 1 last
    base = np.array(head(a, o, mask, 1))
    permuted = np.array(head(a, o[:, perm], mask, 1))
    np.testing.assert_allclose(permuted[:, :3], base[:, perm][:, :3], atol=1e-5)
    assert permuted[1, 3] < -1e8  # padding stays masked


def test_scalar_mix_starts_as_mean_of_normalised_layers():
    mix = ScalarMix(2)
    h1, h2 = mx.ones((1, 4)) * 3.0, mx.ones((1, 4)) * -5.0
    np.testing.assert_allclose(np.array(mix([h1, h2])), 0.0, atol=1e-5)  # rms(h1)=1, rms(h2)=-1


def test_ordinal_levels_form_a_distribution():
    from trutina.head import ordinal_log_probs

    mask = mx.array([[True] * 3 + [False] * 2, [True] * 5])
    p = mx.exp(ordinal_log_probs(mx.array([0.3, 3.6]), mx.array([0.7, 0.4]), mask))
    np.testing.assert_allclose(np.array((p * mask).sum(-1)), [1.0, 1.0], atol=1e-5)
    assert np.argmax(np.array(p[1])) == 4 and np.array(p[0, 3:]).max() < 1e-8


def test_v1_head_options_keep_choice_equivariance():
    head = JointHead(hidden=32, dim=16, layers=1, heads=4, read_state=True, n_sources=3, ordinal_score=True)
    a, o = mx.random.normal((1, 32)), mx.random.normal((1, 4, 32))
    mask = mx.ones((1, 4), mx.bool_)
    mem, mm = mx.random.normal((1, 9, 32)), mx.ones((1, 9), mx.bool_)
    perm = [2, 0, 3, 1]
    kw = {"mem": mem, "mem_mask": mm, "source": mx.array([1]), "is_score": mx.array([False])}
    base = np.array(head(a, o, mask, 0, **kw))
    np.testing.assert_allclose(np.array(head(a, o[:, perm], mask, 0, **kw)), base[:, perm], atol=1e-5)
