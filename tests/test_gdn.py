import mlx.core as mx
import numpy as np
from mlx import nn
from mlx_lm.models.gated_delta import gated_delta_ops, normalize_qk

from trutina.gdn import chunked_gated_delta, unit_lower_inverse


def test_unit_lower_inverse():
    with mx.stream(mx.cpu):
        a = mx.tril(mx.random.normal((3, 64, 64), key=mx.random.key(0)) * 0.2, -1)
        np.testing.assert_allclose(np.array((mx.eye(64) + a) @ unit_lower_inverse(a)),
                                   np.broadcast_to(np.eye(64), (3, 64, 64)), atol=1e-4)


def test_inverse_is_stable_for_correlated_keys():
    # identical unit keys, beta=1, no decay: every strictly-lower entry is 1. A Neumann
    # series overflows here; the true inverse is bidiagonal (1 on the diagonal, -1 below).
    with mx.stream(mx.cpu):
        a = mx.tril(mx.ones((64, 64)), -1)
        inv = np.array(unit_lower_inverse(a))
        assert np.isfinite(inv).all()
        np.testing.assert_allclose(inv, np.eye(64) - np.eye(64, k=-1), atol=1e-5)


def inputs(T=150, B=2, Hk=2, Hv=4, D=32):
    ks = mx.random.split(mx.random.key(1), 6)
    q, k = normalize_qk(mx.random.normal((B, T, Hk, D), key=ks[0]),
                        mx.random.normal((B, T, Hk, D), key=ks[1]), inv_scale=D**-0.5, eps=1e-6)
    v = mx.random.normal((B, T, Hv, D), key=ks[2])
    log_g = -mx.exp(mx.random.normal((Hv,), key=ks[3]) - 1) * nn.softplus(
        mx.random.normal((B, T, Hv), key=ks[4]))
    beta = mx.sigmoid(mx.random.normal((B, T, Hv), key=ks[5]))
    state = mx.random.normal((B, Hv, D, D), key=ks[0]) * 0.1
    return q, k, v, log_g, beta, state


def test_chunked_matches_recurrent_forward_and_grads():
    with mx.stream(mx.cpu):  # exact fp32 reference
        q, k, v, log_g, beta, state = inputs()
        y_ref, s_ref = gated_delta_ops(q, k, v, mx.exp(log_g), beta, state)
        y, s = chunked_gated_delta(q, k, v, log_g, beta, state)
        np.testing.assert_allclose(np.array(y), np.array(y_ref), atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(np.array(s), np.array(s_ref), atol=1e-4, rtol=1e-4)

        w = mx.random.normal(y.shape, key=mx.random.key(9))

        def ref_loss(q, k, v, lg, beta):
            return (gated_delta_ops(q, k, v, mx.exp(lg), beta, state)[0] * w).sum()

        def loss(q, k, v, lg, beta):
            return (chunked_gated_delta(q, k, v, lg, beta, state)[0] * w).sum()

        args = (q, k, v, log_g, beta)
        g_ref = mx.grad(ref_loss, argnums=(0, 1, 2, 3, 4))(*args)
        g = mx.grad(loss, argnums=(0, 1, 2, 3, 4))(*args)
        for a, b in zip(g, g_ref):
            scale = float(mx.abs(b).max())
            np.testing.assert_allclose(np.array(a) / scale, np.array(b) / scale, atol=1e-4)
