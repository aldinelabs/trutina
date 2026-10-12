import numpy as np

from trutina import wide


def test_groups_balanced_and_cover():
    for k in (27, 52, 53, 77, 151, 255):
        gs = wide.groups(k)
        assert sorted(i for g in gs for i in g) == list(range(k))
        assert max(map(len, gs)) <= wide.PASS
        assert max(map(len, gs)) - min(map(len, gs)) <= 1


def test_finalists_fit_one_pass():
    rng = np.random.default_rng(0)
    for k in (27, 77, 151, 255):
        parts = [(g, rng.dirichlet(np.ones(len(g)))) for g in wide.groups(k)]
        f = wide.finalists(parts)
        assert len(f) <= wide.PASS and f == sorted(f)
        for g, p in parts:  # each group's best option reaches the final
            assert g[int(np.argmax(p))] in f


def test_combine_is_a_distribution_led_by_the_final():
    rng = np.random.default_rng(1)
    k = 77
    parts = [(g, rng.dirichlet(np.ones(len(g)))) for g in wide.groups(k)]
    f = wide.finalists(parts)
    pf = rng.dirichlet(np.ones(len(f)))
    out = wide.combine(k, parts, f, pf)
    assert out.shape == (k,) and abs(out.sum() - 1) < 1e-9 and (out >= 0).all()
    assert int(np.argmax(out)) == f[int(np.argmax(pf))]
