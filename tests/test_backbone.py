import mlx.core as mx
import numpy as np
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

from trutina.backbone import fork_cache, forward_hidden, make_cache


def tiny_qwen35():
    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=64, intermediate_size=128, num_hidden_layers=8,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=500,
        linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16,
        linear_value_head_dim=16, tie_word_embeddings=True,
    )
    mx.random.seed(0)
    torso = TextModel(args).model
    mx.eval(torso.parameters())
    return torso


def test_forked_branches_match_sequential():
    torso = tiny_qwen35()
    prefix = list(range(10, 40))
    branches = [[50, 51, 52], [60, 61], [70, 71, 72, 73]]

    cache = make_cache(torso)
    forward_hidden(torso, mx.array([prefix]), cache=cache)
    L = max(map(len, branches))
    padded = mx.array([b + [b[-1]] * (L - len(b)) for b in branches])
    forked, mids = forward_hidden(torso, padded, cache=fork_cache(cache, 3), collect=[4])

    for i, b in enumerate(branches):
        ref, ref_mid = forward_hidden(torso, mx.array([prefix + b]), collect=[4])
        np.testing.assert_allclose(np.array(forked[i, len(b) - 1]), np.array(ref[0, -1]), atol=1e-3)
        np.testing.assert_allclose(np.array(mids[4][i, len(b) - 1]), np.array(ref_mid[4][0, -1]), atol=1e-3)


def test_upto_matches_collected_layer():
    torso = tiny_qwen35()
    toks = mx.array([list(range(1, 20))])
    _, c = forward_hidden(torso, toks, collect=[4])
    half, _ = forward_hidden(torso, toks, upto=4)
    np.testing.assert_allclose(np.array(half), np.array(torso.norm(c[4])), atol=1e-5)


def test_continuing_from_exit_matches_full_pass():
    torso = tiny_qwen35()
    toks = mx.array([list(range(1, 20))])
    full, _ = forward_hidden(torso, toks)
    _, c = forward_hidden(torso, toks, upto=4, collect=[4])
    rest, _ = forward_hidden(torso, toks, start=4, h0=c[4])
    np.testing.assert_allclose(np.array(rest), np.array(full), atol=1e-5)
