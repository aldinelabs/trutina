import mlx.core as mx
import numpy as np
from test_api import REQ
from test_backbone import tiny_qwen35
from test_engines import ByteTok

from trutina.api import handle
from trutina.decider import DeciderEngine, DecisionBatch, MultiExitDecider
from trutina.prompt import encode_question
from trutina.schema import Question


def tiny_decider():
    mx.random.seed(1)
    model = MultiExitDecider(tiny_qwen35(), hidden=64, exits=(4, 6, 8), mix_every=2,
                             head_dim=32, head_layers=1, head_heads=4)
    mx.eval(model.parameters())
    return model


def batch():
    tok = ByteTok()
    qs = [Question("a", "pick", "choice", ("x", "y", "z")), Question("b", "ok?", "noul")]
    return DecisionBatch.build([encode_question(tok, "state", q) for q in qs], [q.kind for q in qs])


def test_every_exit_returns_masked_logits():
    out = tiny_decider()(batch())
    assert len(out) == 3
    for logits in out:
        assert logits.shape == (2, 3)
        assert float(logits[1, 2]) < -1e8  # noul has 2 options; third slot is padding


def test_adaptive_matches_full_pass_and_respects_thresholds():
    model, b = tiny_decider(), batch()
    full = model(b)
    first, used = model.adaptive(b, thresholds=(0.0, 0.0))  # everything is "confident"
    assert used == 0
    np.testing.assert_allclose(np.array(first), np.array(full[0]), atol=1e-4)
    last, used = model.adaptive(b, thresholds=(1.1, 1.1))  # nothing is confident
    assert used == 2
    np.testing.assert_allclose(np.array(last), np.array(full[2]), atol=1e-4)


def test_engine_depth_modes():
    engine = DeciderEngine(tiny_decider(), ByteTok(), thresholds=(1.1, 1.1))
    for depth, exit_used in [("shallow", 0), ("adaptive", 2), ("deep", 2)]:
        out = handle({**REQ, "depth": depth}, engine)["answers"]
        assert engine.last_exit == exit_used
        assert abs(sum(out["route"]["probabilities"].values()) - 1) < 1e-5
        assert set(out["severity"]["legend"]) == {"0", "1", "2"}


def test_default_adaptive_never_goes_past_the_second_exit():
    engine = DeciderEngine(tiny_decider(), ByteTok())
    handle({**REQ, "depth": "adaptive"}, engine)
    assert engine.last_exit in (0, 1)
    handle({**REQ, "depth": "deep"}, engine)
    assert engine.last_exit == 2
