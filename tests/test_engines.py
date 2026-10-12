import mlx.core as mx
import numpy as np
from mlx import nn
from test_api import REQ
from test_backbone import tiny_qwen35

from trutina.api import handle
from trutina.backbone import forward_hidden
from trutina.engines import LMScoringEngine, render_text
from trutina.schema import Question


class ByteTok:
    def encode(self, text: str) -> list[int]:
        return list(text.encode())


def test_end_to_end_response_is_well_formed():
    out = handle(REQ, LMScoringEngine(tiny_qwen35(), ByteTok()))["answers"]
    assert set(out) == {"route", "severity", "mismatch"}
    assert abs(sum(out["route"]["probabilities"].values()) - 1) < 1e-6
    assert 0 <= out["severity"]["score"] <= 2
    assert 0 <= out["mismatch"]["noul"] <= 1


def test_forked_scoring_matches_full_sequence_scoring():
    torso, tok = tiny_qwen35(), ByteTok()
    q = Question("q", "Which?", "choice", ("alpha", "be"), ("", ""))
    got = LMScoringEngine(torso, tok).score_question("some state", q)

    prefix = tok.encode(render_text("some state", q))
    ref = []
    for lbl in q.resolved_options():
        b = tok.encode(" " + lbl)
        h, _ = forward_hidden(torso, mx.array([prefix + b]))
        logp = nn.log_softmax(torso.embed_tokens.as_linear(h[0]).astype(mx.float32), axis=-1)
        pos = mx.arange(len(prefix) - 1, len(prefix) - 1 + len(b))
        ref.append(float(logp[pos, mx.array(b)].mean()))
    ref = np.exp(np.array(ref) - max(ref))
    np.testing.assert_allclose(got, ref / ref.sum(), atol=1e-3)
