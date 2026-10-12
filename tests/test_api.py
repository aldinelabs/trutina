import numpy as np
import pytest

from trutina.api import RequestError, confidence, handle, parse_request

REQ = {
    "state": "The invoice total is $1,240 but the PO says $1,200.",
    "questions": {
        "route": {"type": "choice", "instructions": "Who handles this?",
                  "criteria": {"billing": "money issues", "support": "", "sales": "deals"}},
        "severity": {"type": "score", "instructions": "How severe?",
                     "criteria": ["none", {"label": "minor"}, {"label": "major", "description": "blocks payment"}]},
        "mismatch": {"type": "noul", "instructions": "Do the amounts differ?"},
    },
}


def fixed_engine(req):
    table = {"choice": [0.7, 0.2, 0.1], "score": [0.1, 0.6, 0.3], "noul": [0.25, 0.75]}
    return {q.id: table[q.kind] for q in req.questions}


def test_parse_maps_all_three_types():
    req = parse_request(REQ)
    route, severity, mismatch = req.questions
    assert req.depth == "adaptive"
    assert route.resolved_options() == ("billing", "support", "sales")
    assert route.resolved_descriptions() == ("money issues", "", "deals")
    assert severity.resolved_options() == ("none", "minor", "major")
    assert severity.resolved_descriptions() == ("", "", "blocks payment")
    assert mismatch.resolved_options() == ("no", "yes")


def test_response_shapes():
    ans = handle(REQ, fixed_engine)["answers"]
    assert ans["route"]["choice"] == "billing"
    assert set(ans["route"]["probabilities"]) == {"billing", "support", "sales"}
    assert ans["severity"]["score"] == pytest.approx(0 * 0.1 + 1 * 0.6 + 2 * 0.3)
    assert ans["severity"]["legend"] == {"0": "none", "1": "minor", "2": "major"}
    assert ans["mismatch"] == {"type": "noul", "noul": 0.75}  # no confidence for noul


@pytest.mark.parametrize("patch, msg", [
    ({"state": None}, "state"),
    ({"questions": {}}, "non-empty"),
    ({"depth": "think"}, "depth"),
    ({"questions": {"x": {"type": "rank", "instructions": "?"}}}, "type"),
    ({"questions": {"x": {"type": "choice", "instructions": "?", "criteria": {"a": ""}}}}, "2..255"),
    ({"questions": {"x": {"type": "choice", "instructions": "?",
                          "criteria": {str(i): "" for i in range(256)}}}}, "2..255"),
    ({"questions": {"x": {"type": "score", "instructions": "?", "criteria": list("abcdefghijk")}}}, "2..10"),
    ({"questions": {"x": {"type": "noul", "instructions": "?", "criteria": {"maybe": "?"}}}}, "true"),
    ({"questions": {"x": {"type": "noul"}}}, "instructions"),
])
def test_invalid_requests_are_400(patch, msg):
    with pytest.raises(RequestError, match=msg) as e:
        parse_request({**REQ, **patch})
    assert e.value.status == 400


def test_confidence_bounds():
    assert confidence(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert confidence(np.array([0.25] * 4)) == pytest.approx(0.0)
    assert 0 < confidence(np.array([0.6, 0.4])) < 1


def test_engine_must_cover_every_option():
    with pytest.raises(ValueError, match="invalid probabilities"):
        handle(REQ, lambda req: {q.id: [1.0] for q in req.questions})
