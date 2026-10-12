import base64

import pytest

from trutina.api import RequestError
from trutina.api import handle as handle_systemone
from trutina.decisions_api import handle_decisions, parse_decisions_request
from trutina.vision import ImageRef

PNG = ("data:image/png;base64,"
       + base64.b64encode(bytes.fromhex(
           "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
           "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082")).decode())

REQ = {
    "model": "trutina-4b",
    "input": "The invoice total is $1,240 but the PO says $1,200.",
    "questions": [
        {"type": "choice", "name": "route", "instructions": "Who handles this?",
         "choices": [{"value": "billing", "description": "money issues"}, {"value": "support"},
                     {"value": "sales", "description": "deals"}]},
        {"type": "score", "name": "severity", "instructions": "How severe?",
         "levels": [{"label": "none"}, {"label": "minor"}, {"label": "major", "description": "blocks payment"}]},
        {"type": "predicate", "name": "mismatch", "instructions": "Do the amounts differ?"},
    ],
}

SYSTEMONE = {
    "state": REQ["input"],
    "questions": {
        "route": {"type": "choice", "instructions": "Who handles this?",
                  "criteria": {"billing": "money issues", "support": "", "sales": "deals"}},
        "severity": {"type": "score", "instructions": "How severe?",
                     "criteria": ["none", "minor", {"label": "major", "description": "blocks payment"}]},
        "mismatch": {"type": "noul", "instructions": "Do the amounts differ?"},
    },
}


def fixed_engine(req):
    table = {"choice": [0.7, 0.2, 0.1], "score": [0.1, 0.6, 0.3], "noul": [0.25, 0.75]}
    return {q.id: table[q.kind] for q in req.questions}


def test_same_internal_request_as_systemone():
    from trutina.api import parse_request

    req, model = parse_decisions_request(REQ)
    assert model == "trutina-4b"
    assert req == parse_request(SYSTEMONE)


def test_response_shapes():
    out = handle_decisions(REQ, fixed_engine)
    assert out["model"] == "trutina-4b"
    route, severity, mismatch = out["answers"]
    assert route["name"] == "route" and route["choice"] == "billing"
    assert [p["value"] for p in route["probabilities"]] == ["billing", "support", "sales"]
    assert severity["score"] == pytest.approx(1.2)
    assert severity["probabilities"][2] == {"value": 2, "label": "major", "probability": pytest.approx(0.3)}
    assert mismatch == {"type": "predicate", "name": "mismatch", "probability": 0.75}
    # same numbers as the systemone rendering
    s1 = handle_systemone(SYSTEMONE, fixed_engine)["answers"]
    assert s1["route"]["confidence"] == route["confidence"]


def test_messages_with_images():
    body = {**REQ, "input": [
        {"role": "developer", "content": "Billing desk."},
        {"role": "user", "content": [{"type": "input_text", "text": "Invoice:"},
                                     {"type": "input_image", "image_url": PNG}]}]}
    req, _ = parse_decisions_request(body)
    assert req.state[:3] == ("Billing desk.", "\n\n", "Invoice:")
    assert isinstance(req.state[3], ImageRef)
    text_only, _ = parse_decisions_request({**REQ, "input": [{"role": "user", "content": "a"},
                                                             {"role": "user", "content": "b"}]})
    assert text_only.state == "a\n\nb"


@pytest.mark.parametrize("bad", [
    {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "https://x/y.png"}]}]},
    {"input": [{"role": "assistant", "content": "hi"}]},
    {"questions": [{"type": "predicate", "name": "a", "instructions": "x"},
                   {"type": "predicate", "name": "a", "instructions": "y"}]},
    {"questions": [{"type": "choice", "name": "a", "instructions": "x", "choices": [{"value": "only"}]}]},
    {"questions": [{"type": "choice", "name": "a", "instructions": "x",
                    "choices": [{"value": "p"}, {"value": "p"}]}]},
    {"questions": [{"type": "noul", "name": "a", "instructions": "x"}]},
    {"questions": []},
])
def test_rejects_invalid(bad):
    with pytest.raises(RequestError):
        parse_decisions_request({**REQ, **bad})
