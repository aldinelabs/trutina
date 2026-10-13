"""Decision protocol tests without an NPU; native hardware probes are separate."""
import json
import string
import sys
from pathlib import Path

import numpy as np
import pytest

from trutina.api import RequestError
from trutina.rkllm_engine import RKLLMTrutina


class Tokenizer:
    def encode(self, text, **kw):
        if len(text) == 1 and text in string.ascii_uppercase:
            return [ord(text)]
        if len(text) == 2 and text.isupper():
            return [256+ord(text[0])*26+ord(text[1])]
        return list(text.encode())

    def apply_chat_template(self, messages, **kw):
        return messages[0]["content"] + "\nassistant: "


class Runtime:
    def __init__(self):
        self.calls = []

    def logits(self, tokens, labels):
        self.calls.append((tokens, labels))
        return np.arange(len(labels), dtype=np.float32)


def engine(scale=2, max_context=4096):
    config = {"name": "test", "format_version": 1, "exits": [12, 32]}
    return RKLLMTrutina(Runtime(), Tokenizer(), config, scale, max_context)


def body(kind="choice", k=3):
    q = {"type": kind, "instructions": "pick"}
    if kind == "choice":
        q["criteria"] = {f"o{i}": "" for i in range(k)}
    if kind == "score":
        q["criteria"] = [f"o{i}" for i in range(k)]
    return {"state": "input", "questions": {"q": q}}


@pytest.mark.parametrize("kind", ["choice", "score", "noul"])
def test_release_scale_is_applied_to_logits_before_softmax(kind):
    model = engine()
    out = model.decide(body(kind))["answers"]["q"]
    k = 2 if kind == "noul" else 3
    ref = np.exp(np.arange(k)*2-(k-1)*2)
    ref /= ref.sum()
    if kind == "noul":
        assert out["noul"] == pytest.approx(ref[1])
    else:
        np.testing.assert_allclose(list(out["probabilities"].values()), ref)
    assert model.runtime.calls[0][1] == list(range(65, 65+k))
    assert model.last_exit == 32


def test_wide_choices_return_every_probability_and_use_narrow_passes():
    model = engine()
    out = model.decide(body(k=255))["answers"]["q"]
    assert len(out["probabilities"]) == 255
    assert sum(out["probabilities"].values()) == pytest.approx(1)
    assert len(model.runtime.calls) == 11
    assert all(len(labels) <= 26 for _, labels in model.runtime.calls)


def test_unsupported_depth_and_images_never_reach_native_runtime():
    model = engine()
    req = body()
    req["depth"] = "shallow"
    with pytest.raises(RequestError, match="layer 12"):
        model.decide(req)
    req.pop("depth")
    req["state"] = [{"type": "input_image", "image_url": "data:image/png;base64,AA=="}]
    with pytest.raises(RequestError, match="text-only"):
        model.decide(req)
    assert not model.runtime.calls


def test_context_limit_refuses_truncation():
    model = engine(max_context=20)
    with pytest.raises(RequestError, match="exceed"):
        model.decide(body())
    assert not model.runtime.calls


def test_decisive_mapping_uses_release_parameters():
    from trutina.readout import decisive

    model = engine()
    model.config["noul_decisive"] = {"T": 0.8, "b": 0.2}
    model.noul = "decisive"
    got = model.decide(body("noul"))["answers"]["q"]["noul"]
    assert got == pytest.approx(decisive(1/(1+np.exp(-2)), 0.8, 0.2))


def test_invalid_native_logits_are_rejected():
    model = engine()
    model.runtime.logits = lambda *args: np.array([np.nan]*3)
    with pytest.raises(RuntimeError, match="invalid"):
        model.decide(body())


def test_calibration_uses_public_prompt_and_groups(tmp_path):
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    from export_rkllm import calibration_inputs

    path = tmp_path / "requests.jsonl"
    path.write_text(json.dumps(body(k=27)) + "\n")
    samples = calibration_inputs(path, Tokenizer(), 10000)
    assert len(samples) == 2
    assert all(s["input"].endswith("Best answer: (") for s in samples)
    assert all("(AA)" not in s["input"] for s in samples)
