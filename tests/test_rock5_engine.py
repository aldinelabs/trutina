import numpy as np
import pytest

from trutina.api import parse_request
from trutina.rock5_engine import Rock5Trutina


def request(depth="adaptive", image=False):
    body = {"state": "input", "depth": depth,
            "questions": {"q": {"type": "noul", "instructions": "yes?"}}}
    if image:
        body["state"] = [{"type": "input_image", "image_url": "data:image/png;base64,AA=="}]
    return parse_request(body)


def make_manager():
    events = []
    active = []
    class Backend:
        def __init__(self, name):
            assert not active, "a second model was loaded before releasing the first"
            self.name, self.last_exit = name, 32 if name == "npu" else 16
            active.append(name)
            events.append("load:"+name)

        def __call__(self, req):
            events.append("run:"+self.name)
            return {q.id: np.array([0.1, 0.9]) for q in req.questions}

        def close(self):
            events.append("close:"+self.name)
            active.clear()
    return Rock5Trutina({}, lambda: Backend("cpu"), lambda: Backend("npu")), events, active


def test_routes_original_depths_and_images_without_two_resident_models():
    model, events, active = make_manager()
    model(request())
    model(request("shallow"))
    assert events == ["load:cpu", "run:cpu", "run:cpu"]
    model(request("deep"))
    assert model.last_backend == "npu" and model.last_exit == 32
    model(request("deep", image=True))
    assert model.last_backend == "cpu" and model.last_exit == 16
    assert events[-3:] == ["close:npu", "load:cpu", "run:cpu"]
    model.close()
    model.close()
    assert not active and model.engine is None


def test_failed_switch_leaves_no_resident_model_and_can_retry():
    model, _events, active = make_manager()
    model(request())
    factory = model.factories["npu"]
    def fail():
        assert not active
        raise RuntimeError("initialization failed")
    model.factories["npu"] = fail
    with pytest.raises(RuntimeError, match="initialization"):
        model(request("deep"))
    assert not active and model.engine is None
    model.factories["npu"] = factory
    model(request("deep"))
    assert model.last_backend == "npu"
    model.close()


def test_mismatched_releases_are_rejected_before_loading(tmp_path):
    import json

    cpu, npu = tmp_path / "cpu", tmp_path / "npu"
    cpu.mkdir()
    npu.mkdir()
    (cpu / "trutina.json").write_text(json.dumps({"name": "v1"}))
    (npu / "trutina.json").write_text(json.dumps({"name": "v2"}))
    with pytest.raises(ValueError, match="same Trutina release"):
        Rock5Trutina.load(str(cpu), str(npu))


def test_npu_context_cap_falls_back_to_cpu_after_release():
    from trutina.rkllm_engine import RKLLMContextLimit

    model, events, active = make_manager()
    factory = model.factories["npu"]
    def limited_factory():
        engine = factory()
        class Limited:
            last_exit = 32
            def __call__(self, req):
                raise RKLLMContextLimit("context exceeded")
            def close(self):
                engine.close()
        return Limited()
    model.factories["npu"] = limited_factory
    model(request("deep"))
    assert events == ["load:npu", "close:npu", "load:cpu", "run:cpu"]
    assert model.last_backend == "cpu" and active == ["cpu"]
    model.close()
