"""Guard the experimental continuity gate against misleading SDK success."""
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "probe_segments", Path(__file__).parents[1] / "scripts/probe_rkllm_segments.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def references(path):
    (path / "probe.json").write_text(json.dumps({"hidden": 4, "lengths": [3]}))
    data = np.linspace(0.02, 0.08, 12, dtype="<f4")
    for name in ("full-cpu", "first-raw-cpu", "first-normalized-cpu", "full-npu",
                 "first-raw-npu", "first-normalized-npu", "second-cpu-input-npu", "chained-npu"):
        data.tofile(path / f"{name}-3.f32")


def test_continuity_gate_rejects_normalized_residual_as_raw(tmp_path):
    references(tmp_path)
    raw = np.fromfile(tmp_path / "first-raw-cpu-3.f32", dtype="<f4")
    (raw * 25).tofile(tmp_path / "first-raw-npu-3.f32")
    with pytest.raises(RuntimeError, match="continuity gate failed"):
        probe.compare(tmp_path)
    report = json.loads((tmp_path / "comparison.json").read_text())
    assert not report["passed"]
    assert next(m for m in report["measurements"] if m["name"] == "first-raw-npu")["relative_l2"] > 20


def test_continuity_gate_requires_all_tokens(tmp_path):
    references(tmp_path)
    np.ones(4, dtype="<f4").tofile(tmp_path / "first-raw-npu-3.f32")
    with pytest.raises(ValueError, match="Full-sequence"):
        probe.compare(tmp_path)


def test_continuity_gate_accepts_small_numeric_error(tmp_path):
    references(tmp_path)
    got = np.fromfile(tmp_path / "chained-npu-3.f32", dtype="<f4")
    (got * 1.01).tofile(tmp_path / "chained-npu-3.f32")
    probe.compare(tmp_path)
    assert json.loads((tmp_path / "comparison.json").read_text())["passed"]


def test_sdk_exit_zero_without_completion_is_failure(tmp_path, monkeypatch):
    def fatal(*args, **kwargs):
        (tmp_path / "full.rkllm").write_bytes(b"")
        return subprocess.CompletedProcess(args[0], 0, "ERROR: unsupported dimension", "")
    monkeypatch.setattr(probe.subprocess, "run", fatal)
    with pytest.raises(RuntimeError, match="conversion is unverified"):
        probe.export(tmp_path, "w8a8")
    report = json.loads((tmp_path / "conversion.json").read_text())
    assert report["full"]["returncode"] == 0 and not report["full"]["success"]
