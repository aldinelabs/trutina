"""Run discovery and HTTP access through nested group names."""

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import urlopen

import pytest

from scripts.webui import Runs, handler


def make_run(root, name, *, launch=False):
    d = root / name
    d.mkdir(parents=True)
    marker = "launch.json" if launch else "config.json"
    (d / marker).write_text(json.dumps({"framework": "torch", "train": {"steps": 7500}}))
    (d / "status.json").write_text(json.dumps({"state": "done", "step": 7500}))
    return d


def test_nested_runs_are_listed_and_reachable(tmp_path):
    runs = Runs(tmp_path, [])
    names = {"flat", "private/v2.1b", "private/v3/main", "group/deep/local"}
    for name in names:
        make_run(tmp_path, name, launch=name.endswith("local"))
    # A run's artifacts are not separate runs, and invalid group names are skipped.
    make_run(tmp_path, "private/v3/main/artifacts/copy")
    make_run(tmp_path, ".hidden/run")
    make_run(tmp_path, "invalid group/run")
    assert {r["name"] for r in runs.list()} == names
    assert runs.detail("private/v3/main")["status"]["step"] == 7500


@pytest.mark.parametrize("name", ["../outside", "private/../outside", "/absolute", "a//b", "a/b/", "a\n"])
def test_invalid_paths_are_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        Runs(tmp_path, []).dir(name)


def test_discovery_skips_symlinks_and_access_cannot_escape(tmp_path):
    root, outside = tmp_path / "runs", tmp_path / "outside"
    runs = Runs(root, [])
    make_run(outside, "main")
    (root / "link").symlink_to(outside, target_is_directory=True)
    assert runs.list() == []
    with pytest.raises(ValueError):
        runs.detail("link/main")


def test_http_nested_detail_with_encoded_or_decoded_slashes(tmp_path):
    make_run(tmp_path, "private/v3/main")
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(Runs(tmp_path, []), read_only=True))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for name in ("private%2Fv3%2Fmain", "private/v3/main"):
            with urlopen(f"http://127.0.0.1:{server.server_port}/api/runs/{name}") as response:
                assert json.load(response)["name"] == "private/v3/main"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_live_comparison_preserves_title_phase_units_and_notes(tmp_path):
    directory = tmp_path / "bench/compete-v3-evaluation"
    directory.mkdir(parents=True)
    for model, done in (("v2.1", 4), ("v3", 5)):
        (directory / f"{model}.json").write_text(json.dumps({
            "system": model, "title": "v3 evaluation vs v2.1", "progress_unit": "phases",
            "focus_system": "v3", "notes": ["Fixed DI sample; partial results."],
            "status": "running", "done": done, "total": 7,
            "sets": {"DI": {"metric": "index", "group": "index", "score": 44.71, "n": 2400}},
        }))
    runs = Runs(tmp_path, [])
    listing = runs.bench_list()
    assert listing[0]["title"] == "v3 evaluation vs v2.1"
    assert listing[0]["running"] is True
    report = runs.bench("compete-v3-evaluation")
    assert report["headline"]["value"] == 9 / 14
    assert report["progress_unit"] == "phases"
    assert report["focus_system"] == "v3"
    assert report["notes"] == ["Fixed DI sample; partial results."]
    assert report["rows"][0]["metric"] == "index"
    assert report["rows"][0]["scores"]["v2.1"] == 44.71
    assert report["rows"][0]["group"] == "index"
