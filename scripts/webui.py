"""Local training dashboard: watch runs, start, stop and resume MLX training.

  uv run python scripts/webui.py                  # this Mac only
  uv run python scripts/webui.py --host 0.0.0.0   # also viewable from the local network

Reads the run directories described in `trutina.runs` under --runs. Starting, stopping
and resuming runs is allowed only from this machine (loopback clients); other devices get
a read-only view. With TRUTINA_WEBUI_AUTH=user:password set, every request needs that
HTTP basic auth (for exposing the read-only view beyond the local network).
"""

from __future__ import annotations

import argparse
import base64
import hmac
import ipaddress
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "webui" / "index.html"
SEGMENT = r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}"
NAME = re.compile(rf"{SEGMENT}(/{SEGMENT})*")  # runs may be nested inside multiple groups
INT_FLAGS = ("steps", "warmup", "batch_size", "grad_accum", "rank", "head_dim", "max_tokens", "batch_tokens",
             "eval_every", "ckpt_every", "seed")


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Runs:
    def __init__(self, root: Path, data_roots: list[Path]):
        self.root, self.data_roots = root, data_roots
        root.mkdir(parents=True, exist_ok=True)

    def dir(self, name: str) -> Path:
        if not NAME.fullmatch(name):
            raise ValueError("Run names use letters, digits, dot, dash and underscore in slash-separated groups.")
        d = self.root / name
        if not d.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("Run paths must stay inside the runs directory.")
        return d

    def status(self, d: Path, config: dict) -> dict:
        st = read_json(d / "status.json", {}) or {}
        state = st.get("state", "starting" if (d / "stdout.log").exists() else "unknown")
        if state in ("running", "loading"):
            if str(config.get("framework", "")).startswith("torch"):  # remote (e.g. Modal) run synced here
                state = state if time.time() - st.get("updated", 0) < 300 else "stale"
            elif not pid_alive(st.get("pid")):
                state = "interrupted"
        if state == "starting" and not pid_alive((read_json(d / "launch.json", {}) or {}).get("pid")):
            state = "failed"
        return {**st, "state": state, "steps": st.get("steps") or config.get("train", {}).get("steps")}

    def summary(self, d: Path) -> dict:
        config = read_json(d / "config.json", {}) or {}
        st = self.status(d, config)
        last = None
        for rec in self.metrics(d)[::-1]:
            if rec.get("type") == "train":
                last = rec
                break
        return {"name": str(d.relative_to(self.root)), "framework": config.get("framework"), "status": st,
                "loss": last and last["loss"], "modified": (d / "status.json").stat().st_mtime
                if (d / "status.json").exists() else d.stat().st_mtime}

    def list(self) -> list[dict]:
        def is_run(d: Path) -> bool:
            return (d / "config.json").exists() or (d / "launch.json").exists()

        dirs, groups = [], [self.root]
        while groups:
            for d in groups.pop().iterdir():
                if d.is_symlink() or not d.is_dir() or not re.fullmatch(SEGMENT, d.name):
                    continue
                if is_run(d):
                    dirs.append(d)
                else:
                    groups.append(d)
        return sorted((self.summary(d) for d in dirs), key=lambda r: -r["modified"])

    @staticmethod
    def metrics(d: Path) -> list[dict]:
        out = []
        # metrics-*.jsonl: records recovered from elsewhere (e.g. logs of an interrupted run)
        for f in [*sorted(d.glob("metrics-*.jsonl")), d / "metrics.jsonl"]:
            out += Runs._read_metrics(f)
        # a resumed run re-logs the steps after its checkpoint: keep the latest record per step
        latest = {(r["type"], r.get("step")): r for r in out}
        return sorted(latest.values(), key=lambda r: (r.get("step") or 0, r["type"]))

    @staticmethod
    def _read_metrics(f: Path) -> list[dict]:
        out = []
        try:
            for line in f.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue  # line being written
                if "type" not in rec:  # early torch format: eval rows without a type
                    rec = {"type": "eval", "step": rec.pop("step"), "exits": rec}
                out.append(rec)
        except OSError:
            pass
        return out

    def detail(self, name: str) -> dict:
        d = self.dir(name)
        if not d.exists():
            raise FileNotFoundError(name)
        config = read_json(d / "config.json", {}) or {}
        log = ""
        for fname in ("train.log", "stdout.log"):
            p = d / fname
            if p.exists() and p.stat().st_size:
                with p.open("rb") as f:
                    f.seek(max(0, p.stat().st_size - 48_000))
                    log = f.read().decode(errors="replace")
                if fname == "train.log" or not (d / "train.log").exists():
                    break
        return {**self.summary(d), "config": config, "metrics": self.metrics(d), "log": log,
                "exported": (d / "trainable.safetensors").exists(), "path": str(d),
                "best": read_json(d / "best.json")}

    def bench_list(self) -> list[dict]:
        """Benchmark reports: <runs>/bench/*.json (see lab/eval/bench_report.py), newest first."""
        out = []
        for f in (self.root / "bench").glob("*.json"):
            r = read_json(f)
            if isinstance(r, dict) and r.get("rows") is not None:
                out.append({"name": f.stem, "title": r.get("title", f.stem), "kind": r.get("kind"),
                            "model": r.get("model"), "headline": r.get("headline"), "created": r.get("created") or f.stat().st_mtime,
                            "modified": f.stat().st_mtime})
        for d in sorted((self.root / "bench").glob("compete*/")):
            r = self.compete(d)
            if r:
                out.append({"name": d.name, "title": r["title"], "kind": "compete", "model": r["model"], "headline": r["headline"],
                            "created": r["created"], "modified": r["modified"], "running": r["running"]})
        return sorted(out, key=lambda r: -r["modified"])

    def compete(self, d: Path) -> dict | None:
        """Head-to-head runs (lab/modal/compete_modal.py): one progress file per system under <runs>/bench/compete*/,
        merged into one report with a score per system for every benchmark."""
        parts = [p for p in (read_json(f) for f in sorted(d.glob("*.json"))) if isinstance(p, dict) and p.get("system")]
        if not parts:
            return None
        order = ["trutina", "spark", "decider", "jevk5", "plumb", "sieve"]
        parts.sort(key=lambda p: order.index(p["system"]) if p["system"] in order else len(order))
        names = sorted({k for p in parts for k in p.get("sets", {})})
        rows = []
        for k in names:
            metric = "pair_acc" if k.startswith("judgebench") else "state_macro_acc" if k.startswith("decisionbench") else "acc"
            scores = {p["system"]: (p["sets"].get(k) or {}).get(metric) for p in parts}
            n = max(((p["sets"].get(k) or {}).get("n") or 0) for p in parts)
            extra = {p["system"]: (p["sets"].get(k) or {}).get("kl") for p in parts} if k == "localllama" else None
            rows.append({"name": k, "group": "accuracy", "metric": metric, "n": n, "scores": scores, "kl": extra,
                         "errors": {p["system"]: (p["sets"].get(k) or {}).get("errors") for p in parts}})
        for key, label in (("latency_jevbench_ms", "latency p50, JevBench public"), ("latency_adaptive_ms", "latency p50, adaptive")):
            if any(p.get(key) for p in parts):
                rows.append({"name": label, "group": "latency", "metric": "ms", "lower": True,
                             "scores": {p["system"]: (p.get(key) or {}).get("p50") for p in parts},
                             "p95": {p["system"]: (p.get(key) or {}).get("p95") for p in parts}})
        modified = max(f.stat().st_mtime for f in d.glob("*.json"))
        running = any(p.get("status") in ("starting", "running") for p in parts)
        done = sum(p.get("done", 0) for p in parts)
        total = sum(p.get("total", 0) for p in parts)
        return {"title": "4B decision models, head to head" + (" (smoke)" if d.name.endswith("smoke") else ""), "kind": "compete",
                "model": " · ".join(p["system"] for p in parts), "created": min(p.get("started", modified) for p in parts),
                "modified": modified, "running": running,
                "headline": {"label": "requests done", "value": done / total if total else None, "fmt": "pct"},
                "systems": [{k: p.get(k) for k in ("system", "repo", "revision", "gpu", "status", "set", "done", "total",
                                                    "started", "updated", "failure", "error_counts")} for p in parts],
                "rows": rows,
                "notes": ["Every system runs in its own container with the server its authors ship, at the pinned revision; one client sends "
                          "the same requests to all of them, one at a time, on the same GPU type, and scores them the same way.",
                          "A request a server refuses counts as wrong. DecisionBench: state-macro accuracy; JudgeBench: pair accuracy "
                          "(both orders right); otherwise accuracy. trutina runs at full depth; 'adaptive' rows are its default mode.",
                          "Latency is HTTP wall time per request on the JevBench public tasks after a warm-up."]}

    def bench(self, name: str) -> dict:
        if not re.fullmatch(SEGMENT, name):
            raise ValueError("Bad report name.")
        if name.startswith("compete") and (self.root / "bench" / name).is_dir():
            r = self.compete(self.root / "bench" / name)
            if r is None:
                raise FileNotFoundError(name)
            return {"name": name, **r}
        r = read_json(self.root / "bench" / f"{name}.json")
        if not isinstance(r, dict):
            raise FileNotFoundError(name)
        return {"name": name, **r}

    def datasets(self) -> list[str]:
        found = []
        for r in self.data_roots:
            if r.exists():
                found += [str(p.relative_to(ROOT)) for p in sorted(r.rglob("*.jsonl"))]
        return found

    def command(self, out: Path, model: str, data: str, val: str | None, train: dict) -> list[str]:
        cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), "--model", model,
               "--data", data, "--out", str(out)]
        if val:
            cmd += ["--val", val]
        for k, v in train.items():
            flag = "--" + k.replace("_", "-")
            if v is None:
                continue
            if isinstance(v, bool):
                cmd.append(flag if v else "--no-" + k.replace("_", "-"))
            elif isinstance(v, list):
                cmd += [flag, *map(str, v)]
            else:
                cmd += [flag, str(v)]
        return cmd

    def launch(self, d: Path, cmd: list[str]) -> None:
        d.mkdir(parents=True, exist_ok=True)
        log = (d / "stdout.log").open("ab")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        (d / "launch.json").write_text(json.dumps({"pid": proc.pid, "cmd": cmd, "time": time.time()}))

    def busy(self) -> str | None:
        for r in self.list():
            if r["status"]["state"] in ("running", "loading", "starting") and r["framework"] != "torch":
                return r["name"]
        return None

    def start(self, body: dict) -> str:
        name = body.get("name", "")
        d = self.dir(name)
        if d.exists():
            raise ValueError(f"A run named {name} already exists. Pick another name.")
        if other := self.busy():
            raise ValueError(f"{other} is still training. Stop it first; one run uses the whole GPU.")
        sets = self.datasets()
        data, val = body.get("data"), body.get("val") or None
        if data not in sets or (val and val not in sets):
            raise ValueError("Choose training and validation files from the list.")
        train = {}
        for k in INT_FLAGS:
            if body.get(k) not in (None, ""):
                train[k] = int(body[k])
        if body.get("lr") not in (None, ""):
            train["lr"] = float(body["lr"])
        if body.get("exits"):
            train["exits"] = [int(x) for x in str(body["exits"]).replace(",", " ").split()]
        if body.get("grad_checkpointing"):
            train["grad_checkpointing"] = True
        self.launch(d, self.command(d, body.get("model") or "Qwen/Qwen3.5-4B-Base", data, val, train))
        return name

    def resume(self, name: str) -> None:
        d = self.dir(name)
        config = read_json(d / "config.json")
        if not config or config.get("framework") != "mlx":
            raise ValueError("Only local MLX runs can be resumed here.")
        if self.status(d, config)["state"] in ("running", "loading", "starting"):
            raise ValueError(f"{name} is already running.")
        if other := self.busy():
            raise ValueError(f"{other} is still training. Stop it first.")
        self.launch(d, self.command(d, config["model"], config["data"], config.get("val"), config["train"]))

    def stop(self, name: str) -> None:
        d = self.dir(name)
        config = read_json(d / "config.json", {}) or {}
        pid = (read_json(d / "status.json", {}) or {}).get("pid") or \
              (read_json(d / "launch.json", {}) or {}).get("pid")
        if config.get("framework") == "torch" or not pid_alive(pid):
            raise ValueError(f"{name} is not running on this machine.")
        os.kill(pid, signal.SIGTERM)  # the loop checkpoints and exits


def handler(runs: Runs, read_only: bool = False, auth: str | None = None):
    expected = "Basic " + base64.b64encode(auth.encode()).decode() if auth else None

    class H(BaseHTTPRequestHandler):
        def authorized(self) -> bool:
            if expected is None or hmac.compare_digest(self.headers.get("Authorization", ""), expected):
                return True
            self.send_response(HTTPStatus.UNAUTHORIZED)
            self.send_header("WWW-Authenticate", 'Basic realm="trutina"')
            self.send_header("Content-Length", "0")
            self.end_headers()
            return False

        def log_message(self, *a):
            pass

        def send(self, code: int, body: bytes, ctype: str = "application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def json(self, obj, code: int = 200):
            self.send(code, json.dumps(obj).encode())

        def guard(self, fn):
            try:
                fn()
            except FileNotFoundError:
                self.json({"error": "No such run."}, HTTPStatus.NOT_FOUND)
            except ValueError as e:
                self.json({"error": str(e)}, HTTPStatus.BAD_REQUEST)

        def local(self) -> bool:
            if read_only:
                return False
            return ipaddress.ip_address(self.client_address[0].removeprefix("::ffff:")).is_loopback

        def do_GET(self):
            if not self.authorized():
                return
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/runs":
                return self.json(runs.list())
            if path == "/api/access":
                return self.json({"control": self.local()})
            if path == "/api/datasets":
                return self.json(runs.datasets())
            if path == "/api/bench":
                return self.json(runs.bench_list())
            if m := re.fullmatch(r"/api/bench/(.+)", path):
                return self.guard(lambda: self.json(runs.bench(unquote(m[1]))))
            if m := re.fullmatch(r"/api/runs/(.+)", path):  # a proxy may decode %2F in group/run
                return self.guard(lambda: self.json(runs.detail(unquote(m[1]))))
            self.json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            if not self.authorized():
                return
            if not self.local():
                return self.json({"error": "Read-only on other devices. Use this Mac to change runs."},
                                 HTTPStatus.FORBIDDEN)
            if self.headers.get("Origin") not in (None, f"http://{self.headers.get('Host')}"):
                return self.json({"error": "Cross-origin request refused."}, HTTPStatus.FORBIDDEN)
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            path = self.path.split("?")[0]
            if path == "/api/runs":
                return self.guard(lambda: self.json({"name": runs.start(body)}))
            if m := re.fullmatch(r"/api/runs/(.+)/(stop|resume)", path):
                act = runs.stop if m[2] == "stop" else runs.resume
                return self.guard(lambda: (act(unquote(m[1])), self.json({"ok": True})))
            self.json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

    return H


def lan_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("192.168.0.1", 9))  # no packet is sent; picks the outbound interface
            return s.getsockname()[0]
        except OSError:
            return "this-mac.local"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--data", nargs="+", default=[str(ROOT / "data"), str(ROOT / "lab" / "data" / "build")],
                    help="directories searched for *.jsonl datasets")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 to serve the local network (read-only for other devices)")
    ap.add_argument("--read-only", action="store_true",
                    help="no starting / stopping runs for anyone (e.g. behind a cloud proxy, where "
                         "every client looks local)")
    a = ap.parse_args()
    runs = Runs(Path(a.runs), [Path(p) for p in a.data])
    server = ThreadingHTTPServer((a.host, a.port), handler(runs, a.read_only, os.environ.get("TRUTINA_WEBUI_AUTH") or None))
    print(f"trutina runs: http://127.0.0.1:{a.port}", flush=True)
    if a.host != "127.0.0.1":
        print(f"read-only on the network: http://{lan_ip()}:{a.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
