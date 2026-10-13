"""HTTP server for trutina-4b: POST /v1/systemone and POST /v1/decisions (JSON in, JSON out).

  uv run python scripts/serve.py --port 8000
  curl -s localhost:8000/v1/systemone -d '{"state": "...", "questions": {"q": {"type": "noul", "instructions": "..."}}}'

One request at a time (one GPU stream); requests queue on a lock. `--engine torch` serves on CUDA
(text only, full depth) instead of MLX.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from trutina.api import RequestError, handle
from trutina.decisions_api import handle_decisions

ROUTES = {"/v1/systemone": handle, "/v1/decisions": handle_decisions}


def make_handler(engine, max_body: int):
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send(self, code: int, obj: dict, extra: dict | None = None):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self.send(200, {"status": "ok", "model": engine.config["name"]})
            self.send(404, {"error": {"status": 404, "message": "not found"}})

        def do_POST(self):
            route = ROUTES.get(self.path.split("?")[0])
            if route is None:
                return self.send(404, {"error": {"status": 404, "message": "not found"}})
            n = int(self.headers.get("Content-Length") or 0)
            if n > max_body:
                return self.send(413, {"error": {"status": 413, "message": f"body over {max_body} bytes"}})
            try:
                body = json.loads(self.rfile.read(n) or b"null")
            except ValueError:
                return self.send(400, {"error": {"status": 400, "message": "body is not JSON"}})
            try:
                with lock:
                    t0 = time.perf_counter()
                    out = route(body, engine)
                    ms = (time.perf_counter() - t0) * 1000
                    layer = getattr(engine, "last_exit", None) or 32
            except RequestError as e:
                return self.send(e.status, {"error": {"status": e.status, "message": str(e)}})
            except ValueError as e:  # e.g. an image that cannot be decoded
                return self.send(HTTPStatus.BAD_REQUEST, {"error": {"status": 400, "message": str(e)}})
            self.send(200, out, {"X-Trutina-Ms": f"{ms:.1f}", "X-Trutina-Exit-Layer": str(layer)})

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="aldinelabs/trutina-4b")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-body", type=int, default=64 << 20)
    ap.add_argument("--engine", choices=["mlx", "torch"], default="mlx")
    ap.add_argument("--device", default="cuda", help="torch engine only")
    ap.add_argument("--noul", choices=["decisive", "calibrated"], default=None,
                    help="yes/no readout (default: decisive when the release ships the map)")
    ap.add_argument("--no-cuda-graphs", action="store_true", help="torch engine only: run eagerly")
    ap.add_argument("--warm-graphs", type=int, default=4096,
                    help="torch engine only: capture the length buckets up to this many tokens at start-up")
    a = ap.parse_args()
    if a.engine == "torch":
        from trutina.torch_engine import TorchTrutina

        engine = TorchTrutina.load(a.model, device=a.device, cuda_graphs=not a.no_cuda_graphs)
        if engine.graph_tokens and a.warm_graphs:
            print(f"captured CUDA graphs for {engine.warm_graphs(a.warm_graphs)} length buckets", flush=True)
    else:
        from trutina.engine import Trutina

        engine = Trutina.load(a.model)
    if a.noul:
        engine.noul = a.noul
    server = ThreadingHTTPServer((a.host, a.port), make_handler(engine, a.max_body))
    print(f"trutina-4b on http://{a.host}:{a.port}  (POST /v1/systemone, /v1/decisions)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
