"""Probe an existing combined or resident NPU adaptive HTTP service."""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.request
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default="rock5-http-results.json")
    ap.add_argument("--backend", help="expect this resident backend for every request")
    ap.add_argument("--api-key-file", help="send this file's key as a bearer token")
    a = ap.parse_args()
    from PIL import Image

    invoice = {"state": "The invoice total is $1,240.", "depth": "deep",
               "questions": {"over": {"type": "noul", "instructions": "Is the total over $1,000?"}}}
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), "red").save(buf, format="PNG")
    image = {"state": [{"type": "input_image", "image_url":
             "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()}],
             "questions": {"color": {"type": "choice", "instructions": "What color fills the image?",
                                      "criteria": {"red": "", "blue": "", "green": ""}}}}
    predicate = {"model": "trutina-4b", "input": invoice["state"],
                 "questions": [{"type": "predicate", "name": "over", "instructions":
                                 "Is the total over $1,000?"}]}
    cases = [("deep", "/v1/systemone", invoice, "npu"),
             ("image", "/v1/systemone", image, "cpu"),
             ("adaptive", "/v1/decisions", predicate, "cpu"),
             ("deep_again", "/v1/systemone", invoice, "npu")]
    headers = {"Content-Type": "application/json"}
    if a.api_key_file:
        headers["Authorization"] = "Bearer " + Path(a.api_key_file).read_text().strip()
    results = []
    for name, route, body, expected in cases:
        start = time.perf_counter()
        req = urllib.request.Request(a.url+route, data=json.dumps(body).encode(),
                                     headers=headers)
        with urllib.request.urlopen(req, timeout=360) as r:
            row = {"name": name, "status": r.status, "seconds": time.perf_counter()-start,
                   "backend": r.headers.get("X-Trutina-Backend"),
                   "exit": r.headers.get("X-Trutina-Exit-Layer"), "result": json.loads(r.read())}
        assert row["backend"] == (a.backend or expected), row
        answers = row["result"]["answers"]
        if name == "image":
            assert answers["color"]["choice"] == "red", row
        elif name == "adaptive":
            assert answers[0]["probability"] > 0.5 and int(row["exit"]) < 32, row
        else:
            assert answers["over"]["noul"] > 0.5 and row["exit"] == "32", row
        results.append(row)
        Path(a.out).write_text(json.dumps(results, indent=2)+"\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
