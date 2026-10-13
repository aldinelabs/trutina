"""Repeatable board smoke/latency probe; writes measured results, never estimates."""
from __future__ import annotations

import argparse
import base64
import io
import json
import resource
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", choices=["torch", "rkllm", "rock5"], default="torch")
    ap.add_argument("--rkllm-model", help="rock5: converted NPU bundle")
    ap.add_argument("--library", default="build/rkllm/libtrutina_rkllm.so")
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--cpu-int8", action="store_true")
    ap.add_argument("--out", default="rock5-results.json")
    a = ap.parse_args()
    start = time.perf_counter()
    if a.engine == "torch":
        import torch

        from trutina.torch_engine import TorchTrutina

        torch.set_num_threads(4)
        model = TorchTrutina.load(a.model, device="cpu", batch_tokens=512,
                                  cpu_int8=a.cpu_int8, vision=a.vision)
    elif a.engine == "rock5":
        from trutina.rock5_engine import Rock5Trutina

        if not a.rkllm_model:
            ap.error("--engine rock5 requires --rkllm-model")
        model = Rock5Trutina.load(a.model, a.rkllm_model, library=a.library)
    else:
        from trutina.rkllm_engine import RKLLMTrutina

        model = RKLLMTrutina.load(a.model, library=a.library)
    results = {"engine": a.engine, "cpu_int8": a.cpu_int8,
               "load_seconds": time.perf_counter()-start, "cases": []}
    print(json.dumps({"loaded": results["load_seconds"]}), flush=True)
    requests = []
    for depth in ["adaptive", "deep"] if a.engine == "rkllm" else ["shallow", "adaptive", "deep"]:
        requests.append((depth, {"state": "The invoice total is $1,240.", "depth": depth,
                                  "questions": {"over": {"type": "noul", "instructions":
                                                        "Is the total over $1,000?"}}}))
    requests.append(("types", {"state": "A red apple.", "depth": "deep", "questions": {
        "color": {"type": "choice", "instructions": "What color is the apple?",
                  "criteria": {"red": "", "blue": "", "green": ""}},
        "match": {"type": "score", "instructions": "How well does red describe the apple?",
                  "criteria": ["wrong", "partly", "exactly"]}}}))
    for name, amount in [("independent_false", 240), ("independent_true_again", 1240)]:
        requests.append((name, {"state": f"The invoice total is ${amount:,}.", "depth": "deep",
                               "questions": {"over": {"type": "noul", "instructions":
                                                     "Is the total over $1,000?"}}}))
    requests.append(("wide", {"state": "The exact number is 27.", "depth": "deep", "questions": {
        "number": {"type": "choice", "instructions": "Which number was stated?",
                   "criteria": {str(i): "" for i in range(1, 28)}}}}))
    if a.vision:
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (256, 256), "red").save(buf, format="PNG")
        requests.append(("image", {"state": [{"type": "input_image", "image_url":
            "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()}],
            "questions": {"color": {"type": "choice", "instructions": "What color fills the image?",
                                     "criteria": {"red": "", "blue": "", "green": ""}}}}))
    for name, req in requests:
        start = time.perf_counter()
        result = model.decide(req)
        row = {"name": name, "seconds": time.perf_counter()-start,
               "exit": model.last_exit, "result": result}
        if getattr(model, "last_backend", None):
            row["backend"] = model.last_backend
        answers = result["answers"]
        if name in ["shallow", "adaptive", "deep", "independent_true_again"]:
            row["correct"] = answers["over"]["noul"] > 0.5
        elif name == "independent_false":
            row["correct"] = answers["over"]["noul"] < 0.5
        elif name in ["types", "image"]:
            row["correct"] = answers["color"]["choice"] == "red"
        else:
            row["correct"] = answers["number"]["choice"] == "27"
        results["cases"].append(row)
        results["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        Path(a.out).write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    if hasattr(model, "close"):
        model.close()
    if not all(row["correct"] for row in results["cases"]):
        raise SystemExit("Some smoke decisions failed; inspect the saved results")


if __name__ == "__main__":
    main()
