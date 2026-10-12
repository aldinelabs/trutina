"""Answer one /v1/systemone or OpenAI-style /v1/decisions request (JSON on stdin or --file).

  echo '{"state": "...", "questions": {...}}' | uv run python scripts/decide.py
  echo '{"input": "...", "questions": [...]}' | uv run python scripts/decide.py --format decisions
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from trutina.api import RequestError, handle
from trutina.decisions_api import handle_decisions
from trutina.engine import Trutina


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="aldinelabs/trutina-4b", help="Hugging Face repo id or release directory")
    ap.add_argument("--file", help="request JSON file (default: stdin)")
    ap.add_argument("--format", choices=["systemone", "decisions"], default="systemone")
    a = ap.parse_args()

    body = json.loads(Path(a.file).read_text() if a.file else sys.stdin.read())
    engine = Trutina.load(a.model)
    try:
        t0 = time.perf_counter()
        out = (handle_decisions if a.format == "decisions" else handle)(body, engine)
        print(f"{(time.perf_counter() - t0) * 1000:.0f} ms, answered at layer {engine.last_exit}", file=sys.stderr)
    except RequestError as e:
        out = {"error": {"status": e.status, "message": str(e)}}
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
