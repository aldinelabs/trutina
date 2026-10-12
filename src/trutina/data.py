"""Framework-free training data: JSONL rows -> (state, question, target distribution).

Each row is a /v1/systemone request plus labels::

    {"state": "...", "questions": {"q": {"type": "choice", ...}}, "labels": {"q": <label>}}

`state` may also be a list of text / image parts (`trutina.api`); image paths are relative
to the JSONL file. A label is an option name (choice), a level index or label (score), a bool or P(yes)
(noul), or a full probability list in option order (teacher distillation). Extra keys
(e.g. `meta`) are ignored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trutina.api import parse_request
from trutina.schema import Question, State


@dataclass(frozen=True)
class Example:
    state: State
    question: Question
    target: np.ndarray  # probabilities in option order
    source: str = ""    # who labelled it (meta.teacher, else meta.dataset); "" = unknown
    group: str = ""     # meta.dataset / meta.source (sub-domain), for balanced sampling


def target_vector(q: Question, label: object) -> np.ndarray:
    opts = q.resolved_options()
    if isinstance(label, list):
        t = np.asarray(label, dtype=np.float32)
        if len(t) != len(opts) or abs(t.sum() - 1) > 1e-3:
            raise ValueError(f"{q.id}: distribution must cover {len(opts)} options and sum to 1")
        return t / t.sum()
    if q.kind == "noul":
        p = float(label)  # bool or P(yes)
        return np.array([1 - p, p], dtype=np.float32)
    idx = label if isinstance(label, int) and q.kind == "score" else opts.index(label)
    t = np.zeros(len(opts), dtype=np.float32)
    t[idx] = 1.0
    return t


def load_examples(path: str | Path) -> list[Example]:
    out, base = [], str(Path(path).parent)
    for line in Path(path).read_text().split("\n"):  # not splitlines(): \u2028 can occur in text
        if line.strip():
            row = json.loads(line)
            req = parse_request({k: v for k, v in row.items() if k not in ("labels", "meta")}, base)
            meta = row.get("meta") or {}
            teacher = meta.get("teacher")
            src = str(teacher if teacher and teacher != "human" else meta.get("dataset", ""))
            group = f"{meta.get('dataset', '')}/{meta.get('source', '')}"
            out += [Example(req.state, q, target_vector(q, row["labels"][q.id]), src, group)
                    for q in req.questions]
    return out


def load_all(paths: str | Path | list) -> list[Example]:
    """Examples from one or more JSONL files, in order."""
    return [e for p in ([paths] if isinstance(paths, (str, Path)) else paths) for e in load_examples(p)]
