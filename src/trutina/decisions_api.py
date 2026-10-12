"""OpenAI-compatible `/v1/decisions` wire format, mapped onto the same internal `Request`.

Request::

    {"model": "...",                                   # accepted, echoed back
     "input": "..." | [{"role": "user", "content": "..." | [<input_text / input_image parts>]}],
     "questions": [
        {"type": "predicate", "name": "<id>", "instructions": "..."},
        {"type": "choice", "name": "<id>", "instructions": "...",
         "choices": [{"value": "<opt>", "description": "..."}, ...]},
        {"type": "score", "name": "<id>", "instructions": "...",
         "levels": [{"label": "<lvl0>", "description": "..."}, ...]}],   # lowest to highest
     "depth": "shallow" | "adaptive" | "deep"}         # optional, trutina extension

Response::

    {"model": "...", "answers": [
        {"type": "predicate", "name": "<id>", "probability": p},
        {"type": "choice", "name": "<id>", "choice": "<opt>", "confidence": c,
         "probabilities": [{"value": "<opt>", "probability": p}, ...]},
        {"type": "score", "name": "<id>", "score": s, "confidence": c,
         "probabilities": [{"value": 0, "label": "<lvl0>", "probability": p}, ...]}]}

`predicate` is a `noul` question; `score` is the probability-weighted level index;
`confidence` is 1 - H(p) / log K as in `trutina.api`. Multiple input messages are joined
with blank lines. Images must be inline `data:` URLs, as in OpenAI's API.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from trutina.api import (
    DEPTHS,
    MAX_CHOICE_OPTIONS,
    SCORE_LEVELS,
    Engine,
    _require,
    _text,
    confidence,
    parse_parts,
)
from trutina.schema import Question, Request, State
from trutina.vision import ImageRef

ROLES = ("user", "developer", "system")


def _parse_input(value: object) -> State:
    if isinstance(value, str):
        return value
    _require(isinstance(value, list) and len(value) > 0, "input must be a string or a non-empty array of messages")
    parts: list[str | ImageRef] = []
    for i, msg in enumerate(value):
        at = f"input[{i}]"
        _require(isinstance(msg, dict) and msg.get("role", "user") in ROLES,
                 f"{at} must be a message with role in {list(ROLES)}")
        content = msg.get("content")
        if parts:
            parts.append("\n\n")
        parts += [content] if isinstance(content, str) else list(parse_parts(content, f"{at}.content", None))
    if all(isinstance(p, str) for p in parts):
        return "".join(parts)
    return tuple(parts)


def _entries(q: dict, key: str, label: str, at: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    items = q.get(key)
    _require(isinstance(items, list), f"{at}.{key} must be an array")
    names, descs = [], []
    for j, item in enumerate(items):
        _require(isinstance(item, dict), f"{at}.{key}[{j}] must be an object")
        names.append(_text(item.get(label), f"{at}.{key}[{j}].{label}"))
        descs.append(_text(item.get("description", ""), f"{at}.{key}[{j}].description"))
    return tuple(names), tuple(descs)


def parse_decisions_request(body: object) -> tuple[Request, str | None]:
    _require(isinstance(body, dict), "request body must be a JSON object")
    model = body.get("model")
    _require(model is None or isinstance(model, str), "model must be a string")
    state = _parse_input(body.get("input"))
    questions = body.get("questions")
    _require(isinstance(questions, list) and len(questions) > 0, "questions must be a non-empty array")
    depth = body.get("depth", "adaptive")
    _require(depth in DEPTHS, f"depth must be one of {list(DEPTHS)}")

    parsed, seen = [], set()
    for i, q in enumerate(questions):
        at = f"questions[{i}]"
        _require(isinstance(q, dict), f"{at} must be an object")
        name = q.get("name")
        _require(isinstance(name, str) and name != "", f"{at}.name must be a non-empty string")
        _require(name not in seen, f"{at}.name {name!r} is not unique")
        seen.add(name)
        kind = q.get("type")
        instructions = _text(q.get("instructions"), f"{at}.instructions")
        if kind == "predicate":
            parsed.append(Question(name, instructions, "noul"))
        elif kind == "choice":
            opts, descs = _entries(q, "choices", "value", at)
            _require(2 <= len(opts) <= MAX_CHOICE_OPTIONS,
                     f"{at}: choice needs 2..{MAX_CHOICE_OPTIONS} choices, got {len(opts)}")
            _require(len(set(opts)) == len(opts), f"{at}: choice values must be unique")
            parsed.append(Question(name, instructions, "choice", opts, descs))
        elif kind == "score":
            labels, descs = _entries(q, "levels", "label", at)
            lo, hi = SCORE_LEVELS
            _require(lo <= len(labels) <= hi, f"{at}: score needs {lo}..{hi} levels, got {len(labels)}")
            parsed.append(Question(name, instructions, "score", labels, descs))
        else:
            _require(False, f"{at}.type must be one of ['predicate', 'choice', 'score']")
    return Request(state=state, questions=tuple(parsed), depth=depth), model


def render_decision(q: Question, probs: Sequence[float]) -> dict:
    p = np.asarray(probs, dtype=np.float64)
    opts = q.resolved_options()
    if len(p) != len(opts) or not np.all(p >= 0) or abs(p.sum() - 1) > 1e-3:
        raise ValueError(f"engine returned invalid probabilities for {q.id!r}: {p}")
    p = p / p.sum()
    if q.kind == "noul":
        return {"type": "predicate", "name": q.id, "probability": float(p[1])}
    if q.kind == "score":
        return {"type": "score", "name": q.id, "score": float((np.arange(len(p)) * p).sum()),
                "confidence": confidence(p),
                "probabilities": [{"value": i, "label": lbl, "probability": float(v)}
                                  for i, (lbl, v) in enumerate(zip(opts, p))]}
    return {"type": "choice", "name": q.id, "choice": opts[int(np.argmax(p))],
            "confidence": confidence(p),
            "probabilities": [{"value": o, "probability": float(v)} for o, v in zip(opts, p)]}


def handle_decisions(body: object, engine: Engine) -> dict:
    """Validate an OpenAI-style decisions request, run the engine, render the response."""
    req, model = parse_decisions_request(body)
    probs = engine(req)
    out = {"answers": [render_decision(q, probs[q.id]) for q in req.questions]}
    return {"model": model, **out} if model else out
