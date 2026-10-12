"""`/v1/systemone` wire format: request validation and response rendering.

Request::

    {"state": "..." | [{"type": "input_text", "text": "..."},
                       {"type": "input_image", "image_url": "data:image/png;base64,..."}],
     "questions": {
        "<name>": {"type": "choice", "instructions": "...", "criteria": {"<opt>": "<desc>", ...}},
        "<name>": {"type": "score",  "instructions": "...", "criteria": ["<lvl0>", {"label": "<lvl1>"}, ...]},
        "<name>": {"type": "noul",   "instructions": "...", "criteria": {"true": "...", "false": "..."}}},
     "depth": "shallow" | "adaptive" | "deep"}          # optional, trutina extension

Response::

    {"answers": {
        "<name>": {"type": "choice", "choice": "<opt>", "confidence": c, "probabilities": {"<opt>": p}},
        "<name>": {"type": "score", "score": s, "confidence": c, "legend": {"0": "<lvl0>"}, "probabilities": {"0": p}},
        "<name>": {"type": "noul", "noul": p}}}

A list `state` (trutina extension) interleaves text and images, using the content-part
shapes of OpenAI's Responses / Decisions APIs; images must be inline `data:` URLs.
`score` is the probability-weighted level index; `noul` is P(yes) and carries no confidence.
The public spec does not define `confidence`; we use 1 - H(p) / log K (1 = certain, 0 = uniform).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from trutina.schema import Question, Request, State
from trutina.vision import ImageRef

MAX_CHOICE_OPTIONS = 255
SCORE_LEVELS = (2, 10)
DEPTHS = ("shallow", "adaptive", "deep")
NOUL_CRITERIA_KEYS = {"true", "false"}

# engine: Request -> {question id: probabilities in option order}
Engine = Callable[[Request], Mapping[str, Sequence[float]]]


class RequestError(ValueError):
    """Invalid request; maps to HTTP 400."""

    status = 400


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RequestError(msg)


def _text(value: object, where: str) -> str:
    _require(isinstance(value, str), f"{where} must be a string")
    return value


def _parse_choice(name: str, criteria: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    _require(isinstance(criteria, dict), f"questions.{name}.criteria must be an object")
    n = len(criteria)
    _require(2 <= n <= MAX_CHOICE_OPTIONS,
             f"questions.{name}: choice needs 2..{MAX_CHOICE_OPTIONS} options, got {n}")
    descs = tuple(_text(d, f"questions.{name}.criteria.{k}") for k, d in criteria.items())
    return tuple(criteria), descs


def _parse_score(name: str, criteria: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    _require(isinstance(criteria, list), f"questions.{name}.criteria must be an array of levels")
    lo, hi = SCORE_LEVELS
    _require(lo <= len(criteria) <= hi,
             f"questions.{name}: score needs {lo}..{hi} levels, got {len(criteria)}")
    labels, descs = [], []
    for i, level in enumerate(criteria):
        where = f"questions.{name}.criteria[{i}]"
        if isinstance(level, dict):
            labels.append(_text(level.get("label"), f"{where}.label"))
            descs.append(_text(level.get("description", ""), f"{where}.description"))
        else:
            labels.append(_text(level, where))
            descs.append("")
    return tuple(labels), tuple(descs)


def _parse_noul(name: str, criteria: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if criteria is None:
        return (), ()
    _require(isinstance(criteria, dict), f"questions.{name}.criteria must be an object or null")
    extra = set(criteria) - NOUL_CRITERIA_KEYS
    _require(not extra, f"questions.{name}: noul criteria only allow 'true'/'false', got {sorted(extra)}")
    false = _text(criteria.get("false", ""), f"questions.{name}.criteria.false")
    true = _text(criteria.get("true", ""), f"questions.{name}.criteria.true")
    return (), (false, true)  # aligned with NOUL_OPTIONS = ("no", "yes")


def parse_parts(value: object, where: str, image_base: str | None) -> tuple[str | ImageRef, ...]:
    """OpenAI-style content parts: `input_text` {text} / `input_image` {image_url}."""
    _require(isinstance(value, list) and len(value) > 0, f"{where} must be a non-empty array of parts")
    parts: list[str | ImageRef] = []
    for i, part in enumerate(value):
        at = f"{where}[{i}]"
        _require(isinstance(part, dict) and part.get("type") in ("input_text", "input_image"),
                 f"{at} must be an object with type 'input_text' or 'input_image'")
        if part["type"] == "input_text":
            parts.append(_text(part.get("text"), f"{at}.text"))
        else:
            src = _text(part.get("image_url"), f"{at}.image_url")
            # file paths only for trusted training data (image_base set); requests inline images
            _require(src.startswith("data:image/") or image_base is not None,
                     f"{at}.image_url must be a data:image/...;base64 URL")
            parts.append(ImageRef(src, image_base or ""))
    return tuple(parts)


def _parse_state(value: object, image_base: str | None) -> State:
    if isinstance(value, str):
        return value
    return parse_parts(value, "state", image_base)


_PARSERS = {"choice": _parse_choice, "score": _parse_score, "noul": _parse_noul}


def parse_request(body: object, image_base: str | None = None) -> Request:
    """`image_base`: directory that relative image paths resolve against (training data
    only); without it, images must be inline `data:` URIs."""
    _require(isinstance(body, dict), "request body must be a JSON object")
    state = _parse_state(body.get("state"), image_base)
    questions = body.get("questions")
    _require(isinstance(questions, dict) and len(questions) > 0,
             "questions must be a non-empty object")
    depth = body.get("depth", "adaptive")
    _require(depth in DEPTHS, f"depth must be one of {list(DEPTHS)}")

    parsed = []
    for name, q in questions.items():
        _require(isinstance(name, str) and name != "", "question names must be non-empty strings")
        _require(isinstance(q, dict), f"questions.{name} must be an object")
        kind = q.get("type")
        _require(kind in _PARSERS, f"questions.{name}.type must be one of {list(_PARSERS)}")
        instructions = _text(q.get("instructions"), f"questions.{name}.instructions")
        options, descs = _PARSERS[kind](name, q.get("criteria"))
        parsed.append(Question(name, instructions, kind, options, descs))
    return Request(state=state, questions=tuple(parsed), depth=depth)


def confidence(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    if len(probs) < 2:
        return 1.0
    return float(1.0 - (-(p * np.log(p)).sum()) / np.log(len(probs)))


def render_answer(q: Question, probs: Sequence[float]) -> dict:
    p = np.asarray(probs, dtype=np.float64)
    opts = q.resolved_options()
    if len(p) != len(opts) or not np.all(p >= 0) or abs(p.sum() - 1) > 1e-3:
        raise ValueError(f"engine returned invalid probabilities for {q.id!r}: {p}")
    p = p / p.sum()
    if q.kind == "noul":
        return {"type": "noul", "noul": float(p[1])}
    if q.kind == "score":
        return {
            "type": "score",
            "score": float((np.arange(len(p)) * p).sum()),
            "confidence": confidence(p),
            "legend": {str(i): lbl for i, lbl in enumerate(opts)},
            "probabilities": {str(i): float(v) for i, v in enumerate(p)},
        }
    return {
        "type": "choice",
        "choice": opts[int(np.argmax(p))],
        "confidence": confidence(p),
        "probabilities": {o: float(v) for o, v in zip(opts, p)},
    }


def handle(body: object, engine: Engine) -> dict:
    """Validate a request, run the engine and render the response (raises RequestError)."""
    req = parse_request(body)
    probs = engine(req)
    return {"answers": {q.id: render_answer(q, probs[q.id]) for q in req.questions}}
