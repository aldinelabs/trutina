"""Prompt format of the letter readout, shared by the MLX and torch engines and by training.

Every option is shown at once, labelled (A)/(B)/... (or [1]/[2]/... in the numbered variant); the
answer is read from the next-token logits of the labels after "Best answer: (". The released model
uses the chat template with thinking off, the "sandwich" order (schema, input, schema again) and
letters. No framework imports.
"""

from __future__ import annotations

import itertools
import math
import re
import string

from trutina.schema import Question, State

FORMAT_VERSION = 1  # bump when the rendered prompt changes; trutina.json records it
ORDERS = ("state_first", "schema_first", "sandwich")
LABELS = ("letter", "number")
INTRO = "Read the input, then answer the question by picking exactly one option."
IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"  # one pad per image; the encoder expands it
# Wide questions (more than 26 options, up to WIDE_MAX): the labels are A..Z, then the two-letter strings
# that are single tokens (AA, AB, ...), as in decider and Sieve. In running text "(AA)" would tokenize as
# "(A" "A", so the prompt marks each label with a placeholder and `encode_ids` inserts the label token.
WIDE_MAX = 255
WIDE_MARK = re.compile("\ue000(\\d+)\ue001")
_WIDE: dict[int, tuple[list[str], list[int]]] = {}


def state_text(state: State, images: bool = False) -> str:
    """Text parts joined by newlines; an image becomes `IMAGE` (with `images`) or is refused."""
    if isinstance(state, str):
        return state
    if not images and any(not isinstance(p, str) for p in state):
        raise ValueError("image state")
    return "\n".join(p if isinstance(p, str) else IMAGE for p in state)


def tag(labels: str, i: int) -> tuple[str, str]:
    """(how the option is shown, the token read after the answer prefix)."""
    return (f"({chr(65 + i)})", chr(65 + i)) if labels == "letter" else (f"[{i + 1}]", str(i + 1))


def max_options(labels: str) -> int:
    return WIDE_MAX if labels == "letter" else 9


def wide_labels(tok) -> tuple[list[str], list[int]]:
    """(label strings, token ids), WIDE_MAX entries; the first 26 are A..Z (the narrow letters)."""
    key = id(tok)
    if key not in _WIDE:
        names, ids = [], []
        cands = list(string.ascii_uppercase) + ["".join(p) for p in itertools.product(string.ascii_uppercase, repeat=2)]
        for n in cands:
            t = tok.encode(n, add_special_tokens=False)
            if len(t) == 1:
                names.append(n)
                ids.append(t[0])
            if len(ids) == WIDE_MAX:
                break
        assert names[:26] == list(string.ascii_uppercase) and len(ids) == WIDE_MAX
        _WIDE[key] = (names, ids)
    return _WIDE[key]


def is_wide(q: Question, labels: str) -> bool:
    return labels == "letter" and len(q.resolved_options()) > 26


def schema(q: Question, labels: str) -> str:
    lines = [f"Question: {q.text}", "Options:"]
    for i, (o, d) in enumerate(zip(q.resolved_options(), q.resolved_descriptions())):
        shown = f"(\ue000{i}\ue001)" if is_wide(q, labels) else tag(labels, i)[0]
        lines.append(f"{shown} {o}" + (f": {d}" if d else ""))
    return "\n".join(lines)


def encode_ids(tok, text: str) -> tuple[list[int], list[int]]:
    """Token ids of a prompt whose wide labels are placeholders: the text between them is tokenized piecewise
    and each placeholder becomes its single label token. Returns (ids, positions of the label tokens)."""
    _, lab = wide_labels(tok)
    ids, pos, at = [], [], 0
    for m in WIDE_MARK.finditer(text):
        ids += tok.encode(text[at : m.start()], add_special_tokens=False)
        pos.append(len(ids))
        ids.append(lab[int(m.group(1))])
        at = m.end()
    ids += tok.encode(text[at:], add_special_tokens=False)
    return ids, pos


def prompt(state: State, q: Question, order: str, labels: str, extra: str = "", images: bool = False) -> str:
    """`extra` (e.g. re-read passages) goes after the input, before the closing schema."""
    s, inp = schema(q, labels), f"Input:\n{state_text(state, images)}"
    body = {"state_first": [inp, s], "schema_first": [s, inp], "sandwich": [s, inp, s]}[order]
    if extra:
        body.insert(body.index(inp) + 1, extra)
    return "\n\n".join([INTRO, *body])


def answer_prefix(labels: str) -> str:
    return "Best answer: " + ("(" if labels == "letter" else "[")


def full_text(tok, user: str, chat: bool, labels: str) -> str:
    """Plain text for a base model; chat template (thinking off) opening the assistant turn otherwise."""
    if not chat:
        return f"{user}\n\n{answer_prefix(labels)}"
    head = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    return head + answer_prefix(labels)


def decisive(p_yes: float, t: float, b: float) -> float:
    """The decisive yes/no map: logit shifted by b and divided by t (fitted after the final temperature)."""
    z = math.log(max(p_yes, 1e-9)) - math.log(max(1 - p_yes, 1e-9))
    return 1 / (1 + math.exp(-(z - b) / t))
