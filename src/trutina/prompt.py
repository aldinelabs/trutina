"""Turn a (state, question) pair into token ids plus the positions the pointer head reads.

Layout (one question per sequence):

    <state>\\n{state}\\n</state>\\n<question type="{kind}">{text}</question>\\n<options>\\n
    [0] {opt0}: {desc0}\\n      <- index of the last token of each option line is recorded
    [1] {opt1}\\n               (": desc" only when the request gave a description)
    ...
    </options>\\n<answer>      <- index of the last token is the query position

Segments are encoded separately and concatenated so the recorded indices are exact.
Images in the state become `<|vision_start|>` + image pads + `<|vision_end|>` in place
(`trutina.vision`); `Encoded.images` / `grids` list them in order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from trutina.schema import Question, State, state_parts
from trutina.vision import ImageRef, image_grid, image_tokens


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class Encoded:
    tokens: list[int]
    answer_pos: int
    option_pos: list[int]
    images: tuple[ImageRef, ...] = ()
    grids: tuple[tuple[int, int, int], ...] = ()


def option_line(i: int, label: str, description: str) -> str:
    return f"[{i}] {label}: {description}\n" if description else f"[{i}] {label}\n"


def encode_question(tok: Tokenizer, state: State, q: Question) -> Encoded:
    """`tok.encode` must not add special tokens (BOS would be repeated per segment)."""
    rest = f'\n</state>\n<question type="{q.kind}">{q.text}</question>\n<options>\n'
    images, grids = [], []
    if isinstance(state, str):  # one segment: identical tokenisation to text-only models
        tokens = tok.encode(f"<state>\n{state}{rest}")
    else:
        tokens = tok.encode("<state>\n")
        for part in state_parts(state):
            if isinstance(part, str):
                tokens += tok.encode(part)
            else:
                with part.open() as img:
                    grid = image_grid(img.size)
                tokens += image_tokens(grid)
                images.append(part)
                grids.append(grid)
        tokens += tok.encode(rest)
    option_pos: list[int] = []
    for i, (opt, desc) in enumerate(zip(q.resolved_options(), q.resolved_descriptions())):
        tokens += tok.encode(option_line(i, opt, desc))
        option_pos.append(len(tokens) - 1)
    tokens += tok.encode("</options>\n<answer>")
    return Encoded(tokens=tokens, answer_pos=len(tokens) - 1, option_pos=option_pos,
                   images=tuple(images), grids=tuple(grids))
