"""Internal typed decision request.

The `/v1/systemone` wire format is parsed into these types by `trutina.api`; models and
prompt encoding only ever see them. Every question becomes an ordered list of options,
and an engine returns one probability per option:

- `choice`: options are the criteria keys, in request order.
- `score`:  options are the rubric levels, lowest to highest (index = level number).
- `noul`:   the two options ("no", "yes"); the answer is P(yes).

The state is plain text or an ordered sequence of text and image parts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from trutina.vision import ImageRef

NOUL_OPTIONS = ("no", "yes")
Kind = Literal["choice", "score", "noul"]
Depth = Literal["shallow", "adaptive", "deep"]
State = str | tuple[str | ImageRef, ...]


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    kind: Kind = "choice"
    options: tuple[str, ...] = ()
    descriptions: tuple[str, ...] = ()

    def resolved_options(self) -> tuple[str, ...]:
        if self.kind == "noul":
            return NOUL_OPTIONS
        if len(self.options) < 2:
            raise ValueError(f"{self.kind} question {self.id!r} needs at least 2 options")
        return self.options

    def resolved_descriptions(self) -> tuple[str, ...]:
        n = len(self.resolved_options())
        return self.descriptions if len(self.descriptions) == n else ("",) * n


@dataclass(frozen=True)
class Request:
    state: State
    questions: tuple[Question, ...]
    depth: Depth = "adaptive"


def state_parts(state: State) -> tuple[str | ImageRef, ...]:
    return (state,) if isinstance(state, str) else state


def state_text(state: State) -> str:
    """Text-only view (images become an `[image]` marker), for text-only engines."""
    return "".join(p if isinstance(p, str) else "[image]" for p in state_parts(state))
