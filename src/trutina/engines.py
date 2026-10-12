"""Decision engines: Request -> {question id: probabilities in option order}.

`LMScoringEngine` is the untrained zero-shot baseline. It renders each question as text,
scores every option label by its mean token log-prob under the causal LM and softmaxes
across options. The question prefix is encoded once; options run as forked branches.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx import nn

from trutina.backbone import fork_cache, forward_hidden, make_cache
from trutina.prompt import Tokenizer, option_line
from trutina.schema import Question, Request, State, state_text

NOUL_LABELS = ("No", "Yes")


def render_text(state: State, q: Question) -> str:
    state = state_text(state)
    lines = "".join(
        option_line(i, o, d) for i, (o, d) in enumerate(zip(q.resolved_options(), q.resolved_descriptions()))
    )
    kind = {"choice": "Pick one option.", "score": "Pick one level (ordered low to high).",
            "noul": "Answer Yes or No."}[q.kind]
    return f"{state}\n\nQuestion: {q.text}\n{kind}\n{lines if q.kind != 'noul' else ''}Answer:"


class LMScoringEngine:
    def __init__(self, torso: nn.Module, tok: Tokenizer):
        self.torso, self.tok = torso, tok

    def __call__(self, req: Request) -> dict[str, np.ndarray]:
        return {q.id: self.score_question(req.state, q) for q in req.questions}

    def score_question(self, state: State, q: Question) -> np.ndarray:
        labels = NOUL_LABELS if q.kind == "noul" else q.resolved_options()
        prefix = self.tok.encode(render_text(state, q))
        branches = [self.tok.encode(" " + lbl) or [0] for lbl in labels]

        cache = make_cache(self.torso)
        h_prefix, _ = forward_hidden(self.torso, mx.array([prefix]), cache=cache)
        L = max(map(len, branches))
        padded = mx.array([b + [b[-1]] * (L - len(b)) for b in branches])
        h_branch, _ = forward_hidden(self.torso, padded, cache=fork_cache(cache, len(branches)))

        # Token j of a label is predicted by the hidden state just before it: the prefix's
        # last position for j = 0, the branch's own position j - 1 otherwise.
        lm = self.torso.embed_tokens.as_linear
        # fp32 head: bf16 logits are too coarse to separate close options.
        first = nn.log_softmax(lm(h_prefix[0, -1].astype(mx.float32)), axis=-1)
        scores = []
        for i, b in enumerate(branches):
            lp = [first[b[0]]]
            if len(b) > 1:
                rest = nn.log_softmax(lm(h_branch[i, : len(b) - 1].astype(mx.float32)), axis=-1)
                lp.append(rest[mx.arange(len(b) - 1), mx.array(b[1:])].sum())
            scores.append(float(sum(lp)) / len(b))
        s = np.array(scores)
        p = np.exp(s - s.max())
        return p / p.sum()
