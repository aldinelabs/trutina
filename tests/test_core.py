import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx import nn

from trutina.metrics import accuracy, ece, fit_temperature, softmax
from trutina.model import Batch, Decider, loss_fn
from trutina.prompt import encode_question
from trutina.schema import Question


class ByteTok:
    def encode(self, text: str) -> list[int]:
        return list(text.encode())


class ToyTorso(nn.Module):
    """Stand-in for a pretrained decoder: embeddings + one causal attention layer."""

    def __init__(self, vocab: int = 256, hidden: int = 32):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.attn = nn.MultiHeadAttention(hidden, 4)

    def __call__(self, tokens: mx.array) -> mx.array:
        x = self.emb(tokens)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(tokens.shape[1])
        return x + self.attn(x, x, x, mask=mask)


def test_encode_positions_point_at_option_ends():
    tok = ByteTok()
    q = Question("q1", "Is it raining?", kind="noul")
    e = encode_question(tok, "sky is grey", q)
    assert [chr(e.tokens[p]) for p in e.option_pos] == ["\n", "\n"]
    assert chr(e.tokens[e.answer_pos]) == ">"
    assert e.answer_pos == len(e.tokens) - 1


def test_padded_options_get_zero_probability():
    tok = ByteTok()
    items = [
        encode_question(tok, "s", Question("a", "pick", options=("x", "y", "z"))),
        encode_question(tok, "s", Question("b", "pick", kind="noul")),
    ]
    model = Decider(ToyTorso(), hidden_size=32, head_dim=16)
    probs = mx.softmax(model(Batch(items)), axis=-1)
    assert probs.shape == (2, 3)
    assert float(probs[1, 2]) < 1e-6
    assert np.allclose(np.array(probs.sum(-1)), 1.0, atol=1e-5)


def test_learns_toy_task():
    # Answer is whichever option text appears in the state. Runs on CPU: GPU reductions
    # are non-deterministic, which made the accuracy threshold flaky.
    with mx.stream(mx.cpu):
        _learn_toy_task()


def _learn_toy_task():
    tok = ByteTok()
    rng = np.random.default_rng(0)
    words = ["apple", "brick", "cloud", "delta"]
    data = []
    for _ in range(64):
        opts = tuple(rng.permutation(words)[:3])
        y = int(rng.integers(3))
        data.append((encode_question(tok, f"I saw {opts[y]}", Question("q", "what?", options=opts)), y))
    batch, targets = Batch([d for d, _ in data]), mx.array([y for _, y in data])

    mx.random.seed(0)
    model = Decider(ToyTorso(), hidden_size=32, head_dim=16)
    opt = optim.Adam(learning_rate=3e-3)
    step = nn.value_and_grad(model, loss_fn)
    for _ in range(600):
        _, grads = step(model, batch, targets)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
    acc = float((mx.argmax(model(batch), axis=-1) == targets).mean())
    assert acc >= 0.75  # chance is 1/3


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(1)
    labels = list(rng.integers(0, 2, 500))
    # 70%-accurate predictor with logits that are far too sharp.
    logits = [np.array([0.0, 8.0]) if (y == 1) == (rng.random() < 0.7) else np.array([8.0, 0.0])
              for y in labels]
    t = fit_temperature(logits, labels)
    before = ece([softmax(l) for l in logits], labels)
    after = ece([softmax(l, t) for l in logits], labels)
    assert t > 1.0 and after < before
    assert accuracy([softmax(l, t) for l in logits], labels) == accuracy(
        [softmax(l) for l in logits], labels)
