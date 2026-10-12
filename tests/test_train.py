import json
import random

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from test_backbone import tiny_qwen35
from test_decider import tiny_decider
from test_engines import ByteTok

from trutina.decider import MultiExitDecider
from trutina.runs import RunDir, TrainConfig, encode_all
from trutina.schema import Question
from trutina.train import (
    add_lora,
    load_examples,
    load_trainable,
    save_trainable,
    target_vector,
    train,
)


def test_target_vectors():
    c = Question("c", "?", "choice", ("a", "b", "c"))
    s = Question("s", "?", "score", ("lo", "mid", "hi"))
    n = Question("n", "?", "noul")
    assert target_vector(c, "b").tolist() == [0, 1, 0]
    assert target_vector(s, 2).tolist() == [0, 0, 1]
    assert target_vector(s, "mid").tolist() == [0, 1, 0]
    assert target_vector(n, True).tolist() == [0, 1]
    assert target_vector(n, 0.25).tolist() == [0.75, 0.25]
    assert target_vector(c, [0.2, 0.3, 0.5]).tolist() == pytest.approx([0.2, 0.3, 0.5])
    with pytest.raises(ValueError):
        target_vector(c, [0.5, 0.5])


def toy_rows(n: int, seed: int = 0):
    rng = random.Random(seed)
    words = ["apple", "brick", "cloud", "delta"]
    for _ in range(n):
        opts = rng.sample(words, 3)
        ans = rng.choice(opts)
        yield {"state": f"I saw {ans}", "labels": {"q": ans, "seen": True},
               "questions": {"q": {"type": "choice", "instructions": "what?",
                                   "criteria": {o: "" for o in opts}},
                             "seen": {"type": "noul", "instructions": "anything?"}}}


CFG = TrainConfig(steps=60, lr=3e-3, warmup=5, batch_size=8, rank=4, exits=[4, 6, 8],
                  head_dim=32, eval_every=30, ckpt_every=10, max_tokens=None)


def run_toy(out, items, stop_at=None):
    model = tiny_decider()
    add_lora(model, rank=CFG.rank)
    run = RunDir(out, {}, CFG)
    return model, train(model, items, items[:16], CFG, run, stop_at=stop_at)


def records(out, kind):
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    return [r for r in rows if r["type"] == kind]


def test_lora_training_learns_and_resumes_exactly(tmp_path):
    path = tmp_path / "toy.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in toy_rows(48)))
    examples = load_examples(path)
    assert len(examples) == 96  # two questions per row
    items, _ = encode_all(examples, ByteTok(), None)

    with mx.stream(mx.cpu):  # deterministic
        model, done = run_toy(tmp_path / "a", items)
        trainable = {k for k, _ in tree_flatten(model.trainable_parameters())}
        assert any("lora_a" in k for k in trainable)
        assert not any(k.startswith("torso.") and "lora" not in k for k in trainable)
        assert done
        _, done = run_toy(tmp_path / "b", items, stop_at=25)
        assert not done
        resumed, done = run_toy(tmp_path / "b", items)
        assert done

    train_losses = [r["loss"] for r in records(tmp_path / "a", "train")]
    assert train_losses[-1] < 0.6 * train_losses[0]
    evals = records(tmp_path / "a", "eval")
    assert [r["step"] for r in evals] == [30, 60]
    best = json.loads((tmp_path / "a" / "best.json").read_text())
    assert best["step"] == min(evals, key=lambda r: RunDir.val_score(r["exits"]))["step"]
    assert (tmp_path / "a" / "best.safetensors").exists()
    assert json.loads((tmp_path / "b" / "status.json").read_text())["state"] == "done"
    assert "resumed from step 25" in (tmp_path / "b" / "train.log").read_text()
    a = dict(tree_flatten(model.trainable_parameters()))
    b = dict(tree_flatten(resumed.trainable_parameters()))
    assert all(np.allclose(np.array(a[k]), np.array(b[k]), atol=1e-5) for k in a)


def short_decider():  # last exit (6) before the final layer (8)
    mx.random.seed(2)
    return MultiExitDecider(tiny_qwen35(), hidden=64, exits=(4, 6), mix_every=2,
                            head_dim=32, head_layers=1, head_heads=4)


def test_lora_covers_first_layers_up_to_last_exit(tmp_path):
    model = short_decider()
    add_lora(model, rank=2)
    n = model.exits[-1]
    has = [any("lora_a" in k for k, _ in tree_flatten(l.parameters())) for l in model.torso.layers]
    assert has == [True] * n + [False] * (len(has) - n)

    save_trainable(model, tmp_path / "w.safetensors")
    fresh = short_decider()
    add_lora(fresh, rank=2)
    load_trainable(fresh, tmp_path / "w.safetensors")
    a = dict(tree_flatten(model.trainable_parameters()))
    b = dict(tree_flatten(fresh.trainable_parameters()))
    assert all(mx.array_equal(a[k], b[k]) for k in a)


def test_token_budget_batches_keep_every_question_and_respect_budget():
    import random

    from trutina.prompt import Encoded
    from trutina.runs import Item, step_batches

    rng = random.Random(0)
    items = [Item(Encoded(list(range(rng.choice([20, 50, 400, 1500]))), 0, [0]), "choice",
                  np.ones(1, np.float32)) for _ in range(256)]
    cfg = TrainConfig(batch_size=8, grad_accum=2, batch_tokens=2048)
    stream = step_batches(items, cfg, random.Random(1))
    seen = []
    for _ in range(256 // 16):
        micro = next(stream)
        assert sum(len(m) for m in micro) == 16
        for m in micro:
            longest = max(len(i.enc.tokens) for i in m)
            assert len(m) == 1 or longest * len(m) <= 2048
        seen += [id(i) for m in micro for i in m]
    assert len(set(seen)) == 256  # one epoch covers every question once


def test_old_runs_keep_soft_smoothing_on_resume(tmp_path):
    import dataclasses
    import json as _json

    from trutina.runs import RunDir, TrainConfig

    old = {k: v for k, v in dataclasses.asdict(TrainConfig()).items()
           if k not in ("label_smoothing", "smooth_soft", "self_distill")}
    (tmp_path / "config.json").write_text(_json.dumps({"train": old}))
    cfg = TrainConfig()
    RunDir(tmp_path, {}, cfg)
    assert cfg.smooth_soft is True and cfg.label_smoothing == 0.05


def test_augment_permutes_choices_with_targets_and_drops_descriptions():
    import numpy as np

    from trutina.data import Example
    from trutina.runs import TrainConfig, augment
    from trutina.schema import Question

    q = Question("q", "pick", "choice", ("a", "b", "c", "d"), ("da", "db", "dc", "dd"))
    s = Question("s", "rate", "score", ("lo", "mid", "hi"), ("x", "y", "z"))
    exs = [Example("st", q, np.array([0, 0, 1.0, 0], dtype=np.float32)),
           Example("st", s, np.array([0, 1.0, 0], dtype=np.float32))] * 50
    out = augment(exs, TrainConfig(permute_choices=True, drop_descriptions=0.3))
    orders = set()
    for e in out:
        if e.question.kind == "choice":
            orders.add(e.question.options)
            assert e.question.options[int(np.argmax(e.target))] == "c"   # target follows its option
            if e.question.descriptions:
                assert all(d == "d" + o for o, d in zip(e.question.options, e.question.descriptions))
        else:
            assert e.question.options == ("lo", "mid", "hi")              # levels keep their order
    assert len(orders) > 5
    dropped = sum(not e.question.descriptions for e in out)
    assert 10 < dropped < 50
    assert augment(exs, TrainConfig()) == list(exs)


def test_rebalance_flattens_groups():
    import collections
    import random as _random

    import numpy as np

    from trutina.data import Example
    from trutina.runs import rebalance
    from trutina.schema import Question

    q = Question("q", "ok?", "noul")
    def ex(g):
        return Example("s", q, np.array([0.5, 0.5]), group=g)

    exs = [ex("a/big")] * 900 + [ex("a/small")] * 100 + [ex("b/x")] * 50
    out = rebalance(exs, 0.5, _random.Random(0))
    c = collections.Counter(e.group for e in out)
    assert c["a/small"] == 250 and c["a/big"] == 750   # 100**.5 / (900**.5 + 100**.5) = 1/4
    assert c["b/x"] == 50                               # other datasets keep their size
