import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from mlx_lm.models import qwen3_5 as mlx_qwen
from test_engines import ByteTok
from test_pt_parity import tiny_config
from test_train import toy_rows

from trutina.data import load_examples
from trutina.decider import MultiExitDecider as MxDecider
from trutina.pt.model import JointHead
from trutina.pt.train import build_model, train
from trutina.runs import RunDir, TrainConfig, encode_all
from trutina.train import add_lora as mx_add_lora
from trutina.train import load_trainable

CFG = TrainConfig(steps=40, exits=[4, 6], head_dim=32, rank=4, lr=3e-3, warmup=5, batch_size=8,
                  eval_every=20, ckpt_every=10, max_tokens=None)


def run(out, items, stop_at=None):
    torch.manual_seed(0)
    torso = transformers.Qwen3_5TextModel(tiny_config()).float()
    for p in torso.parameters():
        p.requires_grad_(False)
    torch.manual_seed(1)
    model = build_model(torso, CFG, "cpu")
    model.head = JointHead(64, 32, 1, 4, n_exits=2)  # small head for speed
    done = train(model, items, items[:16], CFG, RunDir(out, {}, CFG), "cpu", stop_at=stop_at)
    return model, done


def test_resumed_run_equals_uninterrupted_and_exports_to_mlx(tmp_path):
    path = tmp_path / "toy.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in toy_rows(48)))
    items, dropped = encode_all(load_examples(path), ByteTok(), None)
    assert dropped == 0

    straight, done = run(tmp_path / "a", items)
    assert done
    _, done = run(tmp_path / "b", items, stop_at=15)  # preempted mid-run
    assert not done
    resumed, done = run(tmp_path / "b", items)
    assert done and "resumed from step 15" in (tmp_path / "b" / "train.log").read_text()
    a, b = straight.trainable_state(), resumed.trainable_state()
    assert all(torch.allclose(a[k], b[k], atol=1e-6) for k in a)

    rows = [json.loads(line) for line in (tmp_path / "a" / "metrics.jsonl").read_text().splitlines()]
    evals = [r for r in rows if r["type"] == "eval"]
    assert evals[-1]["step"] == 40
    assert evals[-1]["exits"]["exit6"]["loss"] < evals[0]["exits"]["exit6"]["loss"]

    mlm = mlx_qwen.Model(mlx_qwen.ModelArgs.from_dict({"model_type": "qwen3_5",
                                                        "text_config": tiny_config().to_dict()}))
    mxd = MxDecider(mlm.model, hidden=64, exits=(4, 6), head_dim=32, head_layers=1, head_heads=4)
    mx_add_lora(mxd, rank=4)
    load_trainable(mxd, tmp_path / "a" / "trainable.safetensors")
    assert np.allclose(np.array(mxd.mixes[1].weights), a["mixes.1.weights"].numpy())


def test_v1_options_train_end_to_end(tmp_path):
    """Every v1 option on at once: augmentation, label sources, state read-back, ordinal
    scores, exit self-distillation; trains, evaluates and exports an MLX-loadable head."""
    import dataclasses

    from trutina.runs import augment, source_vocab

    rows = list(toy_rows(48))
    for i, r in enumerate(rows):
        r["meta"] = {"dataset": "toy", "source": f"d{i % 3}", "teacher": ["t1", "t2"][i % 2]}
        r["questions"]["lvl"] = {"type": "score", "instructions": "how much?", "criteria": ["lo", "mid", "hi"]}
        r["labels"]["lvl"] = [0.2, 0.5, 0.3]
    path = tmp_path / "toy.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    cfg = dataclasses.replace(CFG, steps=12, eval_every=6, ckpt_every=6, permute_choices=True,
                              drop_descriptions=0.3, read_state=True, ordinal_score=True,
                              source_tokens=4, self_distill=0.5, group_alpha=0.5)
    examples = load_examples(path)
    sources = source_vocab(examples, cfg.source_tokens)
    assert sources == {"t1": 1, "t2": 2}
    items, _ = encode_all(augment(examples, cfg), ByteTok(), None, sources)
    assert {i.source for i in items} == {0, 1, 2}
    val, _ = encode_all(examples[:12], ByteTok(), None)
    assert {i.source for i in val} == {0}

    torch.manual_seed(0)
    torso = transformers.Qwen3_5TextModel(tiny_config()).float()
    for p in torso.parameters():
        p.requires_grad_(False)
    model = build_model(torso, cfg, "cpu")
    assert train(model, items, val, cfg, RunDir(tmp_path / "v1", {}, cfg), "cpu")

    mlm = mlx_qwen.Model(mlx_qwen.ModelArgs.from_dict({"model_type": "qwen3_5",
                                                        "text_config": tiny_config().to_dict()}))
    mxd = MxDecider(mlm.model, hidden=64, exits=cfg.exits, head_dim=cfg.head_dim, **cfg.head_kwargs())
    mx_add_lora(mxd, rank=cfg.rank)
    load_trainable(mxd, tmp_path / "v1" / "trainable.safetensors")  # strict: names line up
