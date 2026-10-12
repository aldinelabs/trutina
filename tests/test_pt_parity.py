"""The torch training port must compute exactly what the MLX model computes.

A tiny random Qwen3.5 is built in Hugging Face/torch, loaded into mlx-lm through the
real checkpoint sanitiser, and both deciders (with identical adapters, mixes and head,
transferred via the export file) must give the same logits and losses.
"""

import mlx.core as mx
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from mlx_lm.models import qwen3_5 as mlx_qwen
from safetensors.torch import save_file
from test_engines import ByteTok

from trutina import losses as mlx_losses
from trutina.decider import DecisionBatch as MxBatch
from trutina.decider import MultiExitDecider as MxDecider
from trutina.prompt import encode_question
from trutina.pt import losses as pt_losses
from trutina.pt.lora import add_lora as pt_add_lora
from trutina.pt.model import DecisionBatch as PtBatch
from trutina.pt.model import MultiExitDecider as PtDecider
from trutina.schema import Question
from trutina.train import add_lora as mx_add_lora
from trutina.train import load_trainable

EXITS, LAYERS = (4, 6), 8  # last exit before the final layer: exercises truncation


def tiny_config():
    return transformers.Qwen3_5TextConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=LAYERS, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, vocab_size=500, linear_num_value_heads=4,
        linear_num_key_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, attn_output_gate=True, tie_word_embeddings=True,
        full_attention_interval=4,
    )


def build_pair(tmp_path, **head_kw):
    torch.manual_seed(0)
    cfg = tiny_config()
    hf = transformers.Qwen3_5TextModel(cfg).float().eval()
    with torch.no_grad():  # make zero-initialised norms non-trivial
        for name, p in hf.named_parameters():
            if "norm" in name:
                p.normal_(0, 0.3)

    weights = {"model.language_model." + k: mx.array(v.numpy()) for k, v in hf.state_dict().items()}
    mlm = mlx_qwen.Model(mlx_qwen.ModelArgs.from_dict({"model_type": "qwen3_5",
                                                        "text_config": cfg.to_dict()}))
    mlm.load_weights(list(mlm.sanitize(weights).items()))

    for p in hf.parameters():
        p.requires_grad_(False)
    pt = PtDecider(hf, hidden=64, exits=EXITS, mix_every=2, head_dim=32, head_layers=1, head_heads=4,
                   **head_kw)
    pt_add_lora(pt.torso.layers, rank=2)
    with torch.no_grad():  # non-zero adapters, distinct mixes and a decisive head
        for name, p in pt.named_parameters():
            if p.requires_grad and ("lora_b" in name or "mixes" in name or "source" in name
                                    or "ordinal" in name):
                p.normal_(0, 0.1)
        pt.head.q.weight.mul_(20)
    path = tmp_path / "trainable.safetensors"
    save_file(pt.trainable_state(), str(path))

    mxd = MxDecider(mlm.model, hidden=64, exits=EXITS, mix_every=2, head_dim=32, head_layers=1,
                    head_heads=4, **head_kw)
    mx_add_lora(mxd, rank=2)
    load_trainable(mxd, path)
    return pt.eval(), mxd


@pytest.mark.parametrize("head_kw", [{}, {"read_state": True, "n_sources": 3, "ordinal_score": True}])
def test_torch_port_matches_mlx(tmp_path, head_kw):
    pt, mxd = build_pair(tmp_path, **head_kw)
    sources = [2, 0, 1] if head_kw else None
    tok = ByteTok()
    qs = [Question("a", "pick one", "choice", ("red", "green", "blue")),
          Question("b", "is it ok?", "noul"),
          Question("c", "how good", "score", ("bad", "meh", "fine", "great"))]
    items = [encode_question(tok, f"state number {i} " * (i + 1), q) for i, q in enumerate(qs)]
    kinds = [q.kind for q in qs]
    targets = np.zeros((3, 4), np.float32)
    targets[0, 1] = targets[1, 0] = targets[2, 3] = 1.0

    with torch.no_grad():
        pt_logits = pt(PtBatch.build(items, kinds, sources=sources))
    mb = MxBatch.build(items, kinds, sources=sources)
    # MLX's fp32 GPU matmul runs at reduced precision (~1e-3 relative on M5): compare on CPU.
    with mx.stream(mx.cpu):
        mx_logits = mxd(mb)
        mx.eval(mx_logits)

    for a, b in zip(pt_logits, mx_logits):
        np.testing.assert_allclose(a.numpy(), np.array(b), atol=3e-5, rtol=0)
        pt_loss = pt_losses.decision_loss(a, torch.tensor(targets), torch.tensor(np.array(mb.option_mask)),
                                          torch.tensor(np.array(mb.is_score)))
        mx_loss = mlx_losses.decision_loss(b, mx.array(targets), mb.option_mask, mb.is_score)
        np.testing.assert_allclose(pt_loss.numpy(), np.array(mx_loss), atol=1e-5, rtol=1e-4)
