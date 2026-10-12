"""Torch decider with images: features land on the image placeholders and positions follow
Qwen3.5's multimodal RoPE, exactly as in Hugging Face's own multimodal forward."""

import base64
import io

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from PIL import Image
from test_engines import ByteTok
from test_pt_parity import tiny_config

from trutina.prompt import encode_question
from trutina.pt.lora import add_lora
from trutina.pt.model import DecisionBatch, MultiExitDecider
from trutina.schema import Question
from trutina.vision import IMAGE_PAD, ImageRef


def tiny_vlm():
    text = tiny_config().to_dict()
    text["vocab_size"] = IMAGE_PAD + 8  # real special-token ids must be in the vocabulary
    cfg = transformers.Qwen3_5Config(
        text_config=text,
        vision_config={"depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
                       "out_hidden_size": 64, "num_position_embeddings": 64},
    )
    torch.manual_seed(0)
    model = transformers.Qwen3_5Model(cfg).float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _uri(w, h, seed):
    buf = io.BytesIO()
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_image_forward_matches_hf_multimodal_forward():
    vlm = tiny_vlm()
    model = MultiExitDecider(vlm.language_model, hidden=64, exits=(4, 6), mix_every=2, head_dim=32,
                             head_layers=1, head_heads=4, vision=vlm)
    add_lora(model.torso.layers, rank=2)
    q = Question("q", "Is it red?", kind="noul")
    items = [encode_question(ByteTok(), ("look: ", ImageRef(_uri(64, 96, 0)), " then ",
                                         ImageRef(_uri(300, 200, 1))), q),
             encode_question(ByteTok(), "plain text state", q)]
    b = DecisionBatch.build(items, ["noul", "noul"])
    assert b.pixel_values is not None and b.image_grid_thw.shape == (2, 3)

    def capture():
        hs = {}
        hooks = [model.torso.layers[l - 1].register_forward_hook(
            lambda _m, _a, out, l=l: hs.__setitem__(l, out[0] if isinstance(out, tuple) else out))
            for l in model.mix_layers]
        return hs, hooks

    with torch.no_grad():
        ours, hooks = capture()
        model(b)
        [h.remove() for h in hooks]
        ref, hooks = capture()
        vlm(input_ids=b.tokens, pixel_values=b.pixel_values, image_grid_thw=b.image_grid_thw,
            mm_token_type_ids=(b.tokens == IMAGE_PAD).int(), use_cache=False)
        [h.remove() for h in hooks]
    assert set(ours) == set(model.mix_layers)
    for l in model.mix_layers:
        torch.testing.assert_close(ours[l], ref[l], atol=1e-5, rtol=0)

    # the text-only row is unaffected by sharing a batch with images
    alone = DecisionBatch.build(items[1:], ["noul"])
    with torch.no_grad():
        mixed = model(b)
        solo = model(alone)
    for m, s in zip(mixed, solo):
        torch.testing.assert_close(m[1:], s, atol=1e-5, rtol=0)


def test_image_rows_train_the_adapters():
    vlm = tiny_vlm()
    model = MultiExitDecider(vlm.language_model, hidden=64, exits=(4, 6), mix_every=2, head_dim=32,
                             head_layers=1, head_heads=4, vision=vlm)
    add_lora(model.torso.layers, rank=2)
    q = Question("q", "Is it red?", kind="noul")
    b = DecisionBatch.build([encode_question(ByteTok(), ("x", ImageRef(_uri(64, 64, 2))), q)], ["noul"])
    sum(lg.sum() for lg in model(b)).backward()
    assert all(p.grad is not None for n, p in model.named_parameters() if "lora_a" in n)
    assert not any(p.requires_grad for p in vlm.visual.parameters())
