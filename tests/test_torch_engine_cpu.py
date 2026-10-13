"""Compare segmented release inference with Hugging Face's full forward on CPU."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from trutina.cpu_quant import quantize_torso
from trutina.torch_engine import TorchTrutina


def make_engine():
    cfg = transformers.Qwen3_5TextConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, vocab_size=100,
        linear_num_value_heads=2, linear_num_key_heads=1,
        linear_key_head_dim=16, linear_value_head_dim=16, full_attention_interval=2)
    torch.manual_seed(7)
    torso = transformers.Qwen3_5TextModel(cfg).float().eval()
    from test_rkllm import Tokenizer

    readout = {"lens_logscale": np.zeros(1), "prior_logscale": np.array(0.),
               "lens.0.weight": np.zeros((32, 32), np.float32),
               "lens.0.bias": np.zeros(32, np.float32)}
    return TorchTrutina(torso, Tokenizer(),
                        {"exits": [2, 4], "temperature": {"2": 1., "4": 1.},
                         "adaptive_threshold": 0.86}, readout, "cpu", 1024)


def test_deep_matches_hf_full_forward_with_padding():
    model = make_engine()
    model.depth = "deep"
    seqs = [[4, 8, 9, 2], [3, 5]]
    with torch.inference_mode():
        got = model._read(seqs, [3, 2])
        ids = torch.tensor([[4, 8, 9, 2], [3, 5, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
        hs = model.torso(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        for i, k in enumerate([3, 2]):
            logits = hs[i, len(seqs[i])-1].float() @ model.rows[:k].T
            np.testing.assert_allclose(got[i], torch.softmax(logits, -1).numpy(), atol=1e-6)


def test_adaptive_can_continue_or_stop_without_recomputing():
    model = make_engine()
    counts = [0]*4
    hooks = []
    for i, layer in enumerate(model.torso.layers):
        def count(_m, _a, _out, i=i):
            counts[i] += 1
        hooks.append(layer.register_forward_hook(count))
    with torch.inference_mode():
        model.threshold = 1.0
        model._read([[4, 8, 9]], [2])
        assert model.last_exit == 4 and counts == [1, 1, 1, 1]
        model.threshold = 0.0
        model._read([[4, 8, 9]], [2])
        assert model.last_exit == 2 and counts == [2, 2, 1, 1]
    for h in hooks:
        h.remove()


def test_cpu_int8_runs_all_depths_and_preserves_decisions():
    model = make_engine()
    model.depth = "deep"
    with torch.inference_mode():
        ref = model._read([[4, 8, 9]], [3])[0]
        quantize_torso(model.torso)
        for depth in ["shallow", "adaptive", "deep"]:
            model.depth = depth
            got = model._read([[4, 8, 9]], [3])[0]
            assert np.isfinite(got).all() and got.sum() == pytest.approx(1)
        np.testing.assert_allclose(got, ref, atol=0.02)


def test_close_releases_the_torso():
    import weakref

    model = make_engine()
    ref = weakref.ref(model.torso)
    model.close()
    assert ref() is None
    model.close()


def test_image_engine_matches_hf_full_multimodal_forward(tmp_path):
    from PIL import Image

    from trutina.schema import Question
    from trutina.vision import IMAGE_PAD, ImageRef

    model = make_engine()
    text = model.torso.config.to_dict()
    text["vocab_size"] = IMAGE_PAD+8
    cfg = transformers.Qwen3_5Config(text_config=text, vision_config={
        "depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
        "out_hidden_size": 32, "num_position_embeddings": 64})
    vlm = transformers.Qwen3_5Model(cfg).float().eval()
    model.torso, model.vision = vlm.language_model, vlm
    model.rows = model.torso.embed_tokens.weight[65:91].float()
    model.depth = "deep"
    path = tmp_path / "red.png"
    Image.new("RGB", (256, 256), "red").save(path)
    # Keep the test sequence tiny while still using the real vision preprocessing.
    model.tok.encode = lambda text, **kw: [4, IMAGE_PAD, 9]
    q = Question("q", "color?", options=("red", "blue", "green"))
    captured = {}
    original_prepare = model._prepare
    def prepare(ids, mask, embeds, positions):
        captured.update(ids=ids, embeds=embeds, positions=positions)
        return original_prepare(ids, mask, embeds, positions)
    model._prepare = prepare
    with torch.inference_mode():
        got = model._read_image((ImageRef(str(path)),), q)
        hs = model.torso(inputs_embeds=captured["embeds"],
                         position_ids=captured["positions"], use_cache=False).last_hidden_state
        ref = torch.softmax(hs[0, -1] @ model.rows[:3].T, -1).numpy()
    np.testing.assert_allclose(got, ref, atol=1e-6)
