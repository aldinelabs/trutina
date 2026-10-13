"""The CPU Gated DeltaNet patch must match transformers' reference fallback."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers.models.qwen3_5 import modeling_qwen3_5 as m

from trutina import cpu_gdn


@pytest.fixture
def restore(monkeypatch):
    for name in ("torch_chunk_gated_delta_rule", "causal_conv1d_fn"):
        monkeypatch.setattr(m, name, getattr(m, name))
    monkeypatch.setattr(m, "_trutina_cpu_gdn", False, raising=False)


def test_chunk_size_follows_prompt_length():
    assert [cpu_gdn.chunk_length(n) for n in (1, 3, 70, 128, 129, 300)] == [8, 8, 72, 128, 72, 104]


def test_causal_conv_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(2, 37, 24).transpose(1, 2)  # non-contiguous, as in the model
    weight, bias = torch.randn(24, 4), torch.randn(24)
    for b in (None, bias):
        expected = m.causal_conv1d_fn(x, weight, b, activation="silu")
        torch.testing.assert_close(cpu_gdn.causal_conv1d(x, weight, b, activation="silu"), expected)


@pytest.mark.parametrize("length,state", [(1, False), (70, False), (130, True)])
def test_delta_rule_matches_reference(length, state):
    torch.manual_seed(length)
    q, k, v = (torch.randn(2, length, 4, 16) for _ in range(3))
    g, beta = -torch.rand(2, length, 4), torch.rand(2, length, 4)
    initial = torch.randn(2, 4, 16, 16) if state else None
    expected = m.torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=initial,
                                              output_final_state=True, use_qk_l2norm_in_kernel=True)
    got = cpu_gdn.chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=initial,
                                         output_final_state=True, use_qk_l2norm_in_kernel=True)
    for a, b in zip(got, expected):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


def test_patched_model_matches_reference_and_uses_patch(restore):
    cfg = transformers.Qwen3_5TextConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, vocab_size=100,
        linear_num_value_heads=2, linear_num_key_heads=1,
        linear_key_head_dim=16, linear_value_head_dim=16, full_attention_interval=2)
    torch.manual_seed(7)
    torso = transformers.Qwen3_5TextModel(cfg).float().eval()
    ids = torch.randint(0, 100, (2, 150))
    mask = torch.ones_like(ids)
    mask[1, 120:] = 0
    with torch.inference_mode():
        expected = torso(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    cpu_gdn.enable()
    cpu_gdn.enable()
    calls = []
    patched = m.torch_chunk_gated_delta_rule
    m.torch_chunk_gated_delta_rule = lambda *a, **k: (calls.append(1), patched(*a, **k))[1]
    with torch.inference_mode():
        got = torso(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    assert len(calls) == 2
    torch.testing.assert_close(got[0], expected[0], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(got[1, :120], expected[1, :120], rtol=1e-4, atol=1e-5)
