"""Check quantization and real adaptive execution when an RK3588 bridge is available."""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from trutina.npu_matmul import MatmulRuntime, offload_torso, quantize_weight


def test_weight_quantization_is_per_output_channel_and_handles_zero():
    weight = np.array([[0, 0, 0], [127, -127, 63.5]], np.float32)
    q, scales = quantize_weight(weight)
    np.testing.assert_array_equal(q.T, [[0, 0, 0], [127, -127, 64]])
    np.testing.assert_array_equal(scales, [1, 1])
    assert q.flags.c_contiguous


def test_weight_quantization_rejects_nonfinite():
    with pytest.raises(ValueError, match="finite"):
        quantize_weight(np.array([[np.nan]], np.float32))


@pytest.mark.skipif(not os.environ.get("TRUTINA_MATMUL_LIBRARY"), reason="requires RK3588 NPU")
def test_real_npu_adaptive_stops_and_continues_without_recomputing():
    from test_torch_engine_cpu import make_engine

    torch.set_num_threads(4)
    engine = make_engine()
    engine.depth = "deep"
    with torch.inference_mode():
        reference = engine._read([[4, 8, 9]], [3])[0]
    runtime = MatmulRuntime(os.environ["TRUTINA_MATMUL_LIBRARY"])
    counts = [0]*4
    hooks = []
    try:
        assert offload_torso(engine.torso, runtime) > 0
        engine.npu_runtime = runtime
        for i, layer in enumerate(engine.torso.layers):
            def count(_m, _a, _out, i=i):
                counts[i] += 1
            hooks.append(layer.register_forward_hook(count))
        engine.depth = "adaptive"
        with torch.inference_mode():
            engine.threshold = 0
            engine._read([[4, 8, 9]], [3])
            early_calls = runtime.calls
            assert engine.last_exit == 2 and counts == [1, 1, 0, 0]
            assert early_calls > 0
            engine.threshold = 1
            got = engine._read([[4, 8, 9]], [3])[0]
            assert engine.last_exit == 4 and counts == [2, 2, 1, 1]
            assert runtime.calls - early_calls == 2*early_calls
        np.testing.assert_allclose(got, reference, atol=0.02)
    finally:
        for hook in hooks:
            hook.remove()
        engine.close()
        runtime.close()
