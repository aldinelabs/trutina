"""Check real NPU integer products, chunking and weight rebinding before serving."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from trutina.npu_matmul import MatmulRuntime, quantize_weight


def reference(x, weight):
    q, sw = quantize_weight(weight)
    maximum = np.max(np.abs(x), axis=1)
    sx = np.where(maximum > 0, maximum / 127, 1).astype(np.float32)
    qx = np.clip(np.rint(x / sx[:, None]), -127, 127).astype(np.int32)
    return (qx @ q.astype(np.int32)).astype(np.float32) * sx[:, None] * sw[None]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cores", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    runtime = MatmulRuntime(args.library, cores=args.cores)
    measurements = []
    try:
        weights = [rng.normal(size=(128, 256)).astype(np.float32) for _ in range(2)]
        handles = [runtime.linear(w) for w in weights]
        for length in (7, 64, 65):
            x = rng.normal(size=(length, 256)).astype(np.float32)
            x[0] = 0
            for index in (0, 1, 0):
                start = time.perf_counter()
                got = runtime.forward(handles[index], x, 128)
                seconds = time.perf_counter() - start
                expected = reference(x, weights[index])
                np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-6)
                measurements.append({"rows": length, "weight": index, "seconds": seconds,
                                     "max_abs_error": float(np.max(np.abs(got-expected)))})
        x[0, 0] = np.nan
        try:
            runtime.forward(handles[0], x, 128)
        except RuntimeError as e:
            assert "Non-finite" in str(e)
        else:
            raise AssertionError("NPU bridge accepted NaN activations")
        calls = runtime.calls
    finally:
        runtime.close()
    try:
        runtime.forward(handles[0], np.zeros((1, 256), np.float32), 128)
    except RuntimeError as e:
        assert "closed" in str(e)
    else:
        raise AssertionError("Closed runtime accepted inference")
    args.out.write_text(json.dumps({"passed": True, "cores": args.cores,
                                    "npu_calls": calls, "measurements": measurements}, indent=2) + "\n")
    print(f"NPU matmul checks passed, {calls} hardware calls", flush=True)


if __name__ == "__main__":
    main()
