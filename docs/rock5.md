# ROCK 5B / RK3588

Trutina runs locally on a 16 GB ROCK 5B. The ARM CPU backend supports text,
images, every decision type, wide choices, and all three depth modes. The RKLLM
backend uses the RK3588 NPU for full-depth text decisions. RKLLM does not expose
intermediate decoder layers or continuation from an intermediate layer.

For one endpoint that preserves every feature, use `--engine rock5`. It routes
deep text to the NPU and shallow/adaptive/image requests to the CPU, releases the
previous model before loading the next, and never keeps both backbones resident.
The first request after a switch includes loading: CPU quantization can take about
two to three minutes. Keep requests on one backend when latency matters, or use either
fixed backend below. `X-Trutina-Backend` reports `cpu` or `npu` in combined mode.

| Capability | PyTorch ARM CPU | RKLLM NPU | Combined `rock5` |
|---|---|---|---|
| choice / score / noul | Yes | Yes | Yes |
| 27-255 options | Tournament | Tournament | Tournament |
| Images | `--vision` | Not supported | CPU |
| `shallow` | Layer 12 | HTTP 400 | CPU, layer 12 |
| `adaptive` | Trained exits/continuation | Final exit, 32 | CPU, trained exits/continuation |
| `deep` text | Layer 32 | Layer 32 | NPU, layer 32 |
| Calibration / yes-no mapping | Release parameters | Release parameters | Release parameters |

Quantization can change the probabilities and early-exit decisions. The release's
temperatures and threshold are preserved, but quantized probabilities have not
been recalibrated on a held-out dataset. Do not interpret the smoke results below
as benchmark accuracy or a latency distribution.

## CPU installation and serving

On the board, use Python 3.11 or later and an isolated environment:

```bash
git clone https://github.com/aldinelabs/trutina.git
cd trutina
python3 -m venv .venv
source .venv/bin/activate
# Prefer the CPU PyTorch wheel when available for your Python/aarch64 platform.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[torch]'
hf download aldinelabs/trutina-4b --local-dir models/trutina-4b
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python scripts/serve.py \
  --engine torch --device cpu --cpu-int8 --vision --threads 4 \
  --batch-tokens 512 --model models/trutina-4b --port 8000
```

The server listens on localhost. Use `--host <board-LAN-IP>` for LAN access, or an
SSH tunnel (`ssh -L 8000:localhost:8000 <board>`) to retain localhost binding.
Both `POST /v1/systemone` and `POST /v1/decisions` use the existing request format.
`X-Trutina-Exit-Layer` reports the exit actually used.

CPU INT8 uses QNNPACK for decoder Linear layers. Embeddings, normalization,
convolution, DeltaNet state calculations and the vision tower stay floating point.
Weights are loaded in BF16 and decoder layers are converted one at a time to limit
conversion memory. Startup includes this conversion on every launch.

Optional persistent user service, installed from the same virtual environment:

```bash
python scripts/install_rock5_service.py --model models/trutina-4b --vision
sudo loginctl enable-linger "$USER"
systemctl --user status trutina
journalctl --user -u trutina -f
```

The service starts on localhost and restarts on failure. Re-running the installer
backs up an existing unit, updates it and restarts the service. Use its `--host`
and `--port` options when an explicit LAN binding is needed.
The 16 GB board service caps its cgroup at 13 GiB so startup/packing buffers cannot
exhaust the host. Freed conversion buffers are returned to glibc between layers.

## Reproduce the board experiment

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python scripts/probe_rock5.py \
  --engine torch --model models/trutina-4b --cpu-int8 --vision \
  --out cpu-results.json
python -m pytest -q tests/test_rkllm.py tests/test_torch_engine_cpu.py \
  tests/test_api.py tests/test_decisions_api.py
```

Measured on a 16 GB ROCK 5B, Ubuntu 24.04, kernel `6.1.0-1025-rockchip`,
PyTorch `2.14.1+cu130`, Transformers `5.19.0`, four CPU threads, release revision
`a99efbae7c5a2221c4d66cdf8d5e3b638636113c`. The CUDA wheel was installed during
the experiment, but computation ran entirely on CPU. Raw measurements are in
[rock5b-cpu-int8.json](benchmarks/rock5b-cpu-int8.json).

| Single smoke request | Exit | Seconds |
|---|---:|---:|
| Short invoice yes/no, shallow | 12 | 10.78 |
| Same invoice, adaptive | 16 | 7.92 |
| Same invoice, deep | 32 | 19.36 |
| Choice and score together, deep | 32 | 22.47 |
| 27 options, two rounds, deep | 32 | 95.95 |
| 256 x 256 red image, adaptive | 16 | 11.51 |

The deep choice picked red, the wide question picked 27, and the image question
picked red. Model loading and quantization took 109.6 s; process peak RSS across
startup and all probes was 14,548 MiB. This is close to a 16 GB board's usable RAM:
run one model process and avoid other memory-heavy jobs during startup. These are
sequential samples, including a cold first shallow inference; they are not p50/p95.
Those initial measurements predate the allocator trimming and service memory cap.
The subsequent real HTTP test returned the same invoice probability (0.97497),
exit 16, HTTP 200 through both routes: 14.41 s for the first `/v1/systemone`
request and 6.09 s for the warmed `/v1/decisions` request.

## RKLLM conversion

Use the official SDK **1.3.1**, pinned to commit
`4f7390530443bf84f0394255a449d7cbe81e69d1c`. The converter runs on Linux x86_64;
the runtime runs on Linux aarch64. Qwen3.5 support was added in SDK 1.3.0.

```bash
git clone https://github.com/airockchip/rknn-llm.git /tmp/rknn-llm
git -C /tmp/rknn-llm checkout 4f7390530443bf84f0394255a449d7cbe81e69d1c
docker build --platform linux/amd64 -t trutina-rkllm-converter \
  -f native/rkllm/Dockerfile /tmp/rknn-llm/rkllm-toolkit
mkdir -p tmp
docker run --rm --platform linux/amd64 \
  -v "$PWD:/work" -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  trutina-rkllm-converter --model aldinelabs/trutina-4b \
  --out /work/tmp/trutina-rk3588 --calibration native/rkllm/calibration.jsonl
```

On Apple silicon, Docker emulates x86_64. Allow 36 GiB for FP32 conversion of this
checkpoint: a 24 GiB VM completed layer optimization but ran out of memory during
final serialization in our experiment. Native x86_64 is preferable for speed.
The example calibration file contains synthetic text requests and is suitable for
conversion smoke testing. For deployment, supply a JSONL sample of your actual
input distribution using `--calibration`, then evaluate accuracy and calibration.
Calibration files remain local and are included in the output bundle.

The exporter converts the **merged Trutina checkpoint**, not the original Qwen
checkpoint. It preserves tokenizer/chat template and `trutina.json`; `rkllm.json`
records the context cap, quantization and final readout scale. The manifest is
written only after successful export. Use a fresh output directory for each run.

## NPU runtime

Check the board driver first:

```bash
cat /sys/module/rknpu/version
```

Rockchip's deployment documentation specifies RKNPU **0.9.8 or later**. Our board
successfully ran the short text/tournament probes below on its existing 0.9.7
driver after fixing device permissions. This is an observed compatibility result,
not a guarantee for other kernels, models or long contexts. Prefer the documented
driver version for a new installation. On some images RKNPU is built into the
kernel; installing a userspace library cannot upgrade it.
The serving user must have access to the NPU DRM nodes. SDK 1.3.1 opens
`/dev/dri/card*`, which belong to `video` on Ubuntu; render nodes belong to
`render`. Use `sudo usermod -aG video,render "$USER"`, then a fresh login.
An already running, lingering systemd user manager retains its old groups. Refresh
it with `sudo systemctl restart user@$(id -u).service` when suitable; this restarts
that user's services. Check the serving process's `Groups:` in `/proc/<pid>/status`
if SSH probes work but the persistent service cannot open the NPU.

Copy the exported bundle and the pinned SDK to the board, then:

```bash
bash native/rkllm/build.sh /path/to/rknn-llm
pip install -e '.[rkllm]'
python scripts/serve.py --engine rkllm --model /path/to/trutina-rk3588 \
  --rkllm-library build/rkllm/libtrutina_rkllm.so --port 8000
python scripts/probe_rock5.py --engine rkllm --model /path/to/trutina-rk3588 \
  --out npu-results.json
```

After both bundles are installed, the combined service is:

```bash
python scripts/serve.py --engine rock5 --model models/trutina-4b \
  --rkllm-model models/trutina-rk3588
# Or persist it as a user service:
python scripts/install_rock5_service.py --engine rock5 --model models/trutina-4b \
  --rkllm-model models/trutina-rk3588
```

ROCK 5B NPU smoke results (same release as the CPU table; RKLLM 1.3.1,
W8A8, three NPU cores, 24 synthetic calibration prompts, context cap 4096):

| Request | Seconds | Result |
|---|---:|---|
| Short invoice, full depth | 2.07 | P(yes) 0.98647 |
| Choice and score together | 4.11 | red; score 1.826 / 2 |
| Opposite invoice ($240) | 2.06 | P(yes) 0.00819 |
| Original invoice again | 2.07 | P(yes) 0.98647, unchanged |
| 27 options, two rounds | 16.04 | 27 |

Cold load took 25.56 s; process peak RSS was 5,873 MiB. Raw measurements are in
[rock5b-rkllm-w8a8.json](benchmarks/rock5b-rkllm-w8a8.json). As with the CPU table,
these are individual smoke measurements, not a benchmark accuracy claim or p50.
The opposite/repeated requests verify that this test did not carry decision state
between independent calls. The CPU and NPU quantizations do change some score
probabilities, so both need evaluation on the intended deployment distribution.

The full combined smoke probe, including a CPU→NPU→CPU transition and image input,
also passed. Raw results: [rock5b-combined.json](benchmarks/rock5b-combined.json).
The first deep request after CPU took 50.67 s including switching/loading; the
first image request after NPU took 156.04 s. Process peak RSS was 13,706 MiB.
Use `scripts/probe_rock5_http.py` to check the same transitions through a running
combined service and verify both HTTP routes plus backend/exit response headers.
The persistent service passed that probe with a 13 GiB cgroup cap: deep NPU,
image CPU, adaptive CPU and deep NPU again all returned HTTP 200 with the expected
backends/exits. Raw results: [rock5b-combined-http.json](benchmarks/rock5b-combined-http.json).
Switch-inclusive requests took 55.82 s, 128.86 s and 53.27 s; the warmed CPU
adaptive request took 6.02 s. `/health` reports a null backend before the first
lazy load or during a backend switch.

The bridge compiles against the actual C++ header and checks pinned SDK file
hashes. Python does not re-declare Rockchip's changing structures. Tokens are
rendered by the release tokenizer, chat wrapping is disabled in the runtime,
history is disabled, and the KV cache is cleared before every independent pass.
The callback copies only the selected last-position logits before releasing the
runtime output buffer. Non-finite logits and missing outputs fail explicitly.

Hardware-free native contract test (also run on the board):

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pthread \
  -I/path/to/rknn-llm/rkllm-runtime/Linux/librkllm_api/include \
  native/rkllm/test_bridge.cpp -o /tmp/trutina-test-bridge
/tmp/trutina-test-bridge
```

Sources: [official SDK](https://github.com/airockchip/rknn-llm),
[runtime header](https://github.com/airockchip/rknn-llm/blob/4f7390530443bf84f0394255a449d7cbe81e69d1c/rkllm-runtime/Linux/librkllm_api/include/rkllm.h),
[conversion example](https://github.com/airockchip/rknn-llm/blob/4f7390530443bf84f0394255a449d7cbe81e69d1c/examples/rkllm_api_demo/export/export_rkllm.py).
