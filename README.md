# trutina

English | [中文](README.zh.md)

**trutina-4b** is a typed decision model: give it a state (text, optionally with images) and typed
questions, and it returns a calibrated probability for every option without generating any text.
It is Qwen3.5-4B with a **multi-exit readout**: the answer can be read at layer 12, 16, 20, 24 or 32,
and in the default **adaptive** mode each request stops at the first layer that is confident
enough, continuing from where it stopped when it is not. Easy questions finish in about half the
network; hard ones get all 32 layers.

Weights: [aldinelabs/trutina-4b](https://huggingface.co/aldinelabs/trutina-4b) · Licence: Apache-2.0

| Benchmark | Full depth (32 layers) | Adaptive (default) |
|---|---|---|
| **JevBench public set** (231 tasks, accuracy) | **87.0%** · p50 113 ms | 86.1% · 22.1 layers on average · p50 71 ms |
| **LocalLLaMA typed-decisions** (`test`, zero-shot, 2,000 questions) | **74.85%** · KL 0.149 · p50 201 ms | 74.85% · KL 0.178 · 28.2 layers · p50 177 ms |

- JevBench public: ties for 5th on the board's public set; the four systems above it generate reasoning text before answering.
- LocalLLaMA: 2nd in the zero-shot table and 1st among open-weight models.
- Latency is per question on an M5 Pro with MLX. Inputs: text and images; choice (up to 255 options), score (2–10 levels), yes/no.
- These are our own measurements on public sets; neither set was used for training or for picking weights (see [Evaluation](#evaluation)).

## Quickstart

Apple silicon (MLX):

```bash
pip install "trutina @ git+https://github.com/aldinelabs/trutina"
```

```python
from trutina import Trutina

model = Trutina.load("aldinelabs/trutina-4b")       # downloads the weights from Hugging Face
model.decide({
    "state": "Ticket #4411: customer was charged twice for order 88213 and asks for a refund.",
    "questions": {
        "route": {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": {"billing": "payments, refunds, invoices",
                               "shipping": "delivery and tracking",
                               "tech": "app or account problems"}},
        "urgent": {"type": "noul", "instructions": "Does this need a reply within the hour?"},
        "sentiment": {"type": "score", "instructions": "How upset is the customer?",
                      "criteria": ["calm", "annoyed", "angry"]},
    },
})
# {"answers": {
#   "route": {"type": "choice", "choice": "billing", "confidence": 0.876,
#             "probabilities": {"billing": 0.974, "shipping": 0.010, "tech": 0.016}},
#   "urgent": {"type": "noul", "noul": 0.737},
#   "sentiment": {"type": "score", "score": 0.967, "confidence": 0.160,
#                 "legend": {"0": "calm", "1": "annoyed", "2": "angry"},
#                 "probabilities": {"0": 0.205, "1": 0.623, "2": 0.172}}}}
```

HTTP server (`POST /v1/systemone` and the OpenAI-style `POST /v1/decisions`):

```bash
python scripts/serve.py --port 8000
```

```bash
curl -s localhost:8000/v1/systemone -d '{"state": "The invoice total is $1,240.", "questions": {"over": {"type": "noul", "instructions": "Is the total over $1,000?"}}}'
```

CUDA (PyTorch + transformers, text only; shallow, adaptive and full depth):

```python
from trutina.torch_engine import TorchTrutina

model = TorchTrutina.load("aldinelabs/trutina-4b", device="cuda")
```

```bash
python scripts/serve.py --engine torch --port 8000
```

Install `flash-linear-attention` and `causal-conv1d` for the fast linear-attention kernels.

## Request format

The `/v1/systemone` body is `{"state": ..., "questions": {...}, "depth": ...}`:

| Question type | `criteria` | Answer |
|---|---|---|
| `choice` | `{"<option>": "<description>", ...}`, 2–255 options | `choice`, `confidence`, `probabilities` |
| `score` | `["<lowest level>", ..., "<highest>"]` (or `{"label", "description"}` objects), 2–10 levels | `score` (expected level index), `probabilities` |
| `noul` | optional `{"true": "...", "false": "..."}` | `noul` = P(yes) |

`state` is a string, or a list of OpenAI-style content parts (`{"type": "input_text", "text": ...}`,
`{"type": "input_image", "image_url": "data:image/png;base64,..."}`). `confidence` is
1 − H(p) / log K. All questions in a request share the state and run as one batch.

`depth` picks how much of the network a request may use: `shallow` (answer at layer 12),
`adaptive` (default, below) or `deep` (all 32 layers; the MLX engine also averages over up to 4
rotations of each choice question's options).

## Adaptive depth

```
 tokens ─► layers 1–12 ─► exit 12 ─► confident? ── yes ─► answer
                              │ no
                              ▼  (continue from layer 12's hidden state, nothing is recomputed)
           layers 13–16 ─► exit 16 ─► confident? ── yes ─► answer
                              │ no
           layers 17–20 ─► exit 20 ─►      …
           layers 21–24 ─► exit 24 ─►      …
           layers 25–32 ─► final readout ─────────────────► answer
```

Every exit reads the same thing: a probability for each option label. "Confident" means the top
probability reaches one calibrated threshold (0.84) for every question in the request. Because the
pass continues from the exit's hidden state, an early miss costs only the small exit head, not
a second forward pass.

Accuracy of each exit on its own (temperature-calibrated; measured on the training checkpoint before
the LoRA was merged, which moves the released model's numbers by at most 0.1):

| Exit layer | 12 | 16 | 20 | 24 | 32 | adaptive |
|---|---|---|---|---|---|---|
| Our validation split | 50.0 | 73.5 | 85.8 | 87.4 | 87.9 | 87.5 |
| JevBench public | 42.9 | 71.9 | 87.0 | 87.9 | 87.0 | 86.1 |
| LocalLLaMA test | 43.5 | 59.3 | 72.8 | 75.2 | 74.75 | 74.75 |

Where adaptive answers on our validation split: layer 12 for 7% of questions, 16 for 28%, 20 for 26%,
24 for 9%, and the full 32 layers for 29%. That averages 22.2 layers, for a 0.4-point accuracy cost.
The threshold was set on our validation split as the most aggressive one that costs at most about
half a point of accuracy.

Without the trained exits this is impossible: reading the untrained model's LM head at
intermediate layers (a "logit lens") gives chance-level answers up to about layer 19.

## Architecture

```
 prompt ─► embed
             │
         layers 1–12  ──► lens 12 ──┐
         layers 13–16 ──► lens 16 ──┤
         layers 17–20 ──► lens 20 ──┤   LM-head rows of the option letters (A)…(Z)
         layers 21–24 ──► lens 24 ──┼─► at "Best answer: (" ─► ÷ exit temperature ─► softmax
         layers 25–32 ──► norm ─────┘

 backbone: Qwen3.5-4B (24 Gated DeltaNet + 8 attention layers), LoRA merged
```

- **Letter readout at one position.** Each question is rendered once: the chat template with
  thinking off, the options labelled (A), (B), … shown both before and after the input (a
  "sandwich", so the model reads the input already knowing the question and sees the options again
  right before answering), ending in `Best answer: (`. The answer is the softmax over the LM-head
  rows of the option letters at that position. Reading option letters from the LM head is shared
  with other decision models (e.g. decider-4b, JevK5); it keeps the backbone's own prior, and
  untrained it already reaches 79.7% on the JevBench public set.
- **Tuned-lens exits (ours).** Each early exit is one affine map in the residual stream,
  `norm(h + W h + b)`, with its own logit scale, about 6.5M parameters each and 26M in total. The
  exits read the same LM-head letter rows as the final layer. They are trained by distillation from
  the final layer's answer distribution, with their inputs detached: the exits learn to predict the
  final answer early, and they cannot pull the backbone away from it, so full-depth accuracy is the
  same as without them.
- **Resume, don't restart.** Adaptive inference runs the backbone in segments (layers 1–12, 13–16,
  …) and keeps the hidden states of every token at the segment boundary; the next segment starts
  from them. A request that does not stop at layer 12 has paid for layers 1–12 once, plus one
  2560×2560 matmul per question at the exit.
- **Calibrated per exit.** Each exit has its own temperature (0.83–1.04) fitted on our validation
  split, so a 0.84 at layer 16 means the same as a 0.84 at layer 32. That is what makes one shared
  stopping threshold work.
- **One batch per request.** The questions of a request share the state and run as one padded
  batch; the stop decision is per request, so latency is one forward pass, not one per question.
- **More than 26 options: a tournament (ours).** The model only ever saw the letters A–Z, so larger
  choice questions run in two rounds: balanced groups of at most 26, then a final among each group's
  leaders. Every option still gets a probability (finalists by the final, the rest by their group
  scores). On MASSIVE (59 intents, not in training) this gives 79.3% top-1, against 59.4% for the
  usual alternative, two-letter labels (AA, AB, …) that the model never trained on (`trutina/wide.py`).
- **Images.** The Qwen3.5 vision tower is used as is (frozen); image tokens go into the state with
  multimodal RoPE, and the readout is unchanged.
- **Fine-tune.** LoRA rank 16 on every linear layer (attention, Gated DeltaNet projections, MLP),
  trained on soft and hard labels from about 266k public decision questions plus 35k image
  questions. Options are shuffled and descriptions randomly dropped. The LoRA is merged into the
  released weights, so the checkpoint loads as a plain Qwen3.5-4B.

## Evaluation

All numbers are from the released weights unless marked. Each benchmark below was held out from
training, and weights were selected on our own validation loss only.

**JevBench public set** (231 tasks: easy, original and hard files; accuracy):

| Model | All | Hard (94) |
|---|---|---|
| trutina-4b, full depth | **87.0** | 73.4 |
| trutina-4b, adaptive | 86.1 | |
| Qwen3.5-4B with the same prompt, untrained | 79.7 | |

On the public leaderboard snapshot we checked (2026-10-12), four systems score higher on these
231 tasks: two API models with reasoning (97.8, 97.4) and two 26B-A4B models that generate
reasoning before answering (88.7, 87.4). The leaderboard's headline score also includes held-out
tasks, calibration and speed, which we cannot measure ourselves.

**LocalLLaMA typed-decisions** (`test`, 400 cases / 2,000 questions; zero-shot table):

| Model | Accuracy | KL ↓ |
|---|---|---|
| meraGPT Decider 1 (hosted API) | 76.8 | 0.096 |
| **trutina-4b** (full depth) | **74.85** | 0.149 |
| trutina-4b (adaptive, 28 layers on average) | 74.85 | 0.178 |
| Liquid AI d1 | 74.2 | 0.475 |
| TypeSafe Jev 1.13.0 | 72.7 | 1.442 |
| OpenDecider-small (open weights) | 67.1 | 0.211 |

Other rows are from the dataset card. We ask each question as its own sequence, while the card's
protocol sends the five questions of a case in one request; the model's output does not depend on
that grouping.

**Other held-out decision sets** (zero-shot = the same backbone and prompt without the fine-tune):

| Set | Zero-shot | trutina-4b |
|---|---|---|
| DecisionBench medium (state-macro accuracy) | 75.2 | **83.2** |
| DecisionBench hard (state-macro accuracy) | **53.6** | 49.6 |
| this-that complex decisions | 43.8 | **51.3** |
| fast-decisions (17 single-label domains) | 61.5 | 61.4 (ECE 0.251 → 0.122) |
| JudgeBench (pair accuracy, both orders) | 46.0 | **52.1** |

The gains outside our training distribution are moderate, and DecisionBench hard is worse than the
untrained backbone.

**Decision Index 0.3 (public suite), estimate.** About 44 (±3). This was scored with the official
scorer on a stratified sample of about 12,000 of the suite's requests (at least 48 per benchmark),
not on a full run.

**Images** (our validation split, in-distribution): Mind2Web 91.9, A-OKVQA 89.6, ScienceQA 96.2,
rendered workflow screenshots 84.9.

## Training data

About 266k questions from public datasets, plus about 35k image questions:

- decision and classification sets: tasksource, LegalBench, CUAD, MAUD, ShARC, HaluEval;
- reading comprehension and multi-hop: DROP, MuSiQue, HotpotQA, 2WikiMultihopQA, StrategyQA, TAT-QA, ROPES, BoolQ;
- graded and soft labels: ChaosNLI, SummEval, MT-Bench human judgements, Measuring Hate Speech, SBIC;
- community decision datasets, some generated by other open models;
- procedurally generated tasks with verifiable answers, and temporal reasoning (TempReason, Test of Time);
- images: A-OKVQA, ScienceQA, Mind2Web, rendered workflow screenshots.

Excluded from training:
- every benchmark above (Decision Index suites, DecisionBench, this-that, JudgeBench, fast-decisions, MASSIVE);
- data from the LocalLLaMA workflows;
- JevBench tasks and outputs of the Jev models.

We looked at the LocalLLaMA and JevBench public scores during development (as drift monitors and
when choosing the prompt format), so they are not perfectly blind.

## Limitations

- Multi-step date and number arithmetic in a single forward pass is weak (JevBench temporal tasks: 7/15).
- On contested multi-option judgements (DecisionBench hard) it does not beat the untrained backbone.
- Probabilities are calibrated on our validation mix; recalibrate on your own data for high-stakes thresholds.
- Inputs longer than about 32k tokens are slow on small GPUs; the readout itself has no length limit beyond the backbone's.
- Training data is almost entirely English.

## Repository

```
src/trutina/
  engine.py        MLX engine: load a release, depth modes, images
  torch_engine.py  CUDA engine (text; shallow / adaptive / full depth)
  readout.py       the prompt format shared by training and inference
  wide.py          tournament for more than 26 options
  api.py           /v1/systemone parsing and rendering; decisions_api.py: /v1/decisions
  backbone.py      Qwen3.5 forward with early exits; mlx_vision.py: vision tower and multimodal RoPE
  vision.py        image preprocessing; schema.py: request types
  model.py, head.py, decider.py, train.py, pt/   earlier pointer-head research models (v0/v1)
scripts/
  serve.py, decide.py      serving
  export_release.py        build a release directory from a training run
  webui.py, ../webui/      training dashboard
tests/
```

```bash
uv sync --extra torch
uv run pytest
```

## Licence

Apache-2.0 for the code and the weights. The weights are derived from
[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache-2.0).
