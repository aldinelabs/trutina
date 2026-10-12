# trutina

**trutina-4b** is a typed decision model: give it a state (text, optionally with images) and typed
questions, and it returns a calibrated probability for every option, in one forward pass and
without generating any text. It is Qwen3.5-4B fine-tuned to read its answer from the
language-model head at one position, with tuned-lens early exits for an adaptive depth mode.

Weights: [aldinelabs/trutina-4b](https://huggingface.co/aldinelabs/trutina-4b) · Licence: Apache-2.0

| | |
|---|---|
| **JevBench public set** (231 tasks) | **87.0%** accuracy, full depth · 86.1% adaptive (22 layers on average) · ties for 5th on the board's public set; the four above generate reasoning first |
| **LocalLLaMA typed-decisions test** (zero-shot, 2,000 questions) | **74.85%** accuracy, KL 0.149 · 2nd overall, 1st among open weights |
| Latency (M5 Pro, MLX, one question) | JevBench p50 113 ms full depth, 71 ms adaptive · LocalLLaMA p50 201 ms, 177 ms |
| Inputs | text and images; choice (up to 255 options), score (2–10 levels), yes/no |

These are our own measurements on public sets; neither set was used for training or for picking
weights (details under [Evaluation](#evaluation)).

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

CUDA (PyTorch + transformers, text only, full depth):

```python
from trutina.torch_engine import TorchTrutina

model = TorchTrutina.load("aldinelabs/trutina-4b", device="cuda")
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

`depth` trades latency for accuracy:

| `depth` | What runs | JevBench public | Layers |
|---|---|---|---|
| `shallow` | stop at the first exit (layer 12) | — | 12 |
| `adaptive` (default) | exits at layers 12, 16, 20, 24; stop at the first one whose top probability reaches the calibrated threshold for every question | 86.1% | 22 on average |
| `deep` | all 32 layers, averaged over up to 4 rotations of each choice question's options | — | 32 × up to 4 |

Full depth without rotations (what the benchmark numbers use) is 87.0%.

## How it works

- **Readout.** Each question becomes one prompt: the chat template with thinking off, the options
  labelled (A), (B), … shown before and after the input (a "sandwich"), ending in `Best answer: (`.
  The answer distribution is the softmax over the LM-head rows of the option letters at that
  position, divided by a calibrated temperature. Nothing is generated.
- **Fine-tune.** LoRA (rank 16, every linear layer) on soft and hard labels from about 266k
  public decision questions, with option order shuffled and descriptions randomly dropped. The LoRA
  is merged into the released weights.
- **Early exits.** One tuned-lens transform per exit at layers 12, 16, 20 and 24, distilled from
  the final layer. `adaptive` stops early when the exit is confident enough.
- **Calibration.** A temperature per exit, fitted on our validation split.
- **More than 26 options.** The model only ever saw the letters A–Z, so larger choice questions run
  as a two-round tournament: groups of at most 26, then a final among each group's leaders
  (`trutina/wide.py`). On MASSIVE (59 intents, not in training) this gives 79.3% top-1, against
  59.4% for untrained two-letter labels.
- **Images.** The Qwen3.5 vision tower is used as is (frozen); image tokens are placed in the state.

## Evaluation

All numbers are from the released weights unless marked. Each benchmark below was held out from
training, and weights were selected on our own validation loss only.

**JevBench public set** (231 tasks: easy, original and hard files; accuracy):

| | All | Hard (94) |
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
  torch_engine.py  CUDA engine (text, full depth)
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
