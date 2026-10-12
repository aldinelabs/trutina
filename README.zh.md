# trutina

[English](README.md) | 中文

**trutina-4b** 是一个类型化决策模型（typed decision model）：给它一段 state（文字，可以带图片）和若干带类型的问题，它为每个选项返回一个校准过的概率，全程不生成任何文本。
它是在 Qwen3.5-4B 上加了**多出口读出（multi-exit readout）**：答案可以在第 12、16、20、24 或 32 层读出。默认的 **adaptive** 模式下，每个请求在第一个足够有把握的出口停下；不够有把握就从停下的地方接着往下算。简单题大约只用一半的网络，难题会用满 32 层。

权重：[aldinelabs/trutina-4b](https://huggingface.co/aldinelabs/trutina-4b) · 许可证：Apache-2.0

| 基准 | 满深度（32 层） | adaptive（默认） |
|---|---|---|
| **JevBench 公开集**（231 题，准确率） | **87.0%** · p50 113 ms | 86.1% · 平均 22.1 层 · p50 71 ms |
| **LocalLLaMA typed-decisions**（`test`，零样本，2,000 题） | **74.85%** · KL 0.149 · p50 201 ms | 74.85% · KL 0.178 · 平均 28.2 层 · p50 177 ms |

- JevBench 公开集：在榜单公开集上并列第五；排在前面的四个系统都要先生成推理文本再作答。
- LocalLLaMA：零样本榜第二，开放权重模型中第一。
- 上表延迟是 M5 Pro 上用 MLX 单题测得。在单张 RTX PRO 6000（CUDA，含 HTTP）上：满深度 p50 28.3 ms，adaptive 21.1 ms（[详见评测](#评测)）。支持的输入：文字和图片；choice（最多 255 个选项）、score（2–10 级）、是否题。
- 以上都是我们在公开集上自测的；这两个集都没有用于训练，也没有用于挑选权重（见[评测](#评测)）。

## 快速上手

Apple silicon（MLX）：

```bash
pip install "trutina @ git+https://github.com/aldinelabs/trutina"
```

```python
from trutina import Trutina

model = Trutina.load("aldinelabs/trutina-4b")       # 从 Hugging Face 下载权重
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

HTTP 服务（`POST /v1/systemone`，以及 OpenAI 风格的 `POST /v1/decisions`）：

```bash
python scripts/serve.py --port 8000
```

```bash
curl -s localhost:8000/v1/systemone -d '{"state": "The invoice total is $1,240.", "questions": {"over": {"type": "noul", "instructions": "Is the total over $1,000?"}}}'
```

CUDA（PyTorch + transformers，仅文字，支持 shallow / adaptive / 满深度）：

```python
from trutina.torch_engine import TorchTrutina

model = TorchTrutina.load("aldinelabs/trutina-4b", device="cuda")
```

```bash
python scripts/serve.py --engine torch --port 8000
```

安装 `flash-linear-attention` 和 `causal-conv1d` 可以用上线性注意力的快速 kernel。

## 请求格式

`/v1/systemone` 的请求体是 `{"state": ..., "questions": {...}, "depth": ...}`：

| 题型 | `criteria` | 返回 |
|---|---|---|
| `choice` | `{"<选项>": "<描述>", ...}`，2–255 个选项 | `choice`、`confidence`、`probabilities` |
| `score` | `["<最低级>", ..., "<最高级>"]`（或 `{"label", "description"}` 对象），2–10 级 | `score`（等级的期望值）、`probabilities` |
| `noul` | 可选 `{"true": "...", "false": "..."}` | `noul` = P(yes) |

`state` 可以是字符串，也可以是 OpenAI 风格的内容片段列表（`{"type": "input_text", "text": ...}`、`{"type": "input_image", "image_url": "data:image/png;base64,..."}`）。`confidence` = 1 − H(p) / log K。同一请求里的所有问题共享 state，作为一个 batch 一起算。

`depth` 决定一个请求最多用多少层：`shallow`（在第 12 层作答）、`adaptive`（默认，见下）或 `deep`（满 32 层；MLX 引擎上还会对每道 choice 题的选项做最多 4 种循环轮换后取平均）。

## Adaptive 深度

```
 tokens ─► 第 1–12 层 ─► 出口 12 ─► 有把握？ ── 是 ─► 作答
                             │ 否
                             ▼  （从第 12 层的隐状态继续，不重算）
           第 13–16 层 ─► 出口 16 ─► 有把握？ ── 是 ─► 作答
                             │ 否
           第 17–20 层 ─► 出口 20 ─►      …
           第 21–24 层 ─► 出口 24 ─►      …
           第 25–32 层 ─► 最终读出 ──────────────────► 作答
```

每个出口读的都是同一样东西：每个选项标签的概率。「有把握」的意思是：请求里每道题的最高概率都达到同一个校准后的阈值（0.84）。由于是从出口处的隐状态继续往下算，没在早期出口停下的代价只是一个很小的出口头，而不是再跑一遍前向。

各出口单独作答的准确率（经过温度校准；在合并 LoRA 之前的训练检查点上测得，合并后发布模型的数字最多相差 0.1）：

| 出口层 | 12 | 16 | 20 | 24 | 32 | adaptive |
|---|---|---|---|---|---|---|
| 我们的验证集 | 50.0 | 73.5 | 85.8 | 87.4 | 87.9 | 87.5 |
| JevBench 公开集 | 42.9 | 71.9 | 87.0 | 87.9 | 87.0 | 86.1 |
| LocalLLaMA test | 43.5 | 59.3 | 72.8 | 75.2 | 74.75 | 74.75 |

adaptive 在验证集上的作答位置：7% 的题在第 12 层，28% 在第 16 层，26% 在第 20 层，9% 在第 24 层，29% 用满 32 层。平均 22.2 层，准确率只损失 0.4 个点。阈值是在验证集上选的：在准确率损失不超过约半个点的前提下，选最激进的那个。

没有训练过的出口就做不到这一点：直接在未训练模型的中间层读 LM head（即「logit lens」），到第 19 层左右之前都是随机水平。

## 架构

```
 prompt ─► embed
             │
         第 1–12 层  ──► lens 12 ──┐
         第 13–16 层 ──► lens 16 ──┤
         第 17–20 层 ──► lens 20 ──┤   LM head 中选项字母 (A)…(Z) 对应的行
         第 21–24 层 ──► lens 24 ──┼─► 在 "Best answer: (" 处读出 ─► ÷ 出口温度 ─► softmax
         第 25–32 层 ──► norm ─────┘

 骨干：Qwen3.5-4B（24 层 Gated DeltaNet + 8 层注意力），LoRA 已合并
```

- **在一个位置读选项字母。** 每道题渲染一次：chat template（关闭 thinking），选项标为 (A)、(B)……，在输入的前后各出现一次（「三明治」结构：模型读输入时已经知道问题，作答前又能再看一遍选项），最后以 `Best answer: (` 结尾。答案就是该位置上 LM head 里各选项字母对应行的 softmax。从 LM head 读选项字母这一点和其他决策模型相同（如 decider-4b、JevK5）；它保留了骨干自身的先验，不训练就能在 JevBench 公开集上拿到 79.7%。
- **Tuned-lens 出口（我们的设计）。** 每个早期出口是残差流上的一个仿射变换 `norm(h + W h + b)`，带各自的 logit 缩放，每个约 650 万参数，共 2600 万。出口读的是和最后一层相同的 LM head 字母行。训练方式是蒸馏最后一层的答案分布，并且出口的输入做了 detach：出口学的是「提前预测最终答案」，却不能把骨干往别处拉，所以满深度的准确率和没有出口时一样。
- **接着算，而不是重来。** adaptive 推理把骨干分段运行（第 1–12 层、13–16 层……），保留每个 token 在分段边界处的隐状态，下一段从这里继续。一个没在第 12 层停下的请求，第 1–12 层只算过一次，额外代价只是每道题在出口处的一次 2560×2560 矩阵乘。
- **每个出口单独校准。** 每个出口有自己的温度（0.83–1.04），在验证集上拟合，所以第 16 层的 0.84 和第 32 层的 0.84 含义相同。这正是一个共享停止阈值能成立的原因。
- **一个请求一个 batch。** 同一请求的问题共享 state，作为一个 padded batch 一起跑；是否停下按整个请求判断，所以延迟是一次前向，而不是每题一次。
- **超过 26 个选项：分组淘汰（我们的设计）。** 模型只见过字母 A–Z，所以更多选项的 choice 题分两轮：先均分成每组不超过 26 个，再让各组的领先者进入决赛。每个选项仍然都有概率（决赛选手按决赛结果，其余按组内得分）。在 MASSIVE（59 种意图，不在训练集中）上 top-1 为 79.3%，而常见做法——模型从未训练过的两字母标签（AA、AB……）——只有 59.4%（`trutina/wide.py`）。
- **图片。** 直接使用 Qwen3.5 的视觉塔（冻结），图片 token 放进 state，配合多模态 RoPE，读出方式不变。
- **微调。** 在每个线性层（注意力、Gated DeltaNet 投影、MLP）上做 rank 16 的 LoRA，用约 26.6 万道公开决策题和约 3.5 万道图片题的软标签与硬标签训练；训练时打乱选项顺序，并随机去掉选项描述。LoRA 已合并进发布的权重，所以检查点可以当作普通的 Qwen3.5-4B 加载。

## 评测

除特别注明外，所有数字都来自发布的权重。下面每个基准都没有参与训练，挑选权重只看我们自己的验证损失。

**JevBench 公开集**（231 题：easy、original、hard 三个文件；准确率）：

| 模型 | 全部 | Hard（94 题） |
|---|---|---|
| trutina-4b，满深度 | **87.0** | 73.4 |
| trutina-4b，adaptive | 86.1 | |
| Qwen3.5-4B，相同 prompt，未训练 | 79.7 | |

在我们查看的公开榜单快照（2026-10-12）上，这 231 题中比我们高的有四个系统：两个带推理的 API 模型（97.8、97.4），和两个先生成推理再作答的 26B-A4B 模型（88.7、87.4）。榜单的总分还包括封闭集、校准和速度，这些我们自己测不了。

**LocalLLaMA typed-decisions**（`test`，400 个 case / 2,000 题；零样本榜）：

| 模型 | 准确率 | KL ↓ |
|---|---|---|
| meraGPT Decider 1（托管 API） | 76.8 | 0.096 |
| **trutina-4b**（满深度） | **74.85** | 0.149 |
| trutina-4b（adaptive，平均 28 层） | 74.85 | 0.178 |
| Liquid AI d1 | 74.2 | 0.475 |
| TypeSafe Jev 1.13.0 | 72.7 | 1.442 |
| OpenDecider-small（开放权重） | 67.1 | 0.211 |

其他行来自数据集卡片。我们每道题单独作为一条序列；卡片的协议是一个 case 的五道题放在一个请求里发。模型的输出不受这种分组影响。

**其他 held-out 决策集**（零样本 = 相同骨干和 prompt，不加微调）：

| 数据集 | 零样本 | trutina-4b |
|---|---|---|
| DecisionBench medium（按 state 平均的准确率） | 75.2 | **83.2** |
| DecisionBench hard（按 state 平均的准确率） | **53.6** | 49.6 |
| this-that complex decisions | 43.8 | **51.3** |
| fast-decisions（17 个单标签领域） | 61.5 | 61.4（ECE 0.251 → 0.122） |
| JudgeBench（两种顺序都对才算对） | 46.0 | **52.1** |

训练分布之外的提升幅度中等，DecisionBench hard 还不如未训练的骨干。

**Decision Index 0.3（公开套件），估计值。** 约 44（±3）。这是用官方打分器在约 1.2 万条请求的分层抽样上算的（每个基准至少 48 条），不是全量运行。

**单张 RTX PRO 6000 上的延迟**（串行，一次一个请求，经过 `scripts/serve.py --engine torch`，含 HTTP 往返，预热后测；使用 JevBench 公开集 231 题）。作为对照，下面也列出 JevBench v1.5.4 在其自有 GPU 节点上测得的原始延迟（串行，每个请求一个判断，使用其 450 道公开 standard 和 judge 题）：

| 系统 | GPU | p50 ms | p95 ms | 测量方 |
|---|---|---|---|---|
| **trutina-4b，adaptive** | RTX PRO 6000 | **21.1** | 107.4 | 我们 |
| **trutina-4b，满深度** | RTX PRO 6000 | **28.3** | 112.1 | 我们 |
| Cygnet（Gemma-4-12B） | RTX PRO 6000 | 39.8 | 99.1 | JevBench |
| AutoJev-27B | RTX PRO 6000 | 88.1 | 225.9 | JevBench |
| Surogate Rune 26B-A4B v3 | RTX PRO 6000 | 100.5 | 285.6 | JevBench |
| JevK5 v0.3（4B） | H100 | 15.8 | 46.5 | JevBench |
| Plumb-4B | H100 | 17.3 | 47.4 | JevBench |
| decider-4b v2 | RTX 5090 | 26.4 | 127.5 | JevBench |
| Jev 1.13.0 | 托管 API | 616.5 | 674.0 | JevBench |

两边的请求集不同（我们的包含公开集里的长政策文档题，p95 主要由它们决定），所以这张表只看量级，不作排名。adaptive 的平均延迟 33.7 ms（平均 22.1 层）；满深度的平均值是 99 ms，因为有一个请求遇到了一次性的 kernel 编译（6.7 s），它的 p80 是 46.9 ms。同一次运行的准确率：满深度 87.0%，adaptive 86.1%，与 MLX 完全一致。

**图片**（我们的验证集，属于训练分布内）：Mind2Web 91.9、A-OKVQA 89.6、ScienceQA 96.2、渲染的工作流截图 84.9。

## 训练数据

约 26.6 万道来自公开数据集的题目，外加约 3.5 万道图片题：

- 决策与分类：tasksource、LegalBench、CUAD、MAUD、ShARC、HaluEval；
- 阅读理解与多跳：DROP、MuSiQue、HotpotQA、2WikiMultihopQA、StrategyQA、TAT-QA、ROPES、BoolQ；
- 分级标签与软标签：ChaosNLI、SummEval、MT-Bench 人工评判、Measuring Hate Speech、SBIC；
- 社区的决策数据集，其中一部分由其他开源模型生成；
- 程序化生成、答案可验证的任务，以及时间推理（TempReason、Test of Time）；
- 图片：A-OKVQA、ScienceQA、Mind2Web、渲染的工作流截图。

排除在训练之外的：
- 上面列出的所有基准（Decision Index 各套件、DecisionBench、this-that、JudgeBench、fast-decisions、MASSIVE）；
- 来自 LocalLLaMA 工作流的数据；
- JevBench 的题目以及 Jev 系列模型的输出。

开发过程中我们看过 LocalLLaMA 和 JevBench 公开集的分数（作为漂移监测，以及选择 prompt 格式时），所以它们并不是完全盲测。

## 局限

- 单次前向里做多步日期和数字运算比较弱（JevBench 时间类题目：7/15）。
- 在有争议的多选判断上（DecisionBench hard）不如未训练的骨干。
- 概率是按我们的验证集分布校准的；如果要用在高风险的阈值上，请在自己的数据上重新校准。
- 在小显存 GPU 上，超过约 3.2 万 token 的输入会很慢；读出本身除了骨干的上下文长度外没有长度限制。
- 训练数据几乎全是英文。

## 仓库结构

```
src/trutina/
  engine.py        MLX 引擎：加载发布版、深度模式、图片
  torch_engine.py  CUDA 引擎（文字；shallow / adaptive / 满深度）
  readout.py       训练和推理共用的 prompt 格式
  wide.py          超过 26 个选项时的分组淘汰
  api.py           /v1/systemone 的解析与输出；decisions_api.py：/v1/decisions
  backbone.py      带早退出的 Qwen3.5 前向；mlx_vision.py：视觉塔与多模态 RoPE
  vision.py        图片预处理；schema.py：请求类型
  model.py, head.py, decider.py, train.py, pt/   早期 pointer-head 研究模型（v0/v1）
scripts/
  serve.py, decide.py      服务
  export_release.py        从训练产物构建发布目录
  webui.py, ../webui/      训练看板
tests/
```

```bash
uv sync --extra torch
uv run pytest
```

## 许可证

代码和权重均为 Apache-2.0。权重派生自 [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)（Apache-2.0）。
