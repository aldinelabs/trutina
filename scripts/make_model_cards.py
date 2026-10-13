"""Build the Hugging Face model cards (README.md, README.zh.md) for a release directory from the repo READMEs.

The card is the repo README with the HF front matter, links that work on the Hub (the Chinese card, images
from GitHub), the weights link swapped for the code link, and the repository-layout section replaced by a
short note on what the weights repo holds.

  python scripts/make_model_cards.py runs/release/trutina-4b-v3
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = "https://raw.githubusercontent.com/aldinelabs/trutina/main/"
FRONT = """---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
language:
- en
library_name: trutina
pipeline_tag: zero-shot-classification
tags:
- decision-model
- typed-decisions
- calibration
- adaptive-computation
- early-exit
- mlx
- qwen3.5
- vision
---

"""
CODE = {
    "en": ("## Repository", "## Licence", """## Code

Inference engines, the HTTP server, the release exporter and the training dashboard live at
[github.com/aldinelabs/trutina](https://github.com/aldinelabs/trutina). This repository holds the
weights: a standard Qwen3.5-4B checkpoint layout with the fine-tune merged in (it also loads with
mlx-lm or transformers as a plain language model), `readout.safetensors` (the four exit lenses and
logit scales) and `trutina.json` (prompt format version, per-exit temperatures, adaptive threshold,
yes/no mapping).

"""),
    "zh": ("## 仓库结构", "## 许可证", """## 代码

推理引擎、HTTP 服务、发布导出脚本和训练看板在
[github.com/aldinelabs/trutina](https://github.com/aldinelabs/trutina)。本仓库存放权重：标准的
Qwen3.5-4B 检查点布局，微调已合并（也可以用 mlx-lm 或 transformers 当作普通语言模型加载）；
`readout.safetensors`（四个出口的 lens 和 logit 缩放）；`trutina.json`（prompt 格式版本、各出口温度、
adaptive 阈值、是非题映射）。

"""),
}


def card(lang: str) -> str:
    s = (ROOT / ("README.md" if lang == "en" else "README.zh.md")).read_text()
    if lang == "en":
        s = s.replace("# trutina\n\nEnglish | [中文](README.zh.md)",
                      "# trutina-4b\n\nEnglish | [中文](https://huggingface.co/aldinelabs/trutina-4b/blob/main/README.zh.md)", 1)
        s = s.replace("Weights: [aldinelabs/trutina-4b](https://huggingface.co/aldinelabs/trutina-4b)",
                      "Code: [github.com/aldinelabs/trutina](https://github.com/aldinelabs/trutina)", 1)
    else:
        s = s.replace("# trutina\n\n[English](README.md) | 中文",
                      "# trutina-4b\n\n[English](https://huggingface.co/aldinelabs/trutina-4b/blob/main/README.md) | 中文", 1)
        s = s.replace("权重：[aldinelabs/trutina-4b](https://huggingface.co/aldinelabs/trutina-4b)",
                      "代码：[github.com/aldinelabs/trutina](https://github.com/aldinelabs/trutina)", 1)
    s = re.sub(r"\]\((assets/[^)]+)\)", lambda m: f"]({RAW}{m.group(1)})", s)
    start, end, code = CODE[lang]
    a, b = s.index(start), s.index(end)
    return FRONT + s[:a] + code + s[b:]


if __name__ == "__main__":
    out = Path(sys.argv[1])
    (out / "README.md").write_text(card("en"))
    (out / "README.zh.md").write_text(card("zh"))
    print(out / "README.md", out / "README.zh.md")
