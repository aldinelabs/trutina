"""Collect the numbers the charts show into data.json (run once per data refresh).

  python assets/charts/build_data.py path/to/jevbench-v1.5.4-results.json   # writes data.json and data.js

JevBench rows come from the official v1.5.4 results (benchmarkheaven.com/api/jevbench/v1.5.4):
open-set Intelligence (I_open) and raw p50 / p95 latency measured by the maintainers on their own GPU
pods, one decision per request. LocalLLaMA rows are from the dataset card. trutina rows are our own
measurements (see README "Evaluation").
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIZE = {  # parameters (B) where the system names them; None = not disclosed
    "Cygnet": 12, "Winnow-12B": 12, "Jev-Omni": 12, "AutoJev-27B": 27, "Eikos-27B": 27, "reflex-27b": 27,
    "NInfer Qwen3.8-27B": 27, "Surogate Rune 26B-A4B": 26, "decider-35b-a3b": 35, "djev (Maisa": 26,
    "Qwen3.5-9B": 9, "Bespoke Nimble 9B": 9, "jev-local (Qwen3.5-9B)": 9, "jqv (Qwen3-32B": 32,
    "4B": 4, "-4b": 4, "4B ": 4, "decider-2b": 2, "2B": 2,
}


def size_of(name: str):
    for k, v in SIZE.items():
        if k in name:
            return v
    return None


def short(name: str) -> str:
    base = name.split(" (")[0].replace("Raw Qwen3 4B Instruct 2507 direct logits", "Qwen3-4B-Instruct (raw logits)")
    for tag, label in (("thinking", "thinking"), ("low reasoning", "low effort"), ("medium reasoning", "medium effort"),
                       ("T=1.5", "T=1.5"), ("RTX PRO 6000", "RTX PRO 6000")):
        if tag in name and label not in base:
            base += f" ({label})"
    return base


COMPETE = HERE.parents[1] / "runs" / "release" / "compete"
SYSTEMS = [  # fixed order: trutina first, then the others by JevBench open-set Intelligence
    ("trutina", "Trutina"), ("spark", "spark-s1-4b-v6"), ("decider", "decider-4b"), ("jevk5", "JevK5 v0.3"),
    ("plumb", "Plumb-4B"), ("sieve", "Sieve-4B")]
JB_NAME = {"spark": "spark-s1-4b-v6", "decider": "decider-4b v2", "jevk5": "JevK5 v0.3 (4B)", "plumb": "Plumb-4B"}


def compete() -> dict:
    """trutina-4b and the open 4B decision models, every benchmark we ran through each system's own server
    (lab/modal/compete_modal.py), plus the Decision Index where only reported numbers exist."""
    res = {s: json.loads((COMPETE / f"{s}.json").read_text()) for s, _ in SYSTEMS if (COMPETE / f"{s}.json").exists()}

    def val(s, key, field="acc", scale=100):
        r = res.get(s, {}).get("sets", {}).get(key)
        return None if r is None else round(r[field] * scale, 1)

    def panel(pid, en, zh, m_en, m_zh, key, field="acc", **kw):
        return {"id": pid, "en": en, "zh": zh, "m_en": m_en, "m_zh": m_zh,
                "vals": {s: val(s, key, field) for s, _ in SYSTEMS}, **kw}

    panels = [
        panel("jevbench", "JevBench public", "JevBench 公开集", "231 tasks · accuracy", "231 题 · 准确率", "jevbench"),
        panel("localllama", "LocalLLaMA", "LocalLLaMA", "typed-decisions test · accuracy", "typed-decisions 测试集 · 准确率", "localllama"),
        panel("dbm", "DecisionBench medium", "DecisionBench medium", "state-macro accuracy", "按 state 宏平均的准确率",
              "decisionbench/medium", "state_macro_acc"),
        panel("dbh", "DecisionBench hard", "DecisionBench hard", "state-macro accuracy", "按 state 宏平均的准确率",
              "decisionbench/hard", "state_macro_acc"),
        panel("thisthat", "this-that", "this-that", "complex decisions · accuracy", "复杂决策 · 准确率", "this_that"),
        panel("fast", "fast-decisions", "fast-decisions", "17 single-label domains · accuracy", "17 个单标签领域 · 准确率", "fast_decisions"),
        panel("judge", "JudgeBench", "JudgeBench", "pair accuracy, both orders", "两种选项顺序均正确方计为正确", "judgebench", "pair_acc"),
        panel("massive", "MASSIVE intents", "MASSIVE 意图分类", "59 options per question · accuracy", "每题 59 个选项 · 准确率", "massive"),
    ]
    # Decision Index 0.3, public suite: no system here has been run by us on the full suite.
    panels.append({"id": "di", "en": "Decision Index 0.3", "zh": "Decision Index 0.3",
                   "m_en": "public suite · trutina estimated, Sieve self-reported", "m_zh": "公开套件 · trutina 为抽样估计，Sieve 为自报",
                   "vals": {"trutina": 44.2, "sieve": 50.1, "spark": None, "decider": None, "jevk5": None, "plumb": None}})
    panels.append({"id": "latency", "en": "Latency", "zh": "延迟", "unit": "ms", "lower": True,
                   "m_en": "p50 per request, same GPU · lower is better", "m_zh": "单请求 p50，同一 GPU · 越低越好",
                   "vals": {s: (round(res[s]["latency_jevbench_ms"]["p50"], 1) if s in res and res[s].get("latency_jevbench_ms") else None)
                            for s, _ in SYSTEMS}})
    gpu = next((r["gpu"] for r in res.values()), "")
    return {"systems": [{"id": s, "short": n} for s, n in SYSTEMS], "panels": panels, "gpu": gpu,
            "errors": {s: r.get("error_counts", {}) for s, r in res.items()}}


def main():
    jb = json.loads(Path(sys.argv[1]).read_text())
    rows = []
    for r in jb["systems"]:
        i, sp = r.get("intelligence") or {}, r.get("speed") or {}
        if not i.get("I_open") or sp.get("p50_s_raw") is None:
            continue
        gpu = r.get("gpu") or r.get("endpoint_kind")
        rows.append({"name": short(r["display"]), "display": r["display"], "i_open": round(i["I_open"], 1),
                     "i_sealed": round(i.get("I_sealed") or 0, 1), "composite": round(r.get("jevbench_score") or 0, 1),
                     "p50_ms": round(sp["p50_s_raw"] * 1000, 1), "p95_ms": round(sp["p95_s_raw"] * 1000, 1),
                     "hw": {"RTXPRO6000": "RTX PRO 6000", "RTX5090": "RTX 5090", "RTX6000": "RTX 6000"}.get(gpu, gpu),
                     "kind": "api" if gpu in ("api", "demo") else "cpu" if gpu == "cpu" else "gpu",
                     "params_b": size_of(r["display"])})
    data = {
        "jevbench": {"source": "JevBench v1.5.4 official results (benchmarkheaven.com), retrieved 2026-10-12",
                     "rows": rows},
        "trutina": {
            "jevbench_public_acc": {"full": 87.0, "adaptive": 86.1, "untrained": 79.7},
            "i_open": 69.0,
            "latency_rtxpro6000": {"full": {"p50": 28.3, "p95": 112.1, "mean": 99.5},
                                   "adaptive": {"p50": 21.1, "p95": 107.4, "mean": 35.3}},
            "latency_m5pro": {"jevbench": {"full": 113, "adaptive": 71}, "localllama": {"full": 201, "adaptive": 177}},
            "exits": {"layers": [12, 16, 20, 24, 32],
                      "val": [50.0, 73.5, 85.8, 87.4, 87.9], "jevbench": [42.9, 71.9, 87.0, 87.9, 87.0],
                      "localllama": [43.5, 59.3, 72.8, 75.2, 74.75],
                      "adaptive": {"val": 87.5, "jevbench": 86.1, "localllama": 74.75},
                      "share_val": {"12": 6.8, "16": 28.1, "20": 26.2, "24": 9.5, "32": 29.4},
                      "mean_layers": {"val": 22.2, "jevbench": 22.1, "localllama": 28.2}},
            "heldout": [
                {"set": "JevBench public", "metric": "accuracy", "zero_shot": 79.7, "trutina": 87.0},
                {"set": "DecisionBench medium", "metric": "state-macro acc.", "zero_shot": 75.2, "trutina": 83.2},
                {"set": "JudgeBench", "metric": "pair accuracy", "zero_shot": 46.0, "trutina": 52.1},
                {"set": "this-that", "metric": "accuracy", "zero_shot": 43.8, "trutina": 51.3},
                {"set": "fast-decisions", "metric": "accuracy", "zero_shot": 61.5, "trutina": 61.4},
                {"set": "DecisionBench hard", "metric": "state-macro acc.", "zero_shot": 53.6, "trutina": 49.6}],
            "massive": {"tournament": 79.3, "two_letter_labels": 59.4},
            # image questions, our validation split (in-distribution): v2 (text-only fine-tune) vs the release
            "vision": [{"set": "ScienceQA", "before": 91.6, "after": 96.2}, {"set": "Mind2Web", "before": 75.0, "after": 91.9},
                       {"set": "A-OKVQA", "before": 87.4, "after": 89.6},
                       {"set": "Rendered workflow screenshots", "before": 84.6, "after": 84.9}],
        },
        "localllama": {"source": "LocalLLaMA/typed-decisions dataset card, zero-shot table", "rows": [
            {"name": "meraGPT Decider 1", "acc": 76.8, "kl": 0.096, "open": False},
            {"name": "Trutina", "acc": 74.85, "kl": 0.149, "open": True, "ours": True},
            {"name": "Liquid AI d1", "acc": 74.2, "kl": 0.475, "open": False},
            {"name": "TypeSafe Jev 1.13.0", "acc": 72.7, "kl": 1.442, "open": False},
            {"name": "Featherless Simple Jev", "acc": 71.6, "kl": 0.488, "open": False},
            {"name": "prima-ratio + 12B", "acc": 70.2, "kl": 0.564, "open": True},
            {"name": "OpenDecider-small", "acc": 67.1, "kl": 0.211, "open": True},
            {"name": "Bongard-mini", "acc": 59.4, "kl": 0.256, "open": True},
            {"name": "Jeff-Gemma4-E2B", "acc": 56.1, "kl": 0.403, "open": True}]},
    }
    data["compete"] = compete()
    (HERE / "data.json").write_text(json.dumps(data, indent=1))
    (HERE / "data.js").write_text("window.DATA = " + json.dumps(data) + ";\n")  # for file:// pages
    print(len(rows), "JevBench rows")


if __name__ == "__main__":
    main()
