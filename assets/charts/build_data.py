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
SYSTEMS = [  # fixed order: Trutina first, then the others by JevBench Intelligence on the same tasks
    ("trutina", "Trutina"), ("jevk5", "JevK5 v0.3"), ("plumb", "Plumb-4B"), ("spark", "spark-s1"),
    ("decider", "decider-4b"), ("sieve", "Sieve-4B")]


def compete() -> dict:
    """Trutina and the open 4B decision models through their own servers (lab/modal/compete_modal.py):
    the full run for accuracy, the JevBench re-run (-jb) for latency and v1.5 Intelligence (same run for all)."""
    def load(tag):
        return {s: json.loads((COMPETE / f"{s}{tag}.json").read_text()) for s, _ in SYSTEMS if (COMPETE / f"{s}{tag}.json").exists()}

    full, jb = load(""), load("-jb")
    intel = json.loads((COMPETE / "intelligence-jb.json").read_text()) if (COMPETE / "intelligence-jb.json").exists() else {}

    def vals(key, field="acc"):
        return {s: (round(full[s]["sets"][key][field] * 100, 1) if s in full and key in full[s]["sets"] else None) for s, _ in SYSTEMS}

    refused = {s: {k: n for k, n in (full.get(s, {}).get("error_counts") or {}).items()} for s, _ in SYSTEMS}

    def panel(pid, en, zh, m_en, m_zh, v, **kw):
        return {"id": pid, "en": en, "zh": zh, "m_en": m_en, "m_zh": m_zh, "vals": v, **kw}

    panels = [
        panel("intel", "JevBench Intelligence", "JevBench Intelligence", "v1.5 scoring · 231 public tasks", "v1.5 计分 · 231 道公开题",
              {s: intel.get(s, {}).get("I_open") for s, _ in SYSTEMS}),
        panel("jevbench", "JevBench public", "JevBench 公开集", "231 tasks · accuracy", "231 题 · 准确率", vals("jevbench")),
        panel("localllama", "LocalLLaMA", "LocalLLaMA", "typed-decisions test · accuracy", "typed-decisions 测试集 · 准确率", vals("localllama")),
        panel("dbm", "DecisionBench medium", "DecisionBench medium", "state-macro accuracy", "按 state 宏平均", vals("decisionbench/medium", "state_macro_acc")),
        panel("dbh", "DecisionBench hard", "DecisionBench hard", "state-macro accuracy", "按 state 宏平均", vals("decisionbench/hard", "state_macro_acc")),
        panel("thisthat", "this-that", "this-that", "complex decisions · accuracy", "复杂决策 · 准确率", vals("this_that")),
        panel("fast", "fast-decisions", "fast-decisions", "17 single-label domains", "17 个单标签领域 · 准确率", vals("fast_decisions")),
        panel("judge", "JudgeBench", "JudgeBench", "pair accuracy, both orders", "两种顺序均正确", vals("judgebench", "pair_acc")),
        panel("massive", "MASSIVE intents", "MASSIVE 意图分类", "59 options per question", "每题 59 个选项 · 准确率", vals("massive"),
              refused=[s for s, _ in SYSTEMS if refused.get(s, {}).get("massive", 0) >= 0.5 * 2974]),
        panel("di", "Decision Index 0.3", "Decision Index 0.3", "public suite · reported, not re-run", "公开套件 · 报告值，未统一复跑",
              {"trutina": 44.2, "sieve": 50.1}, est={"trutina": True}),
        panel("latency", "Latency", "延迟", "p50 per request · lower is better", "单请求 p50 · 越低越好",
              {s: (round(jb[s]["latency_jevbench_ms"]["p50"], 1) if s in jb else None) for s, _ in SYSTEMS}, unit="ms", lower=True),
    ]
    for pn in panels:
        pn.setdefault("refused", [])
        if pn["id"] != "massive":
            pn["refused"] = []
    gpu = next((r.get("gpu") for r in full.values()), "")
    # one JevBench re-run for every system: latency and v1.5 Intelligence on the same 231 tasks
    runs = []
    for s, n in SYSTEMS:
        if s not in jb:
            continue
        for mode, lat_key, ikey in (("full", "latency_jevbench_ms", s), ("adaptive", "latency_adaptive_ms", s + "-adaptive")):
            lat = jb[s].get(lat_key)
            if not lat:
                continue
            runs.append({"id": s, "name": n, "mode": mode if s == "trutina" else None, "us": s == "trutina",
                         "p50": round(lat["p50"], 1), "p95": round(lat["p95"], 1), "mean": round(lat["mean"], 1),
                         "i_open": intel.get(ikey, {}).get("I_open")})
    v3 = COMPETE / "trutina-v3.json"
    if v3.exists():  # Trutina rows: v3 with the decisive yes/no map, served with CUDA graphs
        t = json.loads(v3.read_text())
        for r in runs:
            if r["us"]:
                lat = t["latency"][r["mode"]]
                r.update(p50=round(lat["p50"], 1), p95=round(lat["p95"], 1), mean=round(lat["mean"], 1),
                         i_open=t["intelligence"][r["mode"]], v3=True)
    return {"systems": [{"id": s, "short": n} for s, n in SYSTEMS], "panels": panels, "gpu": gpu, "runs": runs,
            "errors": {s: r.get("error_counts", {}) for s, r in full.items()}}


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
            # v3: per-exit accuracy from the calibration report (val) and the saved per-exit LocalLLaMA
            # probabilities with the release temperatures; JevBench full / adaptive from the server run (lab exp11)
            "exits": {"layers": [12, 16, 20, 24, 32],
                      "val": [47.9, 73.0, 87.2, 88.5, 89.4],
                      "localllama": [42.5, 60.9, 72.15, 73.85, 74.65],
                      "adaptive": {"val": 89.1, "jevbench": 86.6, "localllama": 74.65},
                      "full": {"jevbench": 87.4},
                      "share_val": {"12": 3.6, "16": 26.8, "20": 33.2, "24": 10.9, "32": 25.6},
                      "mean_layers": {"val": 22.1, "jevbench": 21.6, "localllama": 27.9}},
            "heldout": [
                {"set": "JevBench public", "metric": "accuracy", "zero_shot": 79.7, "trutina": 87.4},
                {"set": "DecisionBench medium", "metric": "state-macro acc.", "zero_shot": 75.2, "trutina": 83.3},
                {"set": "JudgeBench", "metric": "pair accuracy", "zero_shot": 46.0, "trutina": 54.5},
                {"set": "this-that", "metric": "accuracy", "zero_shot": 43.8, "trutina": 51.3},
                {"set": "fast-decisions", "metric": "accuracy", "zero_shot": 61.5, "trutina": 61.1},
                {"set": "DecisionBench hard", "metric": "state-macro acc.", "zero_shot": 53.6, "trutina": 52.5}],
            "massive": {"tournament": 79.7, "two_letter_labels": 59.4},
            # image questions, our validation split (in-distribution): v2 (text-only fine-tune) vs the release
            "vision": [{"set": "ScienceQA", "before": 91.6, "after": 98.5}, {"set": "Mind2Web", "before": 75.0, "after": 93.0},
                       {"set": "A-OKVQA", "before": 87.4, "after": 90.2},
                       {"set": "Rendered workflow screenshots", "before": 84.6, "after": 84.9}],
        },
        "localllama": {"source": "LocalLLaMA/typed-decisions dataset card, zero-shot table", "rows": [
            {"name": "meraGPT Decider 1", "acc": 76.8, "kl": 0.096, "open": False},
            {"name": "Trutina", "acc": 74.85, "kl": 0.149, "open": True, "ours": True},
            {"name": "Liquid AI d1", "acc": 74.2, "kl": 0.475, "open": False},
            {"name": "TypeSafe Jev 1.13.0", "acc": 72.7, "kl": 1.442, "open": False},
            {"name": "Featherless Simple Jev", "acc": 71.6, "kl": 0.488, "open": False},
            {"name": "jev-vision-27b", "acc": 71.4, "kl": 0.318, "open": True},
            {"name": "prima-ratio + 12B", "acc": 70.2, "kl": 0.564, "open": True},
            {"name": "OpenDecider-small", "acc": 67.1, "kl": 0.211, "open": True},
            {"name": "Bongard-mini", "acc": 59.4, "kl": 0.256, "open": True},
            {"name": "Jeff-Gemma4-E2B", "acc": 56.1, "kl": 0.403, "open": True},
            {"name": "Jeff-Qwen3.5-2B", "acc": 51.1, "kl": 0.460, "open": True},
            {"name": "Jeff-Qwen3.5-0.8B", "acc": 48.3, "kl": 0.679, "open": True}]},
    }
    # typed-decisions (400 cases x 5 questions): Celeris's provider-reported table (harness 0cfd276, via
    # benchlm.ai/decision-models) and Trutina measured by us on the same 2,000-question test (released weights,
    # full depth; Brier = sum over options of (p - soft target)^2, per question)
    data["typed"] = {"source": "Celeris-1 Decision launch table (provider-reported), benchlm.ai/decision-models", "rows": [
        {"name": "Celeris-1 Decision", "size": "closed", "acc": 76.5, "brier": 0.079},
        {"name": "Trutina", "size": "4B", "acc": 74.65, "brier": 0.082, "ours": True},  # v3, served temperature
        {"name": "Decider v1.1", "size": "27B", "acc": 73.8, "brier": 0.173},
        {"name": "Jev 1.13.0", "size": "closed", "acc": 73.6, "brier": 0.149},
        {"name": "GPT-6 Luna", "size": "closed", "acc": 71.8, "brier": 0.204},
        {"name": "Mercury Decide", "size": "closed", "acc": 71.5, "brier": 0.270}],
        # Brier of a uniform guess on the same 2,000 questions (600 two-way, 1,100 four-way, 300 five-way)
        "brier_uniform": 0.2382}
    for r in data["typed"]["rows"]:
        r["cal"] = round(100 * (1 - r["brier"] / data["typed"]["brier_uniform"]), 1)  # Brier skill vs uniform
        r["index"] = round((r["acc"] + max(r["cal"], 0)) / 2, 1)
    data["typed"]["rows"].sort(key=lambda r: -r["index"])
    # Perplexity's 11-task decision panel without Circa (trained on): published numbers for Jev 1.13.0,
    # Qwen3.8-27B and Perplexity Decider 27B (benchlm.ai/decision-models), Trutina from lab/modal/panel_modal.py
    pp = HERE.parents[1] / "runs" / "release" / "perplexity-panel-scores.json"
    if pp.exists():
        pz = json.loads(pp.read_text())
        names = {"winogrande": "WinoGrande", "financial_phrasebank": "FinancialPhraseBank", "ragtruth": "RAGTruth",
                 "judgebench": "JudgeBench", "bbh": "BBH", "jevbench_hard": "JevBench public hard", "tabfact": "TabFact",
                 "contractnli": "ContractNLI", "belebele": "Belebele", "truthfulqa": "TruthfulQA binary"}
        tasks = [{"id": t, "name": names[t], "rows": pz["rows"][t], "trutina": round(pz["ours_full_depth"][t], 1),
                  "jev": pz["published"][t][0], "qwen": pz["published"][t][1], "pplx": pz["published"][t][2]} for t in names]
        n = sum(t["rows"] for t in tasks)
        overall = {k: round(sum(t[k] * t["rows"] for t in tasks) / n, 1) for k in ("trutina", "jev", "qwen", "pplx")}
        data["panel"] = {"tasks": sorted(tasks, key=lambda t: t["qwen"] - t["trutina"]), "overall": overall, "n": n}
    data["compete"] = compete()
    cv = HERE / "cover" / "examples.json"
    if cv.exists():  # the release cover's use cases, answered by the released weights (cover/run_examples.py)
        data["cover"] = json.loads(cv.read_text())
    # larger systems on the same GPU type: JevBench's official raw latency (maintainers' RTX PRO 6000 pods,
    # 450 requests, serial); Trutina from our run (231 public tasks, same GPU type, serial HTTP)
    BIG = {"Cygnet": ("Cygnet", "12B"), "AutoJev-27B (RTX PRO 6000)": ("AutoJev-27B", "27B"),
           "Surogate Rune 26B-A4B v3 (RTX PRO 6000)": ("Surogate Rune 26B-A4B", "26B"),
           "JevOne": ("JevOne · Qwen3.6-35B-A3B", "35B"), "NInfer Qwen3.8-Flash-Next mixed": ("NInfer Qwen3.8-Flash-Next", "?"),
           "swanOne": ("swanOne · Qwen3.8-Flash-Next", "?"), "CLM-8B": ("CLM-8B", "8B")}
    big = []
    for r in jb["systems"]:
        if (r.get("gpu") or "").replace(" ", "") != "RTXPRO6000" or not (r.get("speed") or {}).get("p50_s_raw"):
            continue
        key = next((k for k in BIG if r["display"].startswith(k)), None)
        if key:
            sp = r["speed"]
            big.append({"name": BIG[key][0], "size": BIG[key][1], "p50": round(sp["p50_s_raw"] * 1000, 1),
                        "p95": round(sp["p95_s_raw"] * 1000, 1), "n": sp.get("n"), "by": "official",
                        "i_open": round(r["intelligence"]["I_open"], 1)})
    for r in data["compete"]["runs"]:
        if r["us"]:
            big.append({"name": "Trutina", "mode": r["mode"], "size": "4B", "p50": r["p50"], "p95": r["p95"], "us": True, "by": "us"})
        else:  # the other open 4B models, through their own servers in the same run
            big.append({"name": r["name"], "size": "4B", "p50": r["p50"], "p95": r["p95"], "by": "us"})
    data["bigger"] = sorted(big, key=lambda r: r["p50"])
    # the whole official board at p50 (raw) against Intelligence, with Trutina placed by estimate: it has no official
    # score, so its own-harness Intelligence is shifted by the official-minus-ours difference on the systems run both ways
    LLM = ("GPT-", "Gemini", "DeepSeek", "Qwen3.8 27B (Chutes")
    board = []
    for r in jb["systems"]:
        sp, i = r.get("speed") or {}, (r.get("intelligence") or {}).get("I_open")
        if sp.get("p50_s_raw") and i is not None and i >= 40:
            board.append({"name": r["display"].split(" (")[0], "p50": round(sp["p50_s_raw"] * 1000, 1), "i": round(i, 1),
                          "gpu": r.get("gpu"), "llm": r["display"].startswith(LLM)})
    pairs = {"jevk5": "JevK5 v0.3", "plumb": "Plumb-4B", "spark": "spark-s1-4b-v6"}  # same weights in both runs
    ours = {r["id"]: r["i_open"] for r in data["compete"]["runs"] if not r["us"]}
    offs = {k: round(next(b["i"] for b in board if b["name"] == n) - ours[k], 1) for k, n in pairs.items()}
    lo, hi, mid = min(offs.values()), max(offs.values()), sum(offs.values()) / len(offs)
    for r in data["compete"]["runs"]:
        if r["us"]:
            board.append({"name": "Trutina", "mode": r["mode"], "p50": r["p50"], "i": round(r["i_open"] + mid, 1),
                          "i_lo": round(r["i_open"] + lo, 1), "i_hi": round(r["i_open"] + hi, 1), "i_ours": r["i_open"], "us": True})
    data["board"] = {"rows": board, "offsets": offs, "offset_mean": round(mid, 1)}
    # LocalLLaMA: the dataset card's zero-shot rows, plus the six 4B systems we ran (same 5-questions-per-request protocol)
    card = [dict(r, by="card") for r in data["localllama"]["rows"] if not r.get("ours")]
    names = dict(SYSTEMS)
    for s, n in SYSTEMS:
        f = COMPETE / f"{s}.json"
        if f.exists():
            r = json.loads(f.read_text())["sets"]["localllama"]
            row = {"name": n, "acc": round(r["acc"] * 100, 1), "kl": round(r["kl"], 3), "open": True, "ours": s == "trutina", "by": "us"}
            if s == "trutina":  # v3 at full depth on the same 2,000 questions (lab exp11), served temperature
                row.update(acc=74.65, kl=0.156)
            card.append(row)
    data["localllama"]["rows"] = sorted(card, key=lambda r: -r["acc"])
    (HERE / "data.json").write_text(json.dumps(data, indent=1))
    (HERE / "data.js").write_text("window.DATA = " + json.dumps(data) + ";\n")  # for file:// pages
    print(len(rows), "JevBench rows")


if __name__ == "__main__":
    main()
