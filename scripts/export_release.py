"""Export a trained readout run as a self-contained release directory (the Hugging Face model repo).

The LoRA is merged into the base checkpoint's text weights; every other tensor (vision tower, norms,
embeddings, MTP) is copied unchanged, so the result keeps the base model's layout and loads with
mlx-lm or transformers like Qwen3.5-4B. The readout parameters (tuned-lens exits, logit scales) go to
`readout.safetensors`, and the prompt format and calibration to `trutina.json`.

  uv run python scripts/export_release.py --run runs/private/v2.1b-step4000 --out release/trutina-4b
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download

from trutina.readout import FORMAT_VERSION

TEXT = "model.language_model."


def merged_weights(base: dict[str, mx.array], trained: dict[str, mx.array], scale: float) -> tuple[dict, int]:
    """W' = W + scale * (A @ B)^T for every LoRA pair whose weight is in `base` (one shard), computed in
    float32 and stored in W's dtype."""
    out, n = dict(base), 0
    for k, a in trained.items():
        if not k.endswith(".lora_a"):
            continue
        stem = k[len("torso."): -len(".lora_a")]
        key = f"{TEXT}{stem}.weight"
        if key not in base:
            continue
        w = base[key]
        delta = (a @ trained[k[:-1] + "b"]).T * scale
        if delta.shape != w.shape:
            raise ValueError(f"{key}: delta {delta.shape} vs weight {w.shape}")
        out[key] = (w.astype(mx.float32) + delta).astype(w.dtype)
        n += 1
    return out, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training run: config.json, calibration.json, weights")
    ap.add_argument("--weights", default="best.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="trutina-4b")
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    conf = json.loads((run / "config.json").read_text())
    tc = conf["train"]
    cal = json.loads((run / "calibration.json").read_text())
    base_dir = Path(snapshot_download(conf["model"]))
    out.mkdir(parents=True, exist_ok=True)

    trained = mx.load(str(run / a.weights))
    index = json.loads((base_dir / "model.safetensors.index.json").read_text())
    merged_total = 0
    for shard in sorted(set(index["weight_map"].values())):
        base = mx.load(str(base_dir / shard))
        merged, n = merged_weights(base, trained, tc["lora_scale"])
        merged_total += n
        mx.save_safetensors(str(out / shard), merged, metadata={"format": "pt"})
    n_lora = sum(k.endswith(".lora_a") for k in trained)
    if merged_total != n_lora:
        raise ValueError(f"merged {merged_total} of {n_lora} LoRA pairs")
    for f in base_dir.iterdir():  # config, tokenizer, chat template, processors, index
        if f.suffix != ".safetensors" and f.name not in ("README.md", ".gitattributes", "LICENSE"):
            shutil.copy(f, out / f.name)
    shutil.copy(base_dir / "LICENSE", out / "LICENSE.qwen")

    readout = {k: v for k, v in trained.items() if k.startswith("lens") or k == "prior_logscale"}
    mx.save_safetensors(str(out / "readout.safetensors"), readout, metadata={"format": "pt"})
    exits = [*tc.get("lens_exits", []), 32]
    (out / "trutina.json").write_text(json.dumps({
        "format_version": FORMAT_VERSION,
        "name": a.name,
        "base_model": conf["model"],
        "readout": {"labels": "letter", "order": "sandwich", "chat_template": True, "thinking": False,
                    "answer_prefix": "Best answer: (", "max_options_per_pass": 26, "max_options": 255,
                    "wide": "tournament"},
        "exits": exits,
        "temperature": {str(e): cal["temperature"][str(e)] for e in exits},
        "adaptive_threshold": cal["adaptive_threshold"],
        "noul_decisive": {"T": cal["noul_decisive"]["T"], "b": cal["noul_decisive"]["b"]},
        "lora": {"rank": tc["rank"], "scale": tc["lora_scale"], "merged_pairs": merged_total},
    }, indent=1) + "\n")
    print(f"{out}: {merged_total} LoRA pairs merged; readout {len(readout)} tensors")


if __name__ == "__main__":
    main()
