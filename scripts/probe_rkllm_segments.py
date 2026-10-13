"""Experimental RKLLM segmentation gate, using synthetic Qwen3.5 weights.

This is deliberately separate from the production engine. Successful conversion
does not establish that the runtime accepts missing output_norm or preserves raw
residuals. Run prepare/export in the x86 converter, run the native probe on RK3588,
then compare all tokens against CPU references before attempting real 4B weights.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

VARIANTS = {
    "full": (0, 8, False),
    "first-normalized": (0, 4, False),
    "first-raw": (0, 4, True),
    "second": (4, 8, False),
}
LENGTHS = (3, 17, 63)
HIDDEN = 512


def prepare(out: Path, tokenizer: Path):
    import shutil

    import torch
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    if out.exists():
        raise FileExistsError("Use a fresh probe directory")
    out.mkdir(parents=True)
    torch.manual_seed(7)
    torch.set_num_threads(4)
    cfg = Qwen3_5Config(text_config={
        "hidden_size": HIDDEN, "intermediate_size": 1024, "num_hidden_layers": 8,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 256,
        "vocab_size": 248320, "tie_word_embeddings": True,
        "linear_num_value_heads": 4, "linear_num_key_heads": 2,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "full_attention_interval": 4,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000.,
                            "partial_rotary_factor": 0.25, "mrope_interleaved": True,
                            "mrope_section": [11, 11, 10]},
    }, vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32,
                      "num_heads": 2, "out_hidden_size": HIDDEN, "num_position_embeddings": 64},
        tie_word_embeddings=True)
    model = Qwen3_5ForConditionalGeneration(cfg).float().eval()
    model.save_pretrained(out / "checkpoint")
    # HF 5.8 consumes full_attention_interval when constructing layer_types,
    # then omits it on save. RKLLM's graph converter also needs this field.
    config_path = out / "checkpoint" / "config.json"
    saved = json.loads(config_path.read_text())
    saved["text_config"].update(full_attention_interval=4, attn_output_gate=True,
                                mamba_ssm_dtype="float32")
    config_path.write_text(json.dumps(saved, indent=2) + "\n")
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                 "chat_template.jinja"):
        if (tokenizer / name).is_file():
            shutil.copy2(tokenizer / name, out / "checkpoint" / name)
    torso = model.model.language_model
    captured = {}

    def boundary(_layer, _args, result):
        captured["raw"] = result.detach().clone()

    hook = torso.layers[3].register_forward_hook(boundary)
    for length in LENGTHS:
        ids = (torch.arange(length)[None] * 7 + 4) % 100
        with torch.inference_mode():
            embeds = torso.embed_tokens(ids)
            final = torso(inputs_embeds=embeds, use_cache=False).last_hidden_state
            first_raw = captured["raw"]
            first_normalized = torso.norm(first_raw)
        for name, tensor in (("input", embeds), ("first-raw-cpu", first_raw),
                             ("first-normalized-cpu", first_normalized), ("full-cpu", final)):
            tensor.float().numpy().astype("<f4").tofile(out / f"{name}-{length}.f32")
    hook.remove()
    (out / "probe.json").write_text(json.dumps({
        "synthetic": True, "seed": 7, "hidden": HIDDEN, "lengths": LENGTHS,
        "context": 128, "variants": VARIANTS,
        "status": "CPU references prepared; NPU unverified",
    }, indent=2) + "\n")


def export_one(out: Path, variant: str, quantization: str, omit_embedding: bool = False):
    import pickle
    import re
    import shutil

    import torch
    from rkllm.api import RKLLM
    from safetensors import safe_open
    from safetensors.torch import save_file

    start, stop, omit_norm = VARIANTS[variant]
    source = out / "checkpoint"
    # The converter serializes some architecture metadata from config.json,
    # rather than exclusively from the mutable HF model. Slice the checkpoint
    # before loading, so the binary's layer count agrees with its tensors.
    if variant != "full":
        checkpoint = out / f"checkpoint-{variant}"
        checkpoint.mkdir()
        config = json.loads((source / "config.json").read_text())
        config["text_config"]["num_hidden_layers"] = stop - start
        config["text_config"]["layer_types"] = config["text_config"]["layer_types"][start:stop]
        (checkpoint / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                     "chat_template.jinja"):
            if (source / name).is_file():
                shutil.copy2(source / name, checkpoint / name)
        tensors = {}
        with safe_open(str(source / "model.safetensors"), framework="pt") as f:
            for name in f.keys():  # noqa: SIM118 (safe_open is not a dict)
                match = re.match(r"model\.language_model\.layers\.(\d+)\.", name)
                if match:
                    index = int(match[1])
                    if not start <= index < stop:
                        continue
                    key = name[:match.start(1)] + str(index-start) + name[match.end(1):]
                else:
                    key = name
                tensors[key] = f.get_tensor(name)
        save_file(tensors, str(checkpoint / "model.safetensors"), metadata={"format": "pt"})
        del tensors
        source = checkpoint
    llm = RKLLM()
    if llm.load_huggingface(str(source), device="cpu", dtype="float32"):
        raise RuntimeError("RKLLM load failed")
    llm.base.model.config.full_attention_interval = 4
    llm.base.model.config.attn_output_gate = True
    torso = llm.base.model.model
    # All boundaries align with the original four-layer hybrid-attention period.
    for index, layer in enumerate(torso.layers):
        layer.layer_idx = index
        for name in ("self_attn", "linear_attn"):
            if hasattr(layer, name):
                getattr(layer, name).layer_idx = index
    llm.base.model.config.num_hidden_layers = stop - start
    if omit_norm:
        torso.norm = torch.nn.Identity()
    dataset = out / f"calibration-{variant}.json"
    samples = []
    for length in LENGTHS:
        source = "first-raw-cpu" if start else "input"
        hidden = json.loads((out / "probe.json").read_text())["hidden"]
        embeds = np.fromfile(out / f"{source}-{length}.f32", dtype="<f4").reshape(1, length, hidden)
        sample = out / f"calibration-{variant}-{length}.pkl"
        # Official multimodal calibration format: trusted local pickles of HF
        # forward kwargs, referenced relative to the JSON dataset directory.
        with sample.open("wb") as f:
            pickle.dump({"inputs_embeds": torch.from_numpy(embeds),
                         "attention_mask": torch.ones(1, length, dtype=torch.long),
                         "use_cache": False}, f)
        samples.append({"sample": sample.name, "token_nums": length})
    dataset.write_text(json.dumps(samples))
    if llm.build(do_quantization=quantization == "w8a8", quantized_dtype="w8a8",
                 quantized_algorithm="normal", dataset=str(dataset),
                 target_platform="rk3588", max_context=128):
        raise RuntimeError("RKLLM build failed")
    target = out / f"{variant}.rkllm"
    if llm.export_rkllm(str(target), export_embedding=not omit_embedding) or not target.is_file() or not target.stat().st_size:
        raise RuntimeError("RKLLM export failed")
    print(f"TRUTINA_EXPORT_OK {variant}", flush=True)


def export(out: Path, quantization: str, omit_embedding: bool = False, variants=None):
    # Some SDK fatal errors terminate Python with exit(0). A return code alone
    # therefore cannot confirm successful conversion; require our final marker.
    results = {}
    for variant in variants or VARIANTS:
        target = out / f"{variant}.rkllm"
        if target.exists():
            raise FileExistsError(target)
        command = [sys.executable, "-u", __file__, "export-one",
                                 "--out", str(out), "--variant", variant,
                                 "--quantization", quantization]
        if omit_embedding:
            command.append("--omit-embedding")
        result = subprocess.run(command,
                                capture_output=True, text=True, check=False, timeout=600)
        (out / f"export-{variant}.log").write_text(result.stdout + result.stderr)
        ok = result.returncode == 0 and f"TRUTINA_EXPORT_OK {variant}" in result.stdout
        results[variant] = {"success": ok, "returncode": result.returncode,
                            "quantization": quantization,
                            "embedding_exported": not omit_embedding,
                            "bytes": target.stat().st_size if target.is_file() else 0}
        (out / "conversion.json").write_text(json.dumps(results, indent=2) + "\n")
        if not ok:
            raise RuntimeError(f"{variant}: see export-{variant}.log; conversion is unverified")
        print(f"Exported {variant}: {results[variant]['bytes']} bytes", flush=True)


def run(out: Path, binary: Path):
    """Run inside a memory-limited systemd scope on the board."""
    meta = json.loads((out / "probe.json").read_text())
    for length in meta["lengths"]:
        for variant, source, result in (
            ("full", "input", "full-npu"),
            ("first-normalized", "input", "first-normalized-npu"),
            ("first-raw", "input", "first-raw-npu"),
            ("second", "first-raw-cpu", "second-cpu-input-npu"),
            ("second", "first-raw-npu", "chained-npu"),
        ):
            destination = out / f"{result}-{length}.f32"
            if destination.exists():
                raise FileExistsError(destination)
            p = subprocess.run([str(binary.resolve()), str(out / f"{variant}.rkllm"),
                                str(out / f"{source}-{length}.f32"), str(destination),
                                str(length), str(meta["hidden"]), str(meta["context"])],
                               capture_output=True, text=True, check=False, timeout=60)
            (out / f"run-{result}-{length}.log").write_text(p.stdout + p.stderr)
            if p.returncode or "TRUTINA_HIDDEN_OK" not in p.stdout or not destination.is_file():
                raise RuntimeError(f"{result}/{length}: runtime probe failed; inspect log")
            print(f"Ran {result}/{length}", flush=True)


def compare(out: Path):
    meta = json.loads((out / "probe.json").read_text())
    measurements = []
    for length in meta["lengths"]:
        for name, ref in (("full-npu", "full-cpu"),
                          ("first-normalized-npu", "first-normalized-cpu"),
                          ("first-raw-npu", "first-raw-cpu"),
                          ("second-cpu-input-npu", "full-cpu"), ("chained-npu", "full-cpu")):
            expected = np.fromfile(out / f"{ref}-{length}.f32", dtype="<f4")
            actual = np.fromfile(out / f"{name}-{length}.f32", dtype="<f4")
            if expected.size != length * meta["hidden"] or actual.shape != expected.shape:
                raise ValueError("Full-sequence hidden-state shape mismatch")
            if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise ValueError("Non-finite hidden states")
            relative = float(np.linalg.norm(actual-expected) / max(np.linalg.norm(expected), 1e-12))
            measurements.append({"name": name, "tokens": length, "relative_l2": relative,
                                 "max_abs": float(np.max(np.abs(actual-expected))),
                                 "passed": relative < 0.03})
    passed = all(m["passed"] for m in measurements)
    (out / "comparison.json").write_text(json.dumps({
        "synthetic": True, "passed": passed, "tolerance_relative_l2": 0.03,
        "measurements": measurements,
        "meaning": "Structural gate only; release W8A8 accuracy/calibration still requires validation",
    }, indent=2) + "\n")
    if not passed:
        raise RuntimeError("Segment continuity gate failed; do not enable NPU adaptive")
    print("Synthetic NPU segmentation gate passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["prepare", "export", "export-one", "run", "compare"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path)
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--variants", choices=VARIANTS, nargs="+")
    ap.add_argument("--quantization", choices=["fp16", "w8a8"], default="w8a8")
    ap.add_argument("--omit-embedding", action="store_true")
    ap.add_argument("--binary", type=Path)
    args = ap.parse_args()
    if args.action == "prepare":
        if not args.tokenizer:
            ap.error("prepare requires --tokenizer")
        prepare(args.out, args.tokenizer)
    elif args.action == "export":
        export(args.out, args.quantization, args.omit_embedding, args.variants)
    elif args.action == "export-one":
        if not args.variant:
            ap.error("export-one requires --variant")
        export_one(args.out, args.variant, args.quantization, args.omit_embedding)
    elif args.action == "run":
        if not args.binary:
            ap.error("run requires --binary")
        run(args.out, args.binary)
    else:
        compare(args.out)


if __name__ == "__main__":
    main()
