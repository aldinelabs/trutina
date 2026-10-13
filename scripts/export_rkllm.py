"""Convert merged Trutina weights using the official Linux x86_64 RKLLM 1.3.1 toolkit.

Calibration is JSONL in the public request format. It is rendered with the exact
release tokenizer/prompt, so activation calibration sees real decision inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
from pathlib import Path

from trutina.api import parse_request
from trutina.readout import FORMAT_VERSION, full_text, prompt

SDK_COMMIT = "4f7390530443bf84f0394255a449d7cbe81e69d1c"


def calibration_inputs(path: Path, tok, max_context: int) -> list[dict]:
    import dataclasses

    from trutina import wide

    examples = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        req = parse_request(json.loads(line))
        for q in req.questions:
            for group in wide.groups(len(q.options)) if len(q.options) > wide.PASS else [None]:
                sub = dataclasses.replace(q, options=tuple(q.options[i] for i in group),
                                          descriptions=tuple(q.resolved_descriptions()[i] for i in group)) \
                    if group is not None else q
                text = full_text(tok, prompt(req.state, sub, "sandwich", "letter"), True, "letter")
                if len(tok.encode(text, add_special_tokens=False)) >= max_context:
                    raise ValueError("Calibration prompt exceeds exported context limit")
                examples.append({"input": text, "target": ""})
    if not examples:
        raise ValueError("Provide at least one calibration request")
    return examples


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="aldinelabs/trutina-4b")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calibration", required=True, help="text-only public requests, JSONL")
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--quantization", choices=["w8a8", "w4a16"], default="w8a8")
    ap.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    args = ap.parse_args()
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        ap.error("RKLLM toolkit requires Linux x86_64; use the documented Docker converter")
    if args.max_context < 2:
        ap.error("--max-context must be at least 2")
    from importlib.metadata import version

    if version("rkllm-toolkit") != "1.3.1":
        raise RuntimeError("Use the pinned RKLLM toolkit 1.3.1 for this runtime bundle")
    from rkllm.api import RKLLM
    from safetensors import safe_open
    from transformers import AutoTokenizer

    source = Path(args.model)
    revision = None
    if not source.is_dir():
        from huggingface_hub import snapshot_download

        source = Path(snapshot_download(args.model, revision=args.revision))
        revision = source.name
    config = json.loads((source / "trutina.json").read_text())
    if config["format_version"] != FORMAT_VERSION:
        raise ValueError("Unsupported Trutina release format")
    out = Path(args.out)
    if out.resolve() == source.resolve():
        raise ValueError("The export directory must differ from the source checkpoint")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "rkllm.json").exists() or (out / "model.rkllm").exists():
        raise FileExistsError("Use a new output directory to avoid overwriting an exported bundle")
    tok = AutoTokenizer.from_pretrained(source)
    samples = calibration_inputs(Path(args.calibration), tok, args.max_context)
    dataset = out / "calibration.json"
    dataset.write_text(json.dumps(samples, ensure_ascii=False))
    with safe_open(str(source / "readout.safetensors"), framework="np") as f:
        logscale = float(f.get_tensor("prior_logscale").item())
    scale = math.exp(logscale) / config["temperature"][str(config["exits"][-1])]
    llm = RKLLM()
    try:
        rc = llm.load_huggingface(model=str(source), device="cpu", dtype=args.dtype)
        if rc:
            raise RuntimeError(f"RKLLM load failed ({rc})")
        rc = llm.build(do_quantization=True, optimization_level=1,
                       quantized_dtype=args.quantization,
                       quantized_algorithm="normal" if args.quantization == "w8a8" else "grq",
                       target_platform="rk3588", num_npu_core=3, dataset=str(dataset),
                       max_context=args.max_context)
        if rc:
            raise RuntimeError(f"RKLLM build failed ({rc})")
        rc = llm.export_rkllm(str(out / "model.rkllm"))
        if rc:
            raise RuntimeError(f"RKLLM export failed ({rc})")
    finally:
        # Toolkit 1.3.1 has no release() method; it frees resources on destruction.
        del llm
    for name in ("trutina.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                 "merges.txt", "chat_template.jinja", "config.json", "LICENSE"):
        if (source / name).is_file():
            shutil.copy2(source / name, out / name)
    # Write the manifest last: its presence means conversion completed successfully.
    (out / "rkllm.json").write_text(json.dumps({
        "format_version": 1, "runtime_version": "1.3.1", "sdk_commit": SDK_COMMIT,
        "source_model": args.model, "source_revision": revision,
        "model_file": "model.rkllm", "quantization": args.quantization,
        "max_context": args.max_context, "final_logit_scale": scale,
        "calibration_samples": len(samples),
        "calibration_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "capabilities": {"text": True, "images": False, "intermediate_exits": False},
    }, indent=2) + "\n")
    print(f"Exported {out}; {len(samples)} calibration prompts", flush=True)


if __name__ == "__main__":
    main()
