"""Install a user systemd service for a local Trutina checkout on Linux.

Run as the serving user, not root. All model/code paths are local to that user.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", choices=["torch", "rkllm", "rock5"], default="torch")
    ap.add_argument("--rkllm-model", help="rock5: converted NPU bundle")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--library", default="build/rkllm/libtrutina_rkllm.so")
    ap.add_argument("--npu-matmul", help="torch: experimental RK3588 Linear library")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    model = Path(a.model).resolve()
    if not model.is_dir():
        ap.error("--model must point to a local model directory")
    if not 1 <= a.port <= 65535:
        ap.error("invalid --port")
    if a.npu_matmul and (a.engine != "torch" or not Path(a.npu_matmul).is_file()):
        ap.error("--npu-matmul requires --engine torch and a built library")
    args = [sys.executable, "-u", str(root / "scripts/serve.py"), "--engine", a.engine,
            "--model", str(model), "--host", a.host, "--port", str(a.port)]
    if a.engine == "torch":
        args += ["--device", "cpu", "--threads", "4", "--batch-tokens", "512"]
        args += ["--npu-matmul", str(Path(a.npu_matmul).resolve())] if a.npu_matmul else ["--cpu-int8"]
        if a.vision:
            args += ["--vision"]
    else:
        args += ["--rkllm-library", str(Path(a.library).resolve())]
        if a.engine == "rock5":
            if not a.rkllm_model or not Path(a.rkllm_model).is_dir():
                ap.error("rock5 engine requires a local --rkllm-model directory")
            args += ["--rkllm-model", str(Path(a.rkllm_model).resolve())]
    def quoted(value):
        return shlex.quote(str(value).replace("%", "%%"))
    unit = f"""[Unit]
Description=Trutina local decision model ({a.engine})
After=network.target
StartLimitIntervalSec=300
StartLimitBurst=2

[Service]
Type=simple
WorkingDirectory={quoted(root)}
Environment=OMP_NUM_THREADS=4
Environment=OPENBLAS_NUM_THREADS=4
MemoryMax=13G
ExecStart={' '.join(quoted(arg) for arg in args)}
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
"""
    target = Path.home() / ".config/systemd/user/trutina.service"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.with_suffix(".service.bak").write_text(target.read_text())
    target.write_text(unit)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "trutina.service"], check=True)
    subprocess.run(["systemctl", "--user", "restart", "trutina.service"], check=True)
    print(f"Installed {target}. Startup includes loading/packing; check /health and service logs.")


if __name__ == "__main__":
    main()
