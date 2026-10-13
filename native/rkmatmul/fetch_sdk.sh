#!/usr/bin/env bash
set -euo pipefail
# Public Rockchip RKNN SDK, pinned independently of the RKLLM SDK.
revision=2c723177a94fac65cbc036d722907ddbc91980cf
out="${1:-/tmp/trutina-rknn2}"
base="https://raw.githubusercontent.com/airockchip/rknn-toolkit2/$revision/rknpu2/runtime/Linux/librknn_api"
mkdir -p "$out/include"
for name in rknn_api.h rknn_matmul_api.h; do
  curl --fail --location --retry 3 --connect-timeout 20 "$base/include/$name" -o "$out/include/$name"
done
curl --fail --location --retry 3 --connect-timeout 20 "$base/aarch64/librknnrt.so" -o "$out/librknnrt.so"
printf '%s\n' "$revision" > "$out/revision.txt"
