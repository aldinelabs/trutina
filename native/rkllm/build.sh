#!/usr/bin/env bash
set -euo pipefail
# Usage: bash native/rkllm/build.sh /path/to/rknn-llm /path/to/output
sdk="$(cd "${1:?official rknn-llm checkout required}" && pwd)"
out="${2:-build/rkllm}"
mkdir -p "$out"
runtime="$sdk/rkllm-runtime/Linux/librkllm_api"
printf '%s  %s\n' \
    4ad19442de58df1b7d1ed0884ed75b8d4b5107cc4e75b7e44900b1f95a32c57c "$runtime/include/rkllm.h" \
    f25e9b099db08aaacd0a3ac62b4697d3951d6ae61ae41ea09f6702cfa89eb32c "$runtime/aarch64/librkllmrt.so" \
    | sha256sum --check --status
cp "$runtime/aarch64/librkllmrt.so" "$out/"
"${CXX:-g++}" -std=c++17 -O2 -Wall -Wextra -Werror -fPIC -shared -pthread \
    -I"$runtime/include" "$(dirname "$0")/bridge.cpp" \
    -L"$out" -lrkllmrt -Wl,-rpath,'$ORIGIN' -o "$out/libtrutina_rkllm.so"
