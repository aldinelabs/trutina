#!/usr/bin/env bash
set -euo pipefail
# SDK directory contains include/{rknn_api,rknn_matmul_api}.h and librknnrt.so.
sdk="$(cd "${1:?official RKNN SDK directory required}" && pwd)"
out="${2:-build/rkmatmul}"
mkdir -p "$out"
printf '%s  %s\n' \
  2d4160cd868efd7b83a72587e411c16d6972da5876e067bd73665cea778ce666 "$sdk/include/rknn_api.h" \
  aaadd9a7118de30a06b222996b6731db77095d00f5931a7a98c83a67f14a4d42 "$sdk/include/rknn_matmul_api.h" \
  4b6cc575bedf8952b13db4b1190fa3b3d2ac213f00be774ff6ccd19d651970c0 "$sdk/librknnrt.so" \
  | sha256sum --check --status
cp "$sdk/librknnrt.so" "$out/librknnrt.so.new"
mv "$out/librknnrt.so.new" "$out/librknnrt.so"
"${CXX:-g++}" -std=c++17 -O2 -Wall -Wextra -Werror -fPIC -shared -pthread \
  -I"$sdk/include" "$(dirname "$0")/bridge.cpp" -L"$out" -lrknnrt \
  -Wl,-rpath,'$ORIGIN' -o "$out/libtrutina_matmul.so.new"
mv "$out/libtrutina_matmul.so.new" "$out/libtrutina_matmul.so"
