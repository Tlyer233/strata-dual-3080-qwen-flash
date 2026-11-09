#!/bin/bash
# 3080 本机 Strata v0.2.12（engine 0.1.58）编译（sm_86，nvcc 走 ~/venvs/cuda-build 的 cu13）。
# 补丁只打三个：#1233 PLE + same-chat rewind（apply_local）+ sampling_force（apply_local+port_pin_022）。
# Codex 旁路不打（Responses 直通时代已退役）。只编译不动引擎。
set -uo pipefail
av=$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo); [ "${av:-0}" -ge 25 ] || { echo "MEM GUARD: available ${av}G < 25G, 不编"; exit 1; }; echo "MEM GUARD ok: ${av}G available"
S="$HOME/src/Strata-0.2.12"
P="$HOME/ubuntu_handoff/strata/patches-0.2.12"
CUDA="$HOME/venvs/cuda-build/lib/python3.14/site-packages/nvidia/cu13"
export LD_LIBRARY_PATH="$CUDA/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$CUDA/bin:$PATH"

echo "[1/5] 补丁状态核验（generate.cpp/server.py 已带三补丁，来自与 4090 一致的二进制源码态） $(date +%T)"
r=$(grep -c 'same-chat rewind' "$S/src/program/generate.cpp"); p=$(grep -c 'pg.tensors().size() == 1' "$S/src/program/generate.cpp"); f=$(grep -c sampling_force "$S/serve/server.py"); c=$(grep -c 'strata_conversation\|drop_replaced' "$S/serve/server.py" "$S/src/program/generate.cpp" | cut -d: -f2 | paste -sd+ - | bc)
echo "    markers: rewind=$r ple=$p force=$f conv_total=$c"
[ "$r" -ge 1 ] && [ "$p" -eq 1 ] && [ "$f" -ge 10 ] && [ "$c" -eq 0 ] || { echo "PATCH STATE BAD"; exit 1; }

echo "[2/5] cmake configure (engine, sm_86, tests on) $(date +%T)"
cmake -G Ninja -S "$S" -B "$S/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
  -DCMAKE_CUDA_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK" \
  -DSTRATA_GGML_DIR="$S/_llamacpp" \
  -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_MMQ_KQUANTS=OFF -DSTRATA_Q6K_EXPERTS=OFF -DSTRATA_ORCA_Q4KS_MMQ=OFF \
  -DSTRATA_PREFILL_MMQ=OFF -DSTRATA_PORTABLE=OFF -DSTRATA_WERROR=OFF \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON || exit 1

echo "[3/5] ninja strata + conversation_cache_test $(date +%T)"
ninja -C "$S/build" -j 8 strata conversation_cache_test || exit 1
"$S/build/conversation_cache_test" 2>&1 | tail -1
echo "    -> $("$S/build/strata" --version 2>&1 | head -1)"

echo "[4/5] cmake configure (vision) $(date +%T)"
cmake -G Ninja -S "$S/tools/vision" -B "$S/build-vision" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
  -DCMAKE_CUDA_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK" \
  -DLLAMA_DIR="$S/_llamacpp" \
  -DSTRATA_VISION_CUDA=ON || exit 1

echo "[5/5] ninja strata-vision $(date +%T)"
ninja -C "$S/build-vision" -j 8 || exit 1
ls -la "$S/build-vision/bin/strata-vision" 2>/dev/null || ls -la "$S/build-vision"/*vision* 2>/dev/null
echo "BUILD_DONE $(date +%T)"
