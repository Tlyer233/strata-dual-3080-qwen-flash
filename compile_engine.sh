#!/bin/bash
# Compile Strata 0.1.31 engine+vision for sm_86. Do NOT use the GitHub prebuilt (sm_120).
# This machine: pip CUDA 13.4 nvcc from sglang-0519, not system cuda-toolkit.
set -euo pipefail
ROOT="${STRATA_ROOT:-$HOME/src/Strata}"
HAND="$(cd "$(dirname "$0")" && pwd)"
ARCH="${STRATA_CUDA_ARCH:-86}"
LLAMA_PIN="${LLAMA_CPP_COMMIT:-3cf03257f219afbe7334045ff7c6a06ac68c627d}"
CUDA_PIP="${CUDA_PIP:-$HOME/venvs/sglang-0519/lib/python3.12/site-packages/nvidia/cu13}"
NVCC="${NVCC:-$CUDA_PIP/bin/nvcc}"
PY="${ROOT}/.venv/bin/python"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

if [[ ! -f "$ROOT/setup.py" ]]; then
  echo "missing $ROOT — run: bash $HAND/clone_strata.sh" >&2
  exit 1
fi
if [[ ! -x "$NVCC" ]]; then
  echo "missing nvcc $NVCC — install nvidia-cuda-nvcc-cu13 in a venv, or set NVCC=" >&2
  exit 1
fi

export CCCL_DISABLE_CTK_COMPATIBILITY_CHECK=1
export PATH="$(dirname "$NVCC"):$PATH"
export LD_LIBRARY_PATH="$CUDA_PIP/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ ! -x "$PY" ]]; then
  python3 -m venv --without-pip "$ROOT/.venv"
  curl -fL --retry 5 -o /tmp/get-pip.py \
    "https://gh-proxy.com/https://github.com/pypa/get-pip/raw/main/public/get-pip.py"
  "$ROOT/.venv/bin/python" /tmp/get-pip.py -i "$PIP_INDEX_URL"
fi
"$PY" -m pip install -q -i "$PIP_INDEX_URL" cmake ninja numpy jinja2 regex pyyaml tqdm requests pillow psutil

mkdir -p "$ROOT/third_party" "$ROOT/engine" "$ROOT/data"
if [[ ! -f "$ROOT/third_party/llama.cpp/ggml/CMakeLists.txt" ]]; then
  tmp="$(mktemp -d)"
  aria2c -c -x 4 -s 4 -k 8M --file-allocation=none --max-tries=0 --retry-wait=5 \
    --timeout=120 --connect-timeout=30 --disable-ipv6=true \
    -d "$tmp" -o llama.zip \
    "https://gh-proxy.com/https://github.com/ggml-org/llama.cpp/archive/${LLAMA_PIN}.zip"
  unzip -q "$tmp/llama.zip" -d "$tmp"
  rm -rf "$ROOT/third_party/llama.cpp"
  mv "$tmp"/llama.cpp-"${LLAMA_PIN}"* "$ROOT/third_party/llama.cpp"
  rm -rf "$tmp"
fi

cp -f "$HAND/data/expert-profile.bin" "$ROOT/data/expert-profile.bin"
cp -f "$HAND/data/draft_vocab.bin" "$ROOT/data/draft_vocab.bin"

NINJA="$("$PY" -c 'import shutil; print(shutil.which("ninja"))')"
CMAKE="$("$PY" -c 'import shutil; print(shutil.which("cmake"))')"
JOBS="${JOBS:-$(( $(nproc) / 2 ))}"
[[ "$JOBS" -lt 2 ]] && JOBS=2

cmake_engine() {
  local src="$1" bdir="$2" target="$3"
  shift 3
  "$CMAKE" -G Ninja -DCMAKE_MAKE_PROGRAM="$NINJA" -S "$src" -B "$bdir" \
    -DCMAKE_BUILD_TYPE=Release "$@"
  if ! "$CMAKE" --build "$bdir" --target "$target" -j "$JOBS"; then
    echo "rebuild retry for $target" >&2
    "$CMAKE" --build "$bdir" --target "$target" -j "$JOBS"
  fi
}

cmake_engine "$ROOT" "$ROOT/build" strata \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DSTRATA_GGML_DIR="$ROOT/third_party/llama.cpp"
cp -f "$ROOT/build/strata" "$ROOT/engine/strata"
chmod +x "$ROOT/engine/strata"

cmake_engine "$ROOT/tools/vision" "$ROOT/build-vision" strata-vision \
  -DLLAMA_DIR="$ROOT/third_party/llama.cpp" \
  -DSTRATA_VISION_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DCMAKE_CUDA_COMPILER="$NVCC"
cp -f "$ROOT/build-vision/bin/strata-vision" "$ROOT/engine/strata-vision"
chmod +x "$ROOT/engine/strata-vision"

"$PY" - "$ROOT/engine" "$ARCH" <<'PY'
import json, pathlib, sys
eng = pathlib.Path(sys.argv[1])
arch = int(sys.argv[2])
stamp = {
    "source": "local",
    "version": "0.1.31",
    "archs": [arch],
    "vision": "gpu",
    "cuda_dirs": [],
    "src": "skip",
    "vision_src": "skip",
}
(eng / "BUILD.json").write_text(json.dumps(stamp, indent=1) + "\n")
print((eng / "BUILD.json").read_text())
PY

"$ROOT/engine/strata" --help >/dev/null
echo "engine ready: $ROOT/engine/strata  arch=sm_${ARCH}  llama=${LLAMA_PIN:0:7}"
echo "always launch setup.py with STRATA_SKIP_BUILD=1 after this compile"
