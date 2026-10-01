#!/bin/bash
# Swift 1.5 IQ3_XXS pack + dual 3080 json. Engine is already compiled in ~/src/Strata.
# Draft is Swift Flash's own mtp.* packed q2_0 (not Unsloth, not original-Qwen sidecar).
set -euo pipefail
ROOT="${STRATA_ROOT:-$HOME/src/Strata}"
GGUF="${GGUF_DIR:-$HOME/models/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF}"
FAMILY="${FAMILY:-swift}"
MODEL="${MODEL:-IQ3_XXS}"
# setup.py clamps IQ3 262K→128K on 64 GB RAM; we pass 128K here then raise to 262K + k8v4 after.
CTX="${CTX:-131072}"
VISION="${VISION:-gpu}"
PORT="${PORT:-8080}"
GPUS="${GPUS:-1,0}"
KV="${KV:-k8v4}"
HAND="$(cd "$(dirname "$0")" && pwd)"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export STRATA_PREBUILT_URL="${STRATA_PREBUILT_URL:-https://gh-proxy.com/https://github.com/Niko1221/Strata/releases/latest/download/}"
export STRATA_SKIP_BUILD="${STRATA_SKIP_BUILD:-1}"

if [[ ! -f "$ROOT/setup.py" ]]; then
  echo "missing $ROOT (git clone via gh-proxy into ~/src/Strata)" >&2
  exit 1
fi
if [[ ! -x "$ROOT/engine/strata" || ! -x "$ROOT/engine/strata-vision" ]]; then
  echo "missing compiled engine in $ROOT/engine — rebuild first" >&2
  exit 1
fi
if [[ ! -d "$GGUF" ]]; then
  echo "missing GGUF dir $GGUF" >&2
  exit 1
fi

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  python3 -m venv --without-pip "$ROOT/.venv"
  curl -fL --retry 5 -o /tmp/get-pip.py \
    "https://gh-proxy.com/https://github.com/pypa/get-pip/raw/main/public/get-pip.py"
  "$ROOT/.venv/bin/python" /tmp/get-pip.py -i "$PIP_INDEX_URL"
fi

if [[ ! -f "$HOME/src/Strata-data/mtp/rt/experts.bin" ]]; then
  echo "Swift MTP q2_0 missing — run: bash $HAND/pack_official_mtp.sh" >&2
  exit 1
fi

cd "$ROOT"
"$ROOT/.venv/bin/python" -u setup.py \
  --family "$FAMILY" --model "$MODEL" --context "$CTX" --vision "$VISION" \
  --kv "$KV" --gpus "$GPUS" --layer-split auto --draft-vocab cjk \
  --port "$PORT" --yes --no-start --gguf-dir "$GGUF" "$@"

exec "$ROOT/.venv/bin/python" -u "$HAND/apply_adaptive_mtp.py" \
  --config "$ROOT/strata-swift-iq3_xxs.json"
