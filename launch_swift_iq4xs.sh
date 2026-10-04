#!/bin/bash
# Swift 1.5 bartowski IQ4_XS on both 3080s. Same 0.1.38 engine as launch_swift_iq3xxs.sh.
# Middle path (--resident-experts): fill both GPUs' expert caches first, copy only the
# experts no card holds into RAM. Needs patches/resident-on-split-v0.1.38.patch.
# See RESIDENT_ON_SPLIT.md. PLE/n-gram stays on the GGUF.
set -euo pipefail
ROOT="${STRATA_ROOT:-$HOME/src/Strata}"
PORT="${PORT:-8005}"
CTX="${CTX:-102400}"
PREFILL="${PREFILL:-2048}"
# 500 left 188 MiB free; first prefill died (cublas 14). Engine asked for 824.
VRAM_RESERVE="${VRAM_RESERVE:-1100}"
SPEC_MIN_P="${SPEC_MIN_P:-0.5}"
KV="${KV:-k8v4}"
GPUS="${GPUS:-1,0}"
MTP="${MTP_DIR:-$HOME/src/Strata-data/mtp/rt}"
RUN="${RUN:-$ROOT/run-swift-iq4_xs.sh}"
CFG="${CFG:-$ROOT/strata-swift-iq4_xs.json}"
CUDA_PIP="$HOME/venvs/sglang-0519/lib/python3.12/site-packages/nvidia/cu13"
HAND="$(cd "$(dirname "$0")" && pwd)"

ulimit -l unlimited 2>/dev/null || true

if command -v systemctl >/dev/null 2>&1; then
  systemctl --user stop sglang-dspark 2>/dev/null || true
fi
pkill -x strata 2>/dev/null || true
pkill -x strata-vision 2>/dev/null || true
fuser -k 8000/tcp 8001/tcp 8080/tcp "${PORT}/tcp" 2>/dev/null || true
sleep 2
if fuser "${PORT}/tcp" >/dev/null 2>&1; then
  echo "port $PORT still busy" >&2
  ss -ltnp | grep ":${PORT} " >&2 || true
  exit 1
fi

if [[ ! -x "$RUN" || ! -f "$CFG" ]]; then
  echo "missing $RUN / $CFG — run: bash $HAND/setup_swift_iq4xs.sh" >&2
  exit 1
fi
if [[ ! -f "$MTP/experts.bin" ]]; then
  echo "missing Swift MTP at $MTP — run: bash $HAND/pack_official_mtp.sh" >&2
  exit 1
fi
if ! grep -q "resident RAM mode with a layer split" "$ROOT/src/program/generate.cpp" 2>/dev/null; then
  echo "engine source is missing the dual-GPU resident patch — run: python3 $HAND/patches/apply_local.py && bash $HAND/compile_engine.sh" >&2
  exit 1
fi

"$ROOT/.venv/bin/python" -u "$HAND/apply_adaptive_mtp.py" --config "$CFG" \
  --context "$CTX" --prefill "$PREFILL" --vram-reserve-mib "$VRAM_RESERVE" \
  --spec-min-p "$SPEC_MIN_P" --kv "$KV" --mtp "$MTP" --resident-experts

if ! grep -q -- '--resident-experts' "$CFG"; then
  echo "refusing $CFG: IQ4 dual-GPU needs --resident-experts (GPU cache first, complement in RAM)" >&2
  exit 1
fi
if grep -q -- '--mmap-experts' "$CFG"; then
  echo "refusing $CFG: plain --mmap-experts leaves RAM empty" >&2
  exit 1
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export LD_LIBRARY_PATH="$CUDA_PIP/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset STRATA_ARENA_PIN_GIB
cd "$ROOT"
echo "Strata Swift IQ4_XS  gpus=$GPUS  split=26  ctx=$CTX  kv=$KV  vision(gpu)  MTP=swift  resident-experts (GPU then RAM)  prefill=$PREFILL  reserve=${VRAM_RESERVE}  min_p=$SPEC_MIN_P  host=0.0.0.0  fit_max_tokens  port=$PORT" >&2
exec bash "$RUN"
