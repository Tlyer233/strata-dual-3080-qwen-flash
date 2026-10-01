#!/bin/bash
# Swift 1.5 IQ3_XXS on both 3080s. GPU1 first (cleaner), GPU0 second.
# Adaptive MTP: --spec 6 --mtp-max-t 4 --suffix-draft 3 --spec-min-p 0.5
# 200k ctx + GPU vision. Prefill 2048 so the rest of VRAM is expert cache.
set -euo pipefail
ROOT="${STRATA_ROOT:-$HOME/src/Strata}"
PORT="${PORT:-8005}"
# Only these knobs matter. apply_adaptive_mtp.py writes them into the json
# before the engine starts. The echo below is a banner, not a second config.
CTX="${CTX:-206848}"
PREFILL="${PREFILL:-2048}"
VRAM_RESERVE="${VRAM_RESERVE:-800}"
SPEC_MIN_P="${SPEC_MIN_P:-0.5}"
KV="${KV:-k8v4}"
RUN="${RUN:-$ROOT/run-swift-iq3_xxs.sh}"
CFG="${CFG:-$ROOT/strata-swift-iq3_xxs.json}"
CUDA_PIP="$HOME/venvs/sglang-0519/lib/python3.12/site-packages/nvidia/cu13"
HAND="$(cd "$(dirname "$0")" && pwd)"

ulimit -l unlimited 2>/dev/null || true

if command -v systemctl >/dev/null 2>&1; then
  systemctl --user stop sglang-dspark 2>/dev/null || true
fi
fuser -k 8000/tcp 8001/tcp 8080/tcp "${PORT}/tcp" 2>/dev/null || true
sleep 2
if fuser "${PORT}/tcp" >/dev/null 2>&1; then
  echo "port $PORT still busy" >&2
  ss -ltnp | grep ":${PORT} " >&2 || true
  exit 1
fi

if [[ ! -x "$RUN" || ! -f "$CFG" ]]; then
  echo "missing $RUN / $CFG — wait for IQ3 download, then: bash $HAND/setup_swift_iq3xxs.sh" >&2
  exit 1
fi

"$ROOT/.venv/bin/python" -u "$HAND/apply_adaptive_mtp.py" --config "$CFG" \
  --context "$CTX" --prefill "$PREFILL" --vram-reserve-mib "$VRAM_RESERVE" \
  --spec-min-p "$SPEC_MIN_P" --kv "$KV"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export LD_LIBRARY_PATH="$CUDA_PIP/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$ROOT"
echo "Strata Swift IQ3_XXS  gpus=1,0  split=26  ctx=$CTX  kv=$KV  vision(gpu)  MTP  prefill=$PREFILL  reserve=${VRAM_RESERVE}  min_p=$SPEC_MIN_P  host=0.0.0.0  fit_max_tokens  port=$PORT" >&2
exec bash "$RUN"
