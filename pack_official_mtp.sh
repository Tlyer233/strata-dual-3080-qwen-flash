#!/bin/bash
# Swift 1.5 Qwen3.8-Flash-Next MTP (125B Flash, NOT the 27B dense model).
# Source: ukisai/Swift-Qwen3.8-Flash-Next BF16 mtp.* (author checkpoint), packed q2_0.
# That is the target's own draft head (card: "checkpoint includes the MTP head").
# Unsloth shared-Q8_0 is llama.cpp-only and trained on original Qwen, not Swift.
# Strata has no DFlash; runtime only loads Q2_0 expert blobs.
# 03_downloads_mirrors.md: 7897 + huggingface.co.
set -euo pipefail
ROOT="${STRATA_ROOT:-$HOME/src/Strata}"
DATA="${STRATA_DATA:-$HOME/src/Strata-data}"
MTP="${MTP_DIR:-$DATA/mtp}"
PY="${ROOT}/.venv/bin/python"
HAND="$(cd "$(dirname "$0")" && pwd)"
LOG="${LOG:-$HAND/logs/pack_swift_mtp.log}"
export STRATA_MTP_REPO="${STRATA_MTP_REPO:-swift}"

mkdir -p "$(dirname "$LOG")" "$MTP"
exec > >(stdbuf -oL tee -a "$LOG") 2>&1

proxy="${https_proxy:-${http_proxy:-http://127.0.0.1:7897}}"
if curl -fsS --max-time 8 -o /dev/null --proxy "$proxy" https://huggingface.co; then
  export https_proxy="$proxy" http_proxy="$proxy"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
  echo "=== $(date -Iseconds) MTP q2_0  repo=$STRATA_MTP_REPO  proxy=$https_proxy  out=$MTP ==="
else
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  echo "=== $(date -Iseconds) MTP q2_0  repo=$STRATA_MTP_REPO  no-proxy $HF_ENDPOINT  out=$MTP ==="
fi
if [[ ! -x "$PY" ]]; then
  echo "missing $PY" >&2
  exit 1
fi

if [[ ! -f "$MTP/mtp-manifest.json" ]]; then
  "$PY" -u "$HAND/mtp_aria2_fetch.py" "$MTP"
else
  echo "manifest present, skip fetch"
fi

if [[ ! -f "$MTP/mtp-q2_0.gguf" ]]; then
  "$PY" -u "$ROOT/tools/mtp_pack.py" --src "$MTP" --experts q2_0 --out "$MTP/mtp-q2_0.gguf"
else
  echo "mtp-q2_0.gguf present, skip pack"
fi

if [[ ! -f "$MTP/rt/experts.bin" ]]; then
  "$PY" -u "$ROOT/tools/mtp_rt.py" --gguf "$MTP/mtp-q2_0.gguf" --out "$MTP/rt"
else
  echo "rt/experts.bin present, skip mtp_rt"
fi

if [[ -f "$ROOT/data/draft_vocab.bin" ]]; then
  cp -f "$ROOT/data/draft_vocab.bin" "$MTP/rt/draft_vocab.bin"
fi

ls -lh "$MTP/mtp-q2_0.gguf" "$MTP/rt/experts.bin" "$MTP/rt/dense.bin"
echo "=== $(date -Iseconds) MTP q2_0 ready at $MTP/rt ==="
