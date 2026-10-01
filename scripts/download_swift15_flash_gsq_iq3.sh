#!/bin/bash
# ukisai Swift 1.5 Flash-Next GSQ-RCO IQ3_XXS + mmproj.
# 03_downloads_mirrors.md: 7897 + huggingface.co; 7897 down -> hf-mirror no proxy.
set -euo pipefail

LOG="${LOG:-$(cd "$(dirname "$0")/.." && pwd)/logs/download_swift15_flash_gsq_iq3.log}"
OUT="${OUT:-$HOME/models/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF}"
REPO="ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
mkdir -p "$(dirname "$LOG")" "$OUT"
exec > >(stdbuf -oL tee -a "$LOG") 2>&1

proxy="${https_proxy:-${http_proxy:-http://127.0.0.1:7897}}"
HF_HOST="https://huggingface.co"
if curl -fsS --max-time 8 -o /dev/null --proxy "$proxy" https://huggingface.co; then
  export https_proxy="$proxy" http_proxy="$proxy"
  CURL_PROXY=(--proxy "$https_proxy")
  ARIA_PROXY=(--http-proxy="$https_proxy")
  echo "=== $(date -Iseconds) start pid=$$ proxy=$https_proxy host=$HF_HOST ==="
else
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true
  HF_HOST="${HF_ENDPOINT:-https://hf-mirror.com}"
  CURL_PROXY=()
  ARIA_PROXY=()
  echo "=== $(date -Iseconds) start pid=$$ no-proxy host=$HF_HOST ==="
fi

ARIA=(aria2c -c -x 4 -s 4 -k 8M --file-allocation=none --max-tries=0 --retry-wait=8
  --timeout=120 --connect-timeout=30 --disable-ipv6=true
  --allow-overwrite=true --auto-file-renaming=false
  --console-log-level=notice --summary-interval=1
  "${ARIA_PROXY[@]}")

get() {
  local f="$1" expect="$2"
  local url="${HF_HOST}/${REPO}/resolve/main/${f}"
  if [[ -f "$OUT/$f" ]] && [[ "$(stat -c %s "$OUT/$f")" == "$expect" ]]; then
    echo "=== skip $f (size ok) ==="
    date -Iseconds >"$OUT/${f}.done"
    return 0
  fi
  echo "=== $(date -Iseconds) GET $f expect=$expect ==="
  (cd "$OUT" && "${ARIA[@]}" -o "$f" "$url")
  local got
  got=$(stat -c %s "$OUT/$f")
  if [[ "$got" != "$expect" ]]; then
    echo "size mismatch $f got=$got expect=$expect" >&2
    exit 1
  fi
  date -Iseconds >"$OUT/${f}.done"
}

curl -fsSL "${CURL_PROXY[@]}" --retry 8 --retry-delay 2 \
  -o "$OUT/SHA256SUMS" "${HF_HOST}/${REPO}/resolve/main/SHA256SUMS"

get Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf 39785790560
get Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf 36180282560
get mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf 907543520

echo "=== sizes ==="
stat -c '%n %s' \
  "$OUT/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf" \
  "$OUT/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf" \
  "$OUT/mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf"
(cd "$OUT" && sha256sum -c SHA256SUMS --ignore-missing)
du -sh "$OUT"
echo "=== $(date -Iseconds) ALL DONE ==="
