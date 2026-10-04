#!/bin/bash
# ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF IQ4_XS (3 shards, ~97.7G).
# 03_downloads_mirrors.md: hf-mirror + aria2. Follow 302 to cas-bridge; no --lowest-speed-limit.
set -euo pipefail

LOG="${LOG:-$(cd "$(dirname "$0")/.." && pwd)/logs/download_swift15_flash_iq4xs.log}"
OUT="${OUT:-$HOME/models/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF/IQ4_XS}"
REPO="ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF"
REV="main"
MIRROR="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p "$(dirname "$LOG")" "$OUT"
exec > >(stdbuf -oL tee -a "$LOG") 2>&1

echo "=== $(date -Iseconds) start pid=$$ mirror=$MIRROR ==="
echo "out=$OUT"
echo "log=$LOG"

ARIA=(aria2c -c -x 4 -s 4 -k 8M --file-allocation=none --max-tries=0 --retry-wait=8
  --timeout=120 --connect-timeout=30 --disable-ipv6=true
  --allow-overwrite=true --auto-file-renaming=false
  --console-log-level=notice --summary-interval=1)

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY no_proxy NO_PROXY || true

get_file() {
  local f="$1" expect="$2"
  local path="$OUT/$f"
  if [[ -f "$path" ]] && [[ "$(stat -c %s "$path")" == "$expect" ]]; then
    echo "=== skip $f (size ok) ==="
    return 0
  fi
  local url="${MIRROR}/${REPO}/resolve/${REV}/IQ4_XS/${f}"
  echo "=== $(date -Iseconds) GET $f expect=$expect ==="
  local ok=0 url_try
  for url_try in "$url" "${url}?download=true"; do
    echo "--- try $url_try ---"
    (
      cd "$OUT"
      "${ARIA[@]}" -o "$f" "$url_try"
    ) && ok=1 && break
    echo "--- fail, next url ---"
  done
  if [[ "$ok" != 1 ]]; then
    echo "FAIL $f" >&2
    return 1
  fi
  local got
  got="$(stat -c %s "$path")"
  echo "=== done $f size=$got ==="
  if [[ "$got" != "$expect" ]]; then
    echo "size mismatch $f got=$got expect=$expect" >&2
    return 1
  fi
}

cat > "$OUT/SHA256SUMS" <<'EOF'
75c299e4816fe62303fc262b88a76f4191002f6a8e65d51aaa81e32c3aea2c6c  Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS-00001-of-00003.gguf
c78d27db136d8194bd86441d3a31edd71f3f9a9d04cb8b1473b66ca0b11f5f35  Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS-00002-of-00003.gguf
7ce972d2a67d017a33886f7fd46e37b9eef127953f0acf270b305a8e291a7828  Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS-00003-of-00003.gguf
EOF

get_file Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS-00001-of-00003.gguf 44822920960
get_file Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS-00002-of-00003.gguf 44822625408
get_file Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS-00003-of-00003.gguf 8036289376

echo "=== sizes ==="
stat -c '%n %s' \
  "$OUT/Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS-00001-of-00003.gguf" \
  "$OUT/Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS-00002-of-00003.gguf" \
  "$OUT/Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS-00003-of-00003.gguf"
(cd "$OUT" && sha256sum -c SHA256SUMS)
du -sh "$OUT"
echo "=== $(date -Iseconds) ALL DONE ==="
echo "log: $LOG"
echo "out: $OUT"
