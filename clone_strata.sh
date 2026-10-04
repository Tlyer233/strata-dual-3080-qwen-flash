#!/bin/bash
# Shallow clone Strata via gh-proxy (03_downloads_mirrors.md).
set -euo pipefail
DEST="${STRATA_ROOT:-$HOME/src/Strata}"
TAG="${STRATA_TAG:-v0.1.38}"
ZIP="https://gh-proxy.com/https://github.com/Niko1221/Strata/archive/refs/tags/${TAG}.zip"
if [[ -f "$DEST/setup.py" ]]; then
  echo "already have $DEST" >&2
  exit 0
fi
mkdir -p "$(dirname "$DEST")"
tmp="$(mktemp -d)"
aria2c -c -x 4 -s 4 -k 8M --file-allocation=none --max-tries=0 --retry-wait=5 \
  --timeout=120 --connect-timeout=30 --disable-ipv6=true \
  -d "$tmp" -o strata.zip "$ZIP"
unzip -q "$tmp/strata.zip" -d "$tmp"
mv "$tmp"/Strata-"${TAG#v}" "$DEST"
rm -rf "$tmp"
echo "cloned $DEST" >&2
