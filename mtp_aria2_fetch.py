#!/usr/bin/env python3
"""Swift 1.5 Flash-Next MTP tensors: huggingface.co via 7897, HTTP Range, parallel curl.

Default repo is ukisai/Swift-Qwen3.8-Flash-Next (the target's own mtp.* head).
Qwen/Qwen3.8-Flash-Next is the base-model fallback: STRATA_MTP_REPO=qwen.

Writes tensors/ + mtp-inventory.json + mtp-manifest.json for tools/mtp_pack.py.
Strata cannot load Unsloth GGUF sidecars or DFlash.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
PROXY = os.environ.get("https_proxy") or os.environ.get("http_proxy") or ""
if PROXY:
    os.environ.setdefault("https_proxy", PROXY)
    os.environ.setdefault("http_proxy", PROXY)

REPOS = {
    "swift": f"{ENDPOINT}/ukisai/Swift-Qwen3.8-Flash-Next/resolve/main/",
    "qwen": f"{ENDPOINT}/Qwen/Qwen3.8-Flash-Next/resolve/main/",
}
REPO = REPOS.get(os.environ.get("STRATA_MTP_REPO", "swift"), REPOS["swift"])
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/src/Strata-data/mtp"))
OUT.mkdir(parents=True, exist_ok=True)
TDIR = OUT / "tensors"
TDIR.mkdir(parents=True, exist_ok=True)
INDEX = OUT / "model.safetensors.index.json"


def curl_proxy() -> list[str]:
    return ["--proxy", PROXY] if PROXY else ["--noproxy", "*"]


def curl(args: list[str]) -> subprocess.CompletedProcess:
    cmd = [
        "curl", "-fL", "--retry", "8", "--retry-delay", "2", "--retry-all-errors",
        "--connect-timeout", "30", "--max-time", "600",
        *curl_proxy(), "-H", "User-Agent: strata-mtp-fetch",
        *args,
    ]
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE)


def curl_range(url: str, start: int, end: int, dest: Path | None = None) -> bytes | None:
    extra = ["-r", f"{start}-{end}", url]
    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        subprocess.run(
            [
                "curl", "-fL", "--retry", "8", "--retry-delay", "2", "--retry-all-errors",
                "--connect-timeout", "30", "--max-time", "600",
                *curl_proxy(), "-H", "User-Agent: strata-mtp-fetch",
                "-C", "-", "-o", str(tmp), "-r", f"{start}-{end}", url,
            ],
            check=True,
        )
        tmp.replace(dest)
        return None
    return curl(extra).stdout


print(f"endpoint={ENDPOINT} proxy={PROXY or 'none'} repo={REPO} out={OUT}", flush=True)
if INDEX.exists() and INDEX.stat().st_size > 1000:
    print("index.json already present, skip", flush=True)
else:
    print("GET model.safetensors.index.json", flush=True)
    curl(["-o", str(INDEX), REPO + "model.safetensors.index.json"])

weight_map = json.loads(INDEX.read_text())["weight_map"]
mtp = {k: v for k, v in weight_map.items() if k.startswith("mtp.")}
shards = sorted(set(mtp.values()))
print(f"mtp tensors {len(mtp)} shards {len(shards)}", flush=True)


def shard_header(shard: str):
    url = REPO + shard
    n = struct.unpack("<Q", curl_range(url, 0, 7))[0]
    header = json.loads(curl_range(url, 8, 8 + n - 1))
    return shard, 8 + n, header


headers = {}
with ThreadPoolExecutor(max_workers=4) as ex:
    futs = [ex.submit(shard_header, s) for s in shards]
    for i, fut in enumerate(as_completed(futs), 1):
        shard, base, header = fut.result()
        headers[shard] = (base, header)
        print(f"header {i}/{len(shards)} {shard}", flush=True)

rows, total = [], 0
for shard in shards:
    base, header = headers[shard]
    for name, meta in header.items():
        if name in mtp and mtp[name] == shard:
            a, b = meta["data_offsets"]
            rows.append(
                dict(
                    name=name,
                    shard=shard,
                    dtype=meta["dtype"],
                    shape=meta["shape"],
                    start=base + a,
                    end=base + b - 1,
                    bytes=b - a,
                )
            )
            total += b - a
rows.sort(key=lambda r: r["name"])
(OUT / "mtp-inventory.json").write_text(
    json.dumps(dict(repo=REPO, total_bytes=total, tensors=rows), indent=1), encoding="utf-8"
)
print(f"inventory {len(rows)} tensors {total / 1e9:.3f} GB", flush=True)


def fetch_row(r):
    dest = TDIR / (r["name"] + ".bin")
    if dest.exists() and dest.stat().st_size == r["bytes"]:
        return r["name"], "skip"
    curl_range(REPO + r["shard"], r["start"], r["end"], dest)
    sz = dest.stat().st_size
    if sz != r["bytes"]:
        raise SystemExit(f"{dest}: size {sz} != {r['bytes']}")
    return r["name"], sz


with ThreadPoolExecutor(max_workers=4) as ex:
    futs = [ex.submit(fetch_row, r) for r in rows]
    done = 0
    for fut in as_completed(futs):
        name, info = fut.result()
        done += 1
        print(f"tensor {done}/{len(rows)} {name} {info}", flush=True)

manifest = []
for r in rows:
    path = TDIR / (r["name"] + ".bin")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    manifest.append(dict(r, file=str(path.relative_to(OUT)), sha256=h.hexdigest()))
(OUT / "mtp-manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
print("manifest ok", flush=True)
