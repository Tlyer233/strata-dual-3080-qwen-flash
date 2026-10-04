#!/bin/bash
# Swift 1.5 bartowski IQ4_XS pack on the same 0.1.38 engine as launch_swift_iq3xxs.sh.
# setup.py refuses this filename; pack by hand with --compat-bf16.
# --resident-experts: GPU caches first, RAM holds the rest (needs resident-on-split patch).
set -euo pipefail
ROOT="${STRATA_ROOT:-$HOME/src/Strata}"
GGUF="${GGUF_DIR:-$HOME/models/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF/IQ4_XS}"
SHARD1="$GGUF/Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS-00001-of-00003.gguf"
PACK="${PACK_DIR:-$HOME/src/Strata-data/packs/swift-iq4_xs}"
MTP="${MTP_DIR:-$HOME/src/Strata-data/mtp/rt}"
MMPROJ="${MMPROJ:-$HOME/models/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf}"
HAND="$(cd "$(dirname "$0")" && pwd)"
CTX="${CTX:-102400}"
PORT="${PORT:-8005}"
GPUS="${GPUS:-1,0}"
KV="${KV:-k8v4}"
TAG="swift-iq4_xs"
CFG="$ROOT/strata-${TAG}.json"
RUN="$ROOT/run-${TAG}.sh"
CUDA_PIP="$HOME/venvs/sglang-0519/lib/python3.12/site-packages/nvidia/cu13"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export STRATA_GGUF_PY="${STRATA_GGUF_PY:-$ROOT/third_party/llama.cpp/gguf-py}"

if [[ ! -x "$ROOT/engine/strata" || ! -x "$ROOT/engine/strata-vision" ]]; then
  echo "missing compiled engine in $ROOT/engine" >&2
  exit 1
fi
ver="$("$ROOT/.venv/bin/python" -c 'import json; print(json.load(open("'"$ROOT"'/engine/BUILD.json"))["version"])')"
if [[ "$ver" != "0.1.38" ]]; then
  echo "engine is $ver, need 0.1.38" >&2
  exit 1
fi
if [[ ! -f "$SHARD1" ]]; then
  echo "missing $SHARD1 — run: bash $HAND/scripts/download_swift15_flash_iq4xs.sh" >&2
  exit 1
fi
if [[ "$(stat -c %s "$SHARD1")" != 44822920960 ]]; then
  echo "IQ4_XS 00001 size mismatch" >&2
  exit 1
fi
if [[ ! -f "$MTP/experts.bin" ]]; then
  echo "missing Swift MTP $MTP — run: bash $HAND/pack_official_mtp.sh" >&2
  exit 1
fi
if [[ ! -f "$MMPROJ" ]]; then
  echo "missing Swift mmproj $MMPROJ" >&2
  exit 1
fi
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "missing $ROOT/.venv" >&2
  exit 1
fi

mkdir -p "$PACK"
if [[ ! -f "$PACK/native_experts.txt" || ! -f "$PACK/tokenizer/vocab.json" ]]; then
  echo "packing $SHARD1 -> $PACK (--compat-bf16, no experts.bin)" >&2
  "$ROOT/.venv/bin/python" -u "$ROOT/tools/iq_pack.py" \
    --gguf "$SHARD1" --out "$PACK" --compat-bf16
fi
if [[ ! -f "$PACK/native_experts.txt" ]]; then
  echo "pack failed: no $PACK/native_experts.txt" >&2
  exit 1
fi

"$ROOT/.venv/bin/python" -u - "$CFG" "$RUN" "$ROOT" "$PACK" "$SHARD1" "$MTP" "$MMPROJ" "$CTX" "$PORT" "$GPUS" "$KV" "$CUDA_PIP" <<'PY'
import json, os, stat, sys
cfg_path, run_path, root, pack, shard1, mtp, mmproj, ctx, port, gpus, kv, cuda_pip = sys.argv[1:]
gpu = [int(x) for x in gpus.split(",") if x.strip() != ""]
lib = [
    os.path.join(cuda_pip, "bin"),
    os.path.join(cuda_pip, "lib64"),
]
cfg = {
    "exe": os.path.join(root, "engine", "strata"),
    "args": [
        "--pack", pack,
        "--native", shard1,
        "--expert-profile", os.path.join(root, "data", "expert-profile.bin"),
        "--mtp", mtp,
        "--vision",
        "--max-context", ctx,
        "--kv", kv,
        "--spec", "6",
        "--mtp-max-t", "4",
        "--suffix-draft", "3",
        "--spec-min-p", "0.5",
        "--prefill", "2048",
        "--expert-cache", "auto",
        "--vram-reserve-mib", "1100",
        "--resident-experts",
    ],
    "cwd": root,
    "tokenizer": os.path.join(pack, "tokenizer"),
    "model_name": "swift-1.5-iq4_xs",
    "log": os.path.join(root, "strata-swift-iq4_xs.log"),
    "lib_dirs": lib,
    "port": int(port),
    "gpu": gpu,
    "gpus_asked": True,
    "layer_split": "26",
    "draft_vocab": "cjk",
    "host": "0.0.0.0",
    "fit_max_tokens": True,
    "vision": {
        "exe": os.path.join(root, "engine", "strata-vision"),
        "mmproj": mmproj,
        "model": shard1,
        "gpu": True,
        "max_tokens": 1024,
    },
}
open(cfg_path, "w", encoding="utf-8").write(json.dumps(cfg, indent=1) + "\n")
body = (
    "#!/bin/sh\n"
    f"cd \"{root}\"\n"
    f"exec \"{root}/.venv/bin/python\" \"{root}/serve/server.py\" "
    f"--engine strata --config \"{cfg_path}\" --host 0.0.0.0 --port {port} --fit-max-tokens --open\n"
)
open(run_path, "w", encoding="utf-8", newline="\n").write(body)
os.chmod(run_path, os.stat(run_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
print("wrote", cfg_path)
print("wrote", run_path)
PY

exec "$ROOT/.venv/bin/python" -u "$HAND/apply_adaptive_mtp.py" \
  --config "$CFG" \
  --context "$CTX" --prefill 2048 --vram-reserve-mib 1100 \
  --spec-min-p 0.5 --kv "$KV" --mtp "$MTP" --resident-experts
