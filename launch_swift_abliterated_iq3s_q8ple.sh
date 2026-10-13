#!/bin/bash
# Swift 1.5 GSQ-RCO abliterated IQ3_S, same experts as launch_swift_abliterated_iq3s.sh,
# but n-gram is sleepyeldrazi Q8_0 PLE (--ple-gguf), not native IQ4_NL shard 2.
# Dual-GPU: gpus=1,0 split=26, 268k ctx, prefill 4096, adaptive MTP, pipeline-windows 2.
set -euo pipefail
ROOT="${STRATA_ROOT:-$HOME/src/Strata}"
PORT="${PORT:-8005}"
CTX="${CTX:-268288}"
PREFILL="${PREFILL:-8192}"
VRAM_RESERVE="${VRAM_RESERVE:-700}"
SPEC_MIN_P="${SPEC_MIN_P:-0.5}"
KV="${KV:-k8v4}"
GPUS="${GPUS:-1,0}"
MTP="${MTP_DIR:-$HOME/src/Strata-data/mtp/rt}"
PLE="${PLE_GGUF:-$HOME/models/sleepyeldrazi/Qwen3.8-Flash-Next-GSQ-RCO-Q8_0-PLE-GGUF/Qwen3.8-Flash-Next-GSQ-RCO-Q8_0-PLE-00002-of-00002.gguf}"
RUN="${RUN:-$ROOT/run-swift-abliterated-iq3_s-q8ple.sh}"
CFG="${CFG:-$ROOT/strata-swift-abliterated-iq3_s-q8ple.json}"
CUDA_PIP="$HOME/venvs/sglang-0519/lib/python3.12/site-packages/nvidia/cu13"
HAND="$HOME/ubuntu_handoff/strata"

ulimit -l unlimited 2>/dev/null || true

if [ "${JSON_ONLY:-0}" = 1 ]; then
  echo "JSON_ONLY=1: 跳过 drain/pkill, 不碰运行中的引擎" >&2
else
  if command -v systemctl >/dev/null 2>&1; then
    systemctl --user stop sglang-dspark 2>/dev/null || true
  fi
  if [[ -x "$HAND/gpu-drain.sh" ]]; then
    bash "$HAND/gpu-drain.sh"
  fi
  pkill -x strata 2>/dev/null || true
  pkill -x strata-vision 2>/dev/null || true
  fuser -k 8000/tcp 8001/tcp 8080/tcp "${PORT}/tcp" 2>/dev/null || true
  sleep 2
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >&2 || true
  if fuser "${PORT}/tcp" >/dev/null 2>&1; then
    echo "port $PORT still busy" >&2
    ss -ltnp | grep ":${PORT} " >&2 || true
    exit 1
  fi
fi

if [[ ! -x "$RUN" || ! -f "$CFG" ]]; then
  echo "missing $RUN / $CFG" >&2
  exit 1
fi
if [[ ! -f "$MTP/experts.bin" ]]; then
  echo "missing Swift MTP at $MTP" >&2
  exit 1
fi
if [[ ! -f "$PLE" ]]; then
  echo "missing Q8_0 PLE $PLE" >&2
  exit 1
fi
"$ROOT/.venv/bin/python" -u "$HAND/apply_adaptive_mtp.py" --config "$CFG" \
  --context "$CTX" --prefill "$PREFILL" --vram-reserve-mib "$VRAM_RESERVE" \
  --spec-min-p "$SPEC_MIN_P" --kv "$KV" --mtp "$MTP" --ple-gguf "$PLE" \
  --gpus "$GPUS" --resident-experts --pipeline-windows "${PIPELINE_WINDOWS:-2}"

if ! grep -q -- '--resident-experts' "$CFG"; then
  echo "refusing $CFG: IQ3_S dual-GPU needs --resident-experts (GPU cache first, complement in RAM)" >&2
  exit 1
fi
if ! grep -q -- '--ple-gguf' "$CFG"; then
  echo "refusing $CFG: Q8 PLE path needs --ple-gguf" >&2
  exit 1
fi
if grep -q -- '--mmap-experts' "$CFG"; then
  echo "refusing $CFG: plain --mmap-experts leaves RAM empty" >&2
  exit 1
fi

# --- 会话停放 (conversation cache): 两个会话之间交替时不重算 prefill (Mac 问一句、切回 dsh) ---
# 官方默认 --conversation-cache-mib 0 = 关。快照只加宿主 RAM, 不占显存; 装不下就 skip parking,
# 不会启动失败 (引擎里的拒绝条件只有 "a layer split on one GPU" 这一条, 我们是双卡 split, 支持).
# 实测 (0.1.42 本机, 2026-10-11): 一个 250K 会话的快照 = 最大 4.27 GiB; 30K 会话 = 0.22-0.73 GiB;
# 中位 0.58 GiB. 切换恢复 250K 用 327-432 ms (对照: 全量 prefill 250K = 111 s).
# 引擎准入公式: 快照 MiB + min_free 地板 <= MemAvailable, 否则 skip parking (降级, 不崩).
# 本机实测 (0.1.42): 刚重启没停放 used 36.3 GiB; 停放 5 个 240K (池 24) 时 used 54 GiB.
#   用户给的内存上限: used 顶到 57 GiB (总 62.7). 一个 240K 会话的停放快照实测 3.6 GiB.
# 为什么必须动 --prompt-cache (issue #1920, 本机日志逐字同款: 长 prompt "0 reused" 38 条, "skip parking" 0 条):
#   池 = --prompt-cache 个检查点, 每个覆盖 --prompt-cache-every 个 token. 一个 240K 会话要 240000/every 个.
#     every 16384(默认) -> 15 个/会话, 2 个长会话就互相挤光, 每轮从 token 0 重读 (0.1.42 默认必踩)
#     every 65536       -> 4-5 个/会话, 6 个占用者要 30 个 > 池 24  -> 还是饥饿 (本机踩过, 就是这个)
#     every 131072      -> 2-3 个/会话, 5 个占用者要 ~15 个, 池 28 有足够余量  <- 现在
#   日志里 "N checkpoints" 是写次数不是复用, 所以计数器看着健康.
#   every 131072 的代价: 回退/分支最多重读 128K token (~55 s); 线性 agent 轮次不受影响.
# --conversation-cache-min-tokens 4096 (官方 PR #1597): 短请求不再占停放槽把长会话顶掉
#   (那条实测: 3 条几百 token 的短请求顶掉 19198 token 长会话 -> 41.4 s 重读; 设门槛后 0.3 s, 304x).
CONV_CACHE_MIB="${CONV_CACHE_MIB:-19456}"
CONV_CACHE_SLOTS="${CONV_CACHE_SLOTS:-6}"
CONV_CACHE_MIN_FREE_MIB="${CONV_CACHE_MIN_FREE_MIB:-4096}"
CONV_CACHE_MIN_TOKENS="${CONV_CACHE_MIN_TOKENS:-4096}"
PROMPT_CACHE_N="${PROMPT_CACHE_N:-28}"
PROMPT_CACHE_EVERY="${PROMPT_CACHE_EVERY:-131072}"
CCPY_BIN="$ROOT/.venv/bin/python"
[ -x "$CCPY_BIN" ] || CCPY_BIN="$(command -v python3)"
"$CCPY_BIN" - "$CFG" \
  --conversation-cache-mib "$CONV_CACHE_MIB" \
  --conversation-cache-slots "$CONV_CACHE_SLOTS" \
  --conversation-cache-min-free-mib "$CONV_CACHE_MIN_FREE_MIB" \
  --conversation-cache-min-tokens "$CONV_CACHE_MIN_TOKENS" \
  --prompt-cache "$PROMPT_CACHE_N" \
  --prompt-cache-every "$PROMPT_CACHE_EVERY" <<'CCPY'
import json, os, shutil, sys

path = sys.argv[1]
want = dict(zip(sys.argv[2::2], sys.argv[3::2]))
with open(path, encoding="utf-8") as f:
    cfg = json.load(f)
args = list(cfg.get("args") or [])

def get(args, flag):
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
    return None

def upsert(args, flag, val):
    out, i, done = [], 0, False
    while i < len(args):
        if args[i] == flag:
            if not done:
                out += [flag, val]
                done = True
            i += 1
            while i < len(args) and not str(args[i]).startswith("-"):
                i += 1
            continue
        out.append(args[i])
        i += 1
    if not done:
        out += [flag, val]
    return out

if all(get(args, f) == v for f, v in want.items()):
    print("conversation cache: 配置里已是这些值, 不改")
    raise SystemExit(0)
for flag, val in want.items():
    args = upsert(args, flag, val)
cfg["args"] = args
if not os.path.exists(path + ".bak-conv-cache"):
    shutil.copy2(path, path + ".bak-conv-cache")
with open(path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=1)
    f.write("\n")
print("conversation cache: 已写入 " + "  ".join(f"{f} {v}" for f, v in want.items()))
CCPY
echo "conversation cache: 预算 ${CONV_CACHE_MIB} MiB / 槽 ${CONV_CACHE_SLOTS} / RAM 地板 ${CONV_CACHE_MIN_FREE_MIB} MiB / 短请求门槛 ${CONV_CACHE_MIN_TOKENS} tok / 检查点池 ${PROMPT_CACHE_N} x every ${PROMPT_CACHE_EVERY} tok  (CONV_CACHE_MIB=0 关掉停放)" >&2

# --- 并发流: 默认单流, 开机(systemd 读 json)/手动启动都先把 parallel 清掉 ---
# 实测双 3080 (0.1.43.0): 每 slot 吃 ~1.55 GiB 显存/stage(专家热层 -22%) + ~11.6 GiB pinned RAM,
# slot 无 MTP draft 时并发 decode 仅 9-11 tok/s/路, 单流也被压到 66-74 (纯单流 93).
# 多流走同名 multi 脚本, 它 export PARALLEL=N>1 + BATCH_MODE:
#   draft (默认, 流少合适): --batch-groups 1 + --batch-mtp -> slot 有 MTP draft (串行窗). #2038 在 0.1.43.1 已修.
#   pipe: 默认 pipelined 组 (slot 无 draft), 官方 4 流/2 GPU +25-30%; 流多时用.
# (0.1.43.0 教训 auto=2 groups×1slot 走流水慢路径 #1906 — 43.1 已修 pipelined 组, 待本机 A/B.)
PARALLEL="${PARALLEL:-1}"
BATCH_MODE="${BATCH_MODE:-draft}"
"$CCPY_BIN" - "$CFG" "$PARALLEL" "$BATCH_MODE" <<'PYP'
import json, sys
path, par, mode = sys.argv[1], int(sys.argv[2] or 1), (sys.argv[3] or "draft").lower()
with open(path, encoding="utf-8") as f:
    cfg = json.load(f)
args = list(cfg.get("args") or [])
out, i = [], 0
while i < len(args):
    if args[i] in ("--batch-groups", "--batch-mtp") or (args[i] == "--batch" and par <= 1):
        i += 1
        while i < len(args) and not str(args[i]).startswith("-"):
            i += 1
        continue
    out.append(args[i])
    i += 1
if par > 1:
    cfg["parallel"] = par
    if mode == "draft":
        out = out + ["--batch-groups", "1", "--batch-mtp"]
    # pipe: 43.1 默认就是 pipelined 组, 不加参数
else:
    cfg.pop("parallel", None)
cfg["args"] = out
after = json.dumps(cfg, ensure_ascii=False, indent=1) + "\n"
if open(path, encoding="utf-8").read() != after:
    with open(path, "w", encoding="utf-8") as f:
        f.write(after)
if par > 1:
    print(f"parallel: = {par} 流 mode={mode} " + ("(--batch-groups 1 + --batch-mtp, slot 有 draft)" if mode == "draft" else "(默认 pipelined 组, slot 无 draft)"), flush=True)
else:
    print("parallel: 单流 (parallel/--batch-groups/--batch-mtp 已清除)", flush=True)
PYP

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export LD_LIBRARY_PATH="$CUDA_PIP/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset STRATA_ARENA_PIN_GIB
unset STRATA_RESIDENT_PIN
unset STRATA_STAGE_PIN
unset STRATA_GDN_CHUNKED
export STRATA_PREFILL_HELP="${STRATA_PREFILL_HELP:-1}"
export STRATA_PREFILL_CPU_SHARE="${STRATA_PREFILL_CPU_SHARE:-auto}"
export STRATA_PREFILL_CPU_SHARE_MAX="${STRATA_PREFILL_CPU_SHARE_MAX:-3072}"
export STRATA_STAGE_TRIM="${STRATA_STAGE_TRIM:-1}"
cd "$ROOT"
echo "Strata Swift abliterated IQ3_S + Q8_0 PLE  gpus=$GPUS  split=26  ctx=$CTX  kv=$KV  vision(gpu)  MTP=swift  PLE=sleepyeldrazi-Q8  resident-experts (GPU then RAM)  prefill=$PREFILL  reserve=${VRAM_RESERVE}  pipeline-windows=${PIPELINE_WINDOWS:-2}  min_p=$SPEC_MIN_P  PREFILL_HELP=${STRATA_PREFILL_HELP}  CPU_SHARE=${STRATA_PREFILL_CPU_SHARE}/${STRATA_PREFILL_CPU_SHARE_MAX}  STAGE_TRIM=${STRATA_STAGE_TRIM}  host=0.0.0.0  fit_max_tokens  port=$PORT" >&2
if [ "${JSON_ONLY:-0}" = 1 ]; then
  echo "JSON_ONLY=1: json 已按 parallel=${PARALLEL} 写好, 不启动引擎" >&2
  exit 0
fi
exec bash "$RUN"
