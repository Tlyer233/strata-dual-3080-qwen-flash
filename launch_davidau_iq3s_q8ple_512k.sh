#!/bin/bash
# DavidAU Qwen3.8-Flash-Next Abliterated Uncensored Turbo Brilliance 50x GSQ-RCO IQ3_S, 本机 3080 双卡.
# 抄 launch_swift_abliterated_iq3s_q8ple_512k.sh, 只换模型/pack/mmproj + RUN/CFG (2026-10-20).
# n-gram 用 sleepyeldrazi Q8_0 PLE (--ple-gguf), 不用它自带 IQ4_NL shard 2 (本机盘装不下 28.8 G,
#   且 iq_pack.py 要 shard2 才肯跑 -> pack 直接从 4090 rsync 过来, 1.5 G / 13 s @111 MB/s).
# Dual-GPU: gpus=1,0 split=26, prefill 8192, adaptive MTP, pipeline-windows 2.
# 512K 版: ctx 524288 + YaRN x2 (--rope-scaling yarn --rope-scale 2, 官方 >262K 口径), 停放 3 槽.
# 依据: Strata DETAILS.md yarn 实测 (IQ3_S: 293K NLL 2.04 vs 裸 2.22, <=32K 无退化, 421K yarn2 QA 8/8).
# 温度: 不特殊处理, apply_adaptive_mtp.py 每次写卡片 thinking 值 1.0/0.95/20 + force=true (用户定: 其他默认).
# effort_position=end 写在 json 里 (只改 effort 的请求不重读, 实测 9528 reused / 0.2 s).
set -euo pipefail
ROOT="${STRATA_ROOT:-$HOME/src/Strata}"
PORT="${PORT:-8005}"
CTX="${CTX:-524288}"
PREFILL="${PREFILL:-16384}"
VRAM_RESERVE="${VRAM_RESERVE:-1024}"
SPEC_MIN_P="${SPEC_MIN_P:-0.5}"
KV="${KV:-k8v4}"
GPUS="${GPUS:-1,0}"
MTP="${MTP_DIR:-$HOME/src/Strata-data/mtp/rt}"
PLE="${PLE_GGUF:-$HOME/models/sleepyeldrazi/Qwen3.8-Flash-Next-GSQ-RCO-Q8_0-PLE-GGUF/Qwen3.8-Flash-Next-GSQ-RCO-Q8_0-PLE-00002-of-00002.gguf}"
ROPE_SCALING="${ROPE_SCALING:-yarn}"
ROPE_SCALE="${ROPE_SCALE:-2}"
RUN="${RUN:-$ROOT/run-davidau-iq3s-q8ple.sh}"
CFG="${CFG:-$ROOT/strata-davidau-iq3_s-q8ple.json}"
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
#
# --- 旗杆(检查点) vs 停放(slot), 别混 ---
# 停放 --conversation-cache-slots: 整段会话 KV 拷到内存, 换会话用. 本机 3 槽.
# 旗杆 --prompt-cache: 同一段对话路上的存档, 分叉时退到最近一根再走.
#   官方口径 (~118 MB 宿主 RAM / 根, 不占显存). KV 仍只在 GPU 里留当前那一局.
#   本机 14 根 ≈ 1.6 GiB RAM. 多插几根 = 加大 PROMPT_CACHE_N, 只吃内存.
#   多插几根不会缩短 128K 间隔 (间隔是 EVERY); 只避免 3 段长会话把彼此的旗杆挤掉.
#   把 EVERY 改密 (8K/16K) 才能让分叉少走, 但 3 段 512K 会抢光池, 换会话又变慢.
#
# --- Codex 插话整段 prefill (2026-10-18) ---
# Codex 插话 = 掐断当前请求, 丢掉助手半截, 把新句子接在上一句完整话后面重发.
# 这是尾部分叉, 不是追加. live/停放对不上, 只能退到最近旗杆, 最坏走满 128K (~1 min).
# 别的 agent 只在末尾追加, 前缀命中, 所以只有 Codex 炸.
#
# --- --prompt-cache-tail: 尾巴再插一根旗杆 (Codex 插话 / 尾部分叉) ---
# 人话: 每次快读完时, 在尾巴再插一根旗杆. 插话通常拐在尾巴, 下一轮只重读最后
#   几千 token (一个 --prefill chunk, 本机 8192), 不是 128K.
# 官方 generate.cpp 把这条写死 !multi_gpu (0.1.45 仍是) -> 双卡官方吃不到。本机 2026-10-18 自己
#   打过补丁 (去掉 !multi_gpu, 双卡各 stage 在尾巴 chunk 也存 part)，实测 3080×2 split=26,
#   prompt 22141 < every 131072: A 0 reused + 22141 / 11.4 s; B 同前缀换最后一句 16384 reused
#   + 5757 / 4.9 s。看着漂亮，其实收效甚微: 只有"prompt 短于 every 且正好拐在尾巴 chunk 边"
#   这一格省时间, 真实 Codex 插话的 prompt 长度不固定, 长 prompt 照样退到 128K 周期旗杆。
#   2026-10-18 决定: 补丁撤了 (0.1.45 起不打), 等官方支持多卡尾巴检测点。
#   补丁存档在 sdd3080 仓的 git 历史 (patches/prompt-cache-tail-layer-split.patch, commit 16c3762)。
# 4090 单卡官方就能开 tail, 那边不受这条限制。
# 默认 0 = 不往 json 写 --prompt-cache-tail; 官方哪天支持多卡, PROMPT_CACHE_TAIL=1 直接开回来。
# --conversation-cache-min-tokens 4096 (官方 PR #1597): 短请求不再占停放槽把长会话顶掉
#   (那条实测: 3 条几百 token 的短请求顶掉 19198 token 长会话 -> 41.4 s 重读; 设门槛后 0.3 s, 304x).
#
# --- 2026-10-18 第二轮: 3080 整段重读的真凶是 RAM 闸门, 不是旗杆 ---
# 逐字翻本机日志: skip parking 244 次, "= 0 reused" 537 次, evictions 累计 80,
#   最惨一条 prompt 500187 = 0 reused + 500187 read in 190.7 s (2622 t/s)。
# 账: 一条 500K 快照 snapshot_bytes=6.93 GB (13.85 KB/token), 而 available 只有 4.4 GB
#   -> 物理 RAM 闸门 (generate.cpp:8972) 直接拒停放, 官方注释写明拒了就等于把这段读取全扔。
#   闸门公式: 快照 MiB + 地板 MiB <= MemAvailable。地板 4096 时一条 500K 要 11.0 G 余量, 两条 18.0 G。
# RAM 大头是 --resident-experts 的专家 RAM 层: 实测 31.63 GiB, 引擎 RSS 35.6 GB, 总 64.2 GB。
#   官方源码 530/2604 行: 这层吃满到只剩 resident_headroom, 默认 4 GiB —— 正好就是 available 4.4 GB。
#   官方留了 A/B 臂 STRATA_RESIDENT_HEADROOM_GIB=N (源码 2609 行) = 给停放腾地方, 代价是专家命中率。
# 所以本机: headroom 4->20 GiB (专家层 31.63 -> ~15.6 GiB), 槽 3->2, 地板 4096->2048
#   -> 2 条 500K 停放 13.9 G + 旗杆 2.7 G = 16.6 G <= 20 G 闸门, 停得住, 不再整段重读。
# tail 本机仍吃不到 (官方写死 !multi_gpu, 补丁已撤); 但 spread 没被 gate (走同一条 spread_victim)
#   -> 双卡直接开: 池满时保留摊开的旗杆, 改写历史 N token 就只重读 ~N。
#   池 14 对 500K 会话太小 (4 根周期 + ~20 根轮次 > 14), 尾巴附近的根先被逐 -> 14->24 (114 MB/根, 2.7 G)。
# --- 2026-10-19 回退: 尾巴旗杆 / 摊开保留 / 停放槽 全删, 只留 live 续读 ---
# 只写 --prompt-cache N + --prompt-cache-every E: 这是当前会话的 live 续读, 正常对话不重读全靠它。
#   源码 generate.cpp:14483  live_ok = o.prompt_cache > 0 && req_ckpt  ->  设成 0 就是每轮 token 0 全量 prefill。
# 停放写 --conversation-cache-mib 0: 不再把整段 KV 拷进宿主 RAM, 不占槽, 不驱逐。
# 上面 DROP_PREFIX 把 json 里所有 --conversation-cache* / --prompt-cache* 旧值一次清干净再写回。
# 旗杆官方 6x16384。停放：切 slot 用，mib=0 就是换会话全量 prefill。3080 三槽。
PROMPT_CACHE_N="${PROMPT_CACHE_N:-6}"
PROMPT_CACHE_EVERY="${PROMPT_CACHE_EVERY:-16384}"
CONV_CACHE_MIB="${CONV_CACHE_MIB:-16384}"
CONV_CACHE_SLOTS="${CONV_CACHE_SLOTS:-3}"
CONV_CACHE_MIN_TOKENS="${CONV_CACHE_MIN_TOKENS:-4096}"
CONV_CACHE_MIN_FREE_MIB="${CONV_CACHE_MIN_FREE_MIB:-2560}"
CCPY_BIN="$ROOT/.venv/bin/python"
[ -x "$CCPY_BIN" ] || CCPY_BIN="$(command -v python3)"
"$CCPY_BIN" - "$CFG" "$PROMPT_CACHE_N" "$PROMPT_CACHE_EVERY" \
  "$CONV_CACHE_MIB" "$CONV_CACHE_SLOTS" "$CONV_CACHE_MIN_TOKENS" "$CONV_CACHE_MIN_FREE_MIB" <<'PYCACHE'
import json, os, shutil, sys
path, n, every, mib, slots, mintok, minfree = sys.argv[1:8]
DROP_PREFIX = ("--conversation-cache", "--prompt-cache")
with open(path, encoding="utf-8") as f:
    cfg = json.load(f)
old = list(cfg.get("args") or [])
args, i = [], 0
while i < len(old):
    if any(old[i].startswith(p) for p in DROP_PREFIX):
        i += 1
        if i < len(old) and not str(old[i]).startswith("-"):
            i += 1
        continue
    args.append(old[i])
    i += 1
args += [
    "--prompt-cache", n, "--prompt-cache-every", every,
    "--conversation-cache-mib", mib,
    "--conversation-cache-slots", slots,
    "--conversation-cache-min-tokens", mintok,
    "--conversation-cache-min-free-mib", minfree,
]
if args != old:
    if not os.path.exists(path + ".bak-revert"):
        shutil.copy2(path, path + ".bak-revert")
    cfg["args"] = args
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
        f.write("\n")
print("cache:", " ".join(args[j] + (" " + args[j + 1] if j + 1 < len(args) and not str(args[j + 1]).startswith("-") else "")
                         for j, a in enumerate(args)
                         if a.startswith("--prompt") or a.startswith("--conversation")))
PYCACHE
echo "cache: --prompt-cache ${PROMPT_CACHE_N} --prompt-cache-every ${PROMPT_CACHE_EVERY} --conversation-cache-mib ${CONV_CACHE_MIB} --conversation-cache-slots ${CONV_CACHE_SLOTS}" >&2


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
# 专家 RAM 层给停放让路 (官方 A/B 臂, 源码 2609 行; 默认 4 GiB -> 本机 available 只剩 4.4 G,
#   一条 500K 快照 6.93 G 停不下 -> 0 reused 整段重读 190 s)。decode 命中率掉多少要实测。
export STRATA_RESIDENT_HEADROOM_GIB="${STRATA_RESIDENT_HEADROOM_GIB:-20}"
export STRATA_PREFILL_HELP="${STRATA_PREFILL_HELP:-1}"
export STRATA_PREFILL_CPU_SHARE="${STRATA_PREFILL_CPU_SHARE:-auto}"
export STRATA_PREFILL_CPU_SHARE_MAX="${STRATA_PREFILL_CPU_SHARE_MAX:-3072}"
export STRATA_STAGE_TRIM="${STRATA_STAGE_TRIM:-1}"
cd "$ROOT"
echo "Strata DavidAU GSQ-RCO IQ3_S + Q8_0 PLE  gpus=$GPUS  split=26  ctx=$CTX  rope=${ROPE_SCALING}x${ROPE_SCALE}  kv=$KV  vision(gpu)  MTP=rt  PLE=sleepyeldrazi-Q8  resident-experts (GPU then RAM)  prefill=$PREFILL  reserve=${VRAM_RESERVE}  pipeline-windows=${PIPELINE_WINDOWS:-2}  min_p=$SPEC_MIN_P  PREFILL_HELP=${STRATA_PREFILL_HELP}  CPU_SHARE=${STRATA_PREFILL_CPU_SHARE}/${STRATA_PREFILL_CPU_SHARE_MAX}  STAGE_TRIM=${STRATA_STAGE_TRIM}  host=0.0.0.0  fit_max_tokens  port=$PORT" >&2
if [ "${JSON_ONLY:-0}" = 1 ]; then
  echo "JSON_ONLY=1: json 已按 parallel=${PARALLEL} 写好, 不启动引擎" >&2
  exit 0
fi
exec bash "$RUN"
