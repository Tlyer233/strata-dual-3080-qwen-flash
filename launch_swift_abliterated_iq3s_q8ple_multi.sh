#!/bin/bash
# 多流版 (parallel N slots + BATCH_MODE), 其余与 launch_swift_abliterated_iq3s_q8ple.sh 完全一致
# (同款停放参数, 同款 Q8_0 PLE, 同款 MTP, 双卡 gpus=1,0 split=26, PREFILL=8192).
#
# 用法:
#   bash ~/ubuntu_handoff/strata/launch_swift_abliterated_iq3s_q8ple_multi.sh            # parallel 2 + draft (默认)
#   bash ~/ubuntu_handoff/strata/launch_swift_abliterated_iq3s_q8ple_multi.sh 2 pipe     # 默认 pipelined 组 (流多时用)
#   JSON_ONLY=1 bash ~/ubuntu_handoff/strata/launch_swift_abliterated_iq3s_q8ple_multi.sh [N] [draft|pipe]
#       # 只写 json 不启动不杀引擎; 正常流程 = JSON_ONLY 写好 -> systemctl --user restart strata-swift (~3 分钟)
#
# BATCH_MODE (env 传给 launcher, PARALLEL>1 才生效; 0.1.43.1 语义, #2038 已修):
#   draft: args 加 --batch-groups 1 + --batch-mtp -> 每 slot 有 MTP draft (串行 draft 窗).
#          官方原话: 流少 / prompts 稀疏时这反而可能更快. 本机 2 流 = 默认选它.
#   pipe:  不加参数, 吃 43.1 默认 pipelined 组 (slot 无 draft; 官方 4 流/2 GPU 比串行窗 +25-30%).
#
# 代价 (本机 0.1.43.0 实测, 无 draft 模式): 每 slot ~1.55 GiB 显存/stage (专家热层 -22%) + ~11.6 GiB pinned RAM;
#   两路同时打字各 9-11 tok/s; 被占 slot 的单请求 66-74 (纯单流 93; 43.1 纯单流 101-152).
#   开 2 流前想好停放: 62 G RAM 里 2 slot ≈ 23 G, 停放预算/条数要让位.
# 已知风险 (至 0.1.43.1): #1503 slot+停放可能串上下文 (抽查各会话 marker); #1603 双长上下文会话并发可能挂死
#   (判据: 引擎日志 running:0 queued:0 且两分钟不动). 43.1 已修 #1994 (slot 400 -> 503 retry).
# 回单流: bash launch_swift_abliterated_iq3s_q8ple.sh   (默认 PARALLEL=1, 启动时自动清 json)
set -euo pipefail
export PARALLEL="${1:-${PARALLEL:-2}}"
export BATCH_MODE="${2:-${BATCH_MODE:-draft}}"
exec bash "$HOME/ubuntu_handoff/strata/launch_swift_abliterated_iq3s_q8ple.sh"
