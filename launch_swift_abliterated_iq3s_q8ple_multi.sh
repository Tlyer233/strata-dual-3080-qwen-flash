#!/bin/bash
# 多流版 (parallel N slots, 默认 2), 其余与 launch_swift_abliterated_iq3s_q8ple.sh 完全一致
# (同款停放参数, 同款 Q8_0 PLE, 同款 MTP, 双卡 gpus=1,0 split=26)。
#
# 用法:
#   bash ~/ubuntu_handoff/strata/launch_swift_abliterated_iq3s_q8ple_multi.sh        # parallel 2 启动 (~3 分钟)
#   bash ~/ubuntu_handoff/strata/launch_swift_abliterated_iq3s_q8ple_multi.sh 3      # parallel 3
#   JSON_ONLY=1 bash ~/ubuntu_handoff/strata/launch_swift_abliterated_iq3s_q8ple_multi.sh   # 只写 json 不启动
#
# 代价 (本机 双 3080 实测, 0.1.42/0.1.43 一致, 0.1.43 官方 BATCHING.md 同口径 10-25%):
#   - 每 slot ~1.55 GiB 显存/stage (专家热层 -22%) + ~11.6 GiB pinned RAM
#   - slot 没有 MTP draft: 两路同时在打字时各 ~9-11 tok/s; 占 slot 时单请求 66-74 (纯单流 93)
#   - 256K x2 双冷 prefill 互踩: 149.9 s + 387.1 s (单流同长 ~150 s)
#   - launcher 已强制 --batch-groups 1: 双卡 auto = 2 groups of 1 slot 走流水慢路径 (issue #1906 实测慢)
#   - #1503 (slot 并发 + 停放可能串上下文) 0.1.42/0.1.43 二进制都没补丁, 结果抽查
# 回单流: bash launch_swift_abliterated_iq3s_q8ple.sh   (默认 PARALLEL=1, 启动时自动清 json)
set -euo pipefail
export PARALLEL="${1:-${PARALLEL:-2}}"
exec bash "$HOME/ubuntu_handoff/strata/launch_swift_abliterated_iq3s_q8ple.sh"
