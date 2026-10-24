# AGENT: strata-dual-3080-qwen-flash

Reproduce the dual RTX 3080 20GB + 64 GB Strata launches in this folder. This file is for an agent, not a human. Paths below are the reference PC unless remapped.

**当前状态（2026-10-22 晚）**：两台（3080 双卡、4090 单卡）都跑 **Strata v0.2.2 + 512K**，版本只剩这一套（旧的 0.1.4x 树全部进 `~/src/old/`）。
引擎 = 官方 tag `v0.2.2` 源码编译；**该 tag 的 `CMakeLists.txt` 写的是 `project(strata VERSION 0.1.47)`，所以 `strata --version` 打 0.1.47**，
认版本要看源码树 + sha256（见「Strata 0.2.2」一节）。跑法：ctx 524288 + `--rope-scaling yarn --rope-scale 2`、k8v4、`--resident-experts`、
`--pipeline-windows 2`、单流、vision gpu、`effort_position: "end"`、旗杆 `6×16384`，tail / spread 都没开。
模型仍是 **DavidAU Qwen3.8-Flash-Next GSQ-RCO abliterated IQ3_S**，n-gram 仍用外挂 **sleepyeldrazi Q8_0 PLE**（`--ple-gguf`），MTP 沿用 `mtp/rt`。
引擎补丁三个：**#1233 PLE（不打启动即崩）**、**same-chat rewind（Codex 插话不再占停放槽，见「same-chat rewind」一节）**、**Codex 旁路请求与同线程替换（2026-10-24，见下面那一节）**。
serve 也一并升到 0.2.2，并把「钉 sampling」补丁搬过去（`patches/port_pin_022.py`，v0.2.2 的锚点和 0.1.38/0.1.45 那份不一样）。
2026-10-24 这一轮只改了源码树和运行目录的 `serve/server.py`，**没重编、没重启**（3080 GPU 在跑数据）。在跑的 `engine/strata` 仍是 2026-10-22 那份：有 PLE 和 same-chat rewind，没有 `dropped %zu replaced`。槽数不动，3080 仍是 2。
停放：3080 `--conversation-cache-mib 24576 / slots 2`，4090 `24576 / slots 4`，两边 `min-tokens 18432`、`min-free 2560`。
温度：4090 钉死 `0.9 / 0.95 / 20`（json 顶层 `sampling` + `sampling_force`，客户端传什么都压不过）；3080 的 json 没这两键 → 不钉。
开机：3080 = drop-in `~/.config/systemd/user/strata-swift.service.d/override.conf` 指 `launch_davidau_iq3s_q8ple_512k.sh`；4090 = user unit 本体同一份脚本。
本轮 512K 的 decode / prefill 数还没重测；**0.2.2 上已测的是 256K 那轮**（decode AVG 94.6、256K 冷读 78.1 s / 3195 t/s，
`results/davidau_iq3_256k.json`，同 ctx 的 0.1.43.1 基线 85.0 / 85.6 s），口径写在下面 0.2.2 那节里。

**IQ3_XXS** (`launch_swift_iq3xxs.sh`): full RAM arena, ~70–80 tok/s decode on this box. Quant fits 64 GB.

**IQ4_XS** (`launch_swift_iq4xs.sh`): dual-GPU **VRAM first, then RAM**. Upstream Strata v0.1.38 does not merge this yet. Method, patch, and citations: **[RESIDENT_ON_SPLIT.md](RESIDENT_ON_SPLIT.md)** ([Niko1221/Strata#604](https://github.com/Niko1221/Strata/issues/604), [#364](https://github.com/Niko1221/Strata/issues/364), [#384](https://github.com/Niko1221/Strata/issues/384), [PR #639](https://github.com/Niko1221/Strata/pull/639), [PR #640](https://github.com/Niko1221/Strata/pull/640), [PR #731](https://github.com/Niko1221/Strata/pull/731)).


## 版本实测（按引擎版本；每版本 = model / ngram / script / 该版本实测表）

### Strata 0.2.2（当前在用；2026-10-21 起官方 tag `v0.2.2` 源码编译，带 #1233 PLE + same-chat rewind 两个引擎补丁）

* **版本命名坑（先说，别对着 `--version` 找 0.2.2）**：官方 tag `v0.2.2` 的 `CMakeLists.txt:11` 还是 `project(strata VERSION 0.1.47)`，
  `STRATA_VERSION` 取 `PROJECT_VERSION` → `strata --version` 打 **0.1.47**；`engine/BUILD.json` 里甚至是上一版留下的 `"version": "0.1.45"`
  （换二进制时 `src=skip`，那个文件没重写）。认版本的可靠办法：比对源码树 + sha256，或者看引擎日志里最后一次 `engine started` 之后那行
  `session is up (engine 0.1.47)`（**重启判健康也别只看 `/v1/models` 的 `n_ctx`：换 ctx 之前那个数就已经在那儿了**）。
  2026-10-22 21:2x 两台在线二进制（都已含 same-chat rewind）：3080 `engine/strata` = `~/src/Strata-0.2.2/build/strata` = `bf5b11c5640a…`，
  `engine/strata-vision` = `build-vision/bin/strata-vision` = `6ffc8eac7459…`；4090 `engine/strata` = 它那棵 0.2.2 树的 `build/strata` = `4e9dc7daa1f2…`。
* **换法**：`v0.2.2.zip` 走 gh-proxy 解到 `~/src/Strata-0.2.2` → **`patches/ple-only-native-dense-v0.1.43.1.patch` 还得打**（0.2.2 依旧没修 #1233，不打启动即崩
  `duplicate split shard number`），但**行号已挪**：补丁 hunk 头是 `generate.cpp` 第 2551 行，0.2.2 树里 `ple_only` 那段在 3225 行，直接 `patch` 会拒，按 `ple_only` 逐字贴进去即可
  → `cmake -G Ninja -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 …` → `ninja` 出 `strata` + `build-vision`（先 unlink 再拷，运行中的 inode 会「文本文件忙」）。
  同一个 0.2.2 树还要打 **same-chat rewind**（在 #1233 之后、重编之前）：
  `python3 -c "import pathlib,importlib.util as iu;s=iu.spec_from_file_location('al','patches/apply_local.py');m=iu.module_from_spec(s);s.loader.exec_module(m);m.apply_same_chat_rewind(pathlib.Path.home()/'src/Strata-0.2.2')"`
  → 打印 `patched generate.cpp: same-chat rewind`，`grep -c same_chat_rewind src/program/generate.cpp` 应为 4。
  今天没跑整份 `apply_local.py`：它开头还要改 `setup.py` 和 0.1.38 时代的 serve 锚点，跟已经钉过 sampling 的 0.2.2 serve 会撞，所以只单独调这一个函数（两个锚点 `if (borrow && !pin_sibling && park_worth) {` 和那个 `} else if ((!from_live || incoming || slot_source >= 0) && !pin_sibling && park_worth &&` 在 0.2.2 里各恰好 1 处，函数会自己核对）。
  之后 `cmake --build build --target strata -j $(nproc)` 增量重编（只改 `generate.cpp`：3080 16 核约 3 分钟，4090 18 核约 2 分钟）。
* **换 serve 的坑（2026-10-22 本机真踩过）**：v0.2.2 把进程看护从 `serve/` 拆成**树根的 `strata_control` 包**，`serve/server.py:6224` 还要延迟 `import strata_cli.client`。
  只把 `serve/` 拷进运行目录、或把运行目录里"看着不像运行用"的东西搬走，就会 `ModuleNotFoundError: No module named 'strata_control'`，8005 直接没人听（而且启动脚本已经把旧引擎停了，两头空）。
  搬法：`ln -s ../Strata-0.2.2/strata_control` 和 `ln -s ../Strata-0.2.2/strata_cli` 到运行目录，然后
  `cd <运行目录> && .venv/bin/python -c "import serve.server, strata_cli.client"` **必须 rc=0** 才算完（`import serve.server` 单跑不算数，看护那个 import 在函数里，跑得起来才炸）。
  钉 sampling 那份补丁在 v0.2.2 上锚点全变了，用 `patches/port_pin_022.py`（幂等，四个锚点各必须恰好 1 处，对不上就拒绝动手；`python3 patches/port_pin_022.py` 打印 `patched … py_compile 过`，再跑一遍打印 `skip …`）。
* **冷启到可对话的耗时**：256K 无 YaRN 约 **2 分钟**（日志逐字：`FileExpertSource: mapped pinned cache complement ready: resident 32.48 GiB, pinned 32.48 GiB in 56 s`，
  光 page-locked 常驻专家层就 56 s）；512K + YaRN×2 双卡 3080 实测 **120-128 s**、单卡 4090 **36 s**（模型页还在 page cache 时）。0.1.45 @512K 那版模型加载是 235 s。
* **现在两台都跑 512K**（2026-10-22 晚切回）：`launch_davidau_iq3s_q8ple_512k.sh`，ctx 524288 + `--rope-scaling yarn --rope-scale 2`、k8v4、`--resident-experts`、`--pipeline-windows 2`、单流、vision gpu。
  `PREFILL` 4090 **32768**（单卡 sm_89）、3080 **16384**（双卡 `gpus 1,0` + `layer_split 26`）；停放两边 `--conversation-cache-mib 24576`、`min-tokens 18432`、`min-free 2560`，槽 **4090 = 4 / 3080 = 2**（3080 定 2 的理由见「same-chat rewind」那节的停放实测）；旗杆仍 `6×16384`，tail / spread 都没开。
  256K 那版（ctx 262144、`--rope-scaling none`、停放 `20480/5`）是同一条模型的另一种切法，脚本还在：`launch_davidau_iq3s_q8ple_256k.sh`，实测表在下一节。
* **script**：[launch_davidau_iq3s_q8ple_256k.sh](launch_davidau_iq3s_q8ple_256k.sh)（从 512K 版派生，只改 8 处：`CTX` / `ROPE_SCALING` / `ROPE_SCALE` / run 名 / json 名 / `CONV_CACHE_MIB` / `CONV_CACHE_SLOTS` / echo 文案），
  json 快照 [example-strata-davidau-iq3_s-q8ple-256k.json](example-strata-davidau-iq3_s-q8ple-256k.json)。开机单元已改指 **512K** 那份（2026-10-22 晚：3080 是 drop-in `~/.config/systemd/user/strata-swift.service.d/override.conf`，4090 是 user unit `strata-swift.service` 本体）。

#### same-chat rewind：Codex 插话不再占停放槽（2026-10-22，两台在线都有）

**问题**（今天 4090 日志逐字，补丁前）：一条 16 万 token 的会话，插话那一次 `prompt 162841 tokens = 0 reused + 162841 read in 33647 ms`，
紧接着 `conversation cache: parked 162256 tokens … parked=1 bytes=2839379148`，再插一次就 `parked=2`。**每插一次话就多停一条僵尸会话**，
5 槽很快被这些 fork 占满（4090 那份引擎 log 第 1911 行已见 `evictions=2`）。停一条 452K 的快照实测 6.00 GiB（`snapshot_bytes=6445805784`），500K 是 6.57 GiB（`7053600632`），
约 **13.9-15.7 KB/token**（旧记录里 18.5-35 KB/token 是 0.1.45 时代 ctx 524288 + 池 14 那版，别混用）。

**补丁**：`patches/apply_local.py` 里的 `apply_same_chat_rewind()`，打 `src/program/generate.cpp` 的停放判定。判据逐字：
`!from_live && !incoming && slot_source < 0 && resume > 0 && !checks.empty()`，且 `resume` 恰好等于本链某个 checkpoint 的长度、且 `resume > checks.front().ids.size()`（链根）
→ 这是**同一条会话回退**（Codex 插话：掐断当前请求、丢掉助手半截、接在上一句完整话后面重发），不是新会话，于是跳过 park 并打一行
`conversation cache: same-chat rewind at N (root R); not parking`；另一处 else 分支加 `!same_chat_rewind`。别的会话照旧 park，借停放快照那条路径没动。

**实测（3080，v0.2.2+该补丁，2026-10-22 21:2x）**

```
1) 建会话     prompt=56546  cached=0      重读=56546  wall=26.3s
2) 退一步     prompt=51751  cached=49152  重读=2599   wall=3.15s
   conversation cache: same-chat rewind at 49152 (root 16384); not parking
   conversation cache: skip parking (15172 tokens, below the 18432 minimum; parked=2)
```

退一步命中的是 49152 那根旗杆（`every=16384`），重读 2599 token / 3.15 s，**停放计数没涨**。4090 那边 19:18-21:0x 这段真跑了 4 次，逐字
`same-chat rewind at 29157 (root 26129); not parking`、`same-chat rewind at 264305 (root 18684); not parking`；3080 那台同期 8 次。

**配套的门槛**：`--conversation-cache-min-tokens` 4096 → **18432**（`launch_davidau_iq3s_q8ple_512k.sh` 里注释写清了理由：完整工具表的链根约 18684 仍占格，
`get_goal` 那种 ~15k 的短工具表不占格）。同一条会话的插话不靠这条门槛，靠上面的 same-chat rewind。

**停放能停几条（3080 62 G、常驻专家 32.48 GiB，2026-10-22 实测）**

```
预算 24576 / 槽 2      MemAvailable     SwapFree     结果
------------------------------------------------------------------------------
停 1 条 (452K)          14.8 GiB        0.1 GiB     park 3.8 s，OK
停 2 条 (452K x2)        8.8 GiB        0.0 GiB     park 4.5 s，OK（parked 共 14.24 GiB）
想停第 3 条              3.6 GiB        0.0 GiB     门槛要 6.0 GiB + 2560 MiB floor -> 不给，
                                                    且第 3 条的 prefill 从 153 s 拖到 >170 s（整机在抖）
```

所以 3080 定 `slots 2`（留 3 槽只会让第 3 条 park 把 swap 顶满、整机抖，然后马上被 LRU 逐出）；4090 停 4 槽但同样受 RAM 门槛约束。
停放命中时回访是秒回，逐字：`conversation cache: restored 451864 tokens (checkpoint) in 651.0 ms`，整次切换 wall **1.06 s**（重读 5 token）；
A↔B 交替 3 轮 6 次数都在 **1.34-1.58 s**，重读 23 token。对照新建一条 452K = **153.0 s**（2953 t/s），差 100-150 倍。

#### Codex 旁路请求与同线程替换（2026-10-24）

针对 Codex。dsh 不发 `conv=`，也不发 `ckpt=0`，这条对它不发生。3080 槽数保持 **2**，不抄 4090 的 4。`--conversation-cache-min-tokens` 仍是 18432。

same-chat rewind 只盖住「插话的切点正好落在本链某根旗杆上，且比链根深」。compact 把正文换成另一段、subagent 各自一条线程时，旧引擎把它们当成新会话整段停进去。短快照占满槽之后，最老的那条被逐出。

补丁是 `patches/apply_codex_side_request.py`。前提：这棵树已经打过 same-chat rewind。幂等。每个锚点必须恰好 1 处，否则退出，不写文件。半截打过（有的标记在、有的不在）也退出。不编译、不启动。

```bash
python3 patches/apply_codex_side_request.py ~/src/Strata-0.2.2 ~/src/Strata/serve/server.py
```

第二个参数是运行目录那份 `serve/server.py`。它和源码树不是同一个文件，两份都要有 `conv=`。只打源码树时不传第二个参数。2026-10-24 本机已经跑过这行：源码树四份都是 skip，运行目录的 `serve/server.py` 打上了。

脚本改这几处：

* `serve/server.py` 的 `sampling_keys`：`strata_conversation` 是字符串、长度 1–128、字符属于 `[A-Za-z0-9._:-]` 时，GEN 行拼 `conv=<id>`。不合规则不拼。`strata_checkpoint is False` 仍拼上游已有的 `ckpt=0`（`#830`），这个含义不改。Python 的 `False` 才拼；缺字段、`true`、`0` 都不拼。
* `include/strata/core/conversation_cache.hpp`：`SavedConversation` 加 `std::string conv;`，空字符串表示不知道是哪条。不写进磁盘会话文件。`drop_replaced(conv, prompt, images)`：id 为空返回 0；否则丢掉 `conv` 相等、且 `conversation_prefix(live, prompt) != live.ids.size()` 的条目（停放的 token 不是这条新提示词的严格前缀）。正在被借走的 donor（`guard`）不删。
* `src/program/generate.cpp`：活会话旁加 `live_conv`。解析 GEN 键 `conv=`。`park_current` 里 `put` 之前 `image.conv = live_conv`。`conversations.best` 之前，仅 `req_ckpt && !req_conv.empty()` 时调用 `drop_replaced`，删了就打 `dropped N replaced snapshot(s) (conv ...); parked=`。请求正常结束且 `live_ok` 为真：`live_conv = req_conv`。`ckpt=0` 时 `live_ok` 为假，不改 `live_conv`。取消但保留活会话的那条（`#1620`）同样写 `live_conv`。
* `src/core/conversation_cache_test.cpp`：同线程的延长留着，换正文丢掉，另一个线程不动，空 id 不丢。

一次 compact 之后槽里有什么：

1. 父对话是活 KV，`live_conv` 是它的 thread id。
2. compact 请求 `strata_checkpoint=false`，所以是 `ckpt=0`。父对话仍 `live_ok`，按原逻辑停进 1 格，快照带上父对话的 id。这条跑完 `live_ok=false`。
3. 后面的摘要请求也是旁路：`park_current` 看到 `!live_ok`，不入槽。
4. 压缩后的正式下一轮：同一个 id，提示词不是旧快照的延长。`drop_replaced` 在 `best()` 之前丢掉旧快照。新正文进显存，不新占一格。

一次 subagent 之后槽里有什么：subagent 正文不带 `strata_checkpoint: false`，`conv=` 是子线程自己的 id。第一条到来时父对话停 1 格。再切走时，子线程停进另外 1 格。切回来，新提示词延长那一格里的 token，`drop_replaced` 不删，换回显存。同一个子线程里的 compact、预热、标题仍是旁路，不另占格。

没有 `conv=` 时，旁路仍然不入槽，但第 4 步不会发生，压缩前的快照会留到槽满才被挤走。

插话不靠本补丁。插话没有旁路标记，由 same-chat rewind 在同一条链的检查点上跳过停放。切点没有落在检查点上时，这条对话还会再停一格。那一格不是从 token 0 重读，匹配到的前缀留着，只重读最近旗杆之后的尾巴。

**这次没换二进制。** 3080 GPU 在跑数据，没有 `ninja`，没有覆盖 `~/src/Strata/engine/strata`，没有重启。那份二进制的 mtime 是 2026-10-22 21:20，`strings` 里 `same-chat rewind` 有 1 处，`dropped %zu replaced` 有 0 处。运行目录的 `serve/server.py` 现在会拼 `conv=`；旧引擎跳过不认识的 GEN 键，所以在跑的进程不受影响。GPU 空了之后再增量重编（只动过 `generate.cpp` 和头文件的话，3080 16 核大约 3 分钟），先停掉占用端口的进程再拷贝，运行中的 inode 会「文本文件忙」。`pkill -x strata` 可能杀不掉，进程名有时是 `strata-host`。重编之后日志里才有 `dropped N replaced snapshot (conv <id>); parked=`。

#### 0.2.2 + DavidAU IQ3_S + 无 YaRN @ 256K（已被 512K 取代；停放 5 槽 / 20480 MiB，`PREFILL=16384`，`layer_split 26`，单流；bench 2026-10-22 09:39，[results/davidau_iq3_256k.json](results/davidau_iq3_256k.json)）

（**已被 512K 取代**，见开头「当前状态」；这轮是 0.2.2 上目前唯一测过的 sweep，保留做版本口径。）

* **model**：https://huggingface.co/DavidAU/Qwen3.8-Flash-Next-Abliterated-Uncensored-Turbo-Brilliance-50x-Reasoning-Instruct-GSQ-RCO-GGUF/tree/main
  （GSQ-RCO abliterated IQ3_S：分片1 `55063664928` + 分片2 `28800138432` + `mmproj-Qwen3.8-Flash-Next-BF16.gguf` `907543008`；本机 `~/models/DavidAU/…/`，`--native` 指分片1）
* **ngram（PLE，外挂 Q8_0，替换原生 IQ4_NL shard 2）**：还是那一个 https://huggingface.co/sleepyeldrazi/Qwen3.8-Flash-Next-GSQ-RCO-Q8_0-PLE-GGUF/tree/main
  （51 G；本机 `~/models/sleepyeldrazi/…-00002-of-00002.gguf`，`--ple-gguf`。换 ctx 不用重下，也不换表；「shard 2 即使被 PLE 顶替也必须落盘」的原因见 0.1.45 那节，0.2.2 行为不变）
* **script**：[launch_davidau_iq3s_q8ple_256k.sh](launch_davidau_iq3s_q8ple_256k.sh)（启动器）· [example-strata-davidau-iq3_s-q8ple-256k.json](example-strata-davidau-iq3_s-q8ple-256k.json)（live json 快照）
  · [scripts/decode_ctx_tps.py](scripts/decode_ctx_tps.py)（本轮跑的 bench）· [results/davidau_iq3_256k.json](results/davidau_iq3_256k.json)（原始数据）

```
  tier    code   human    math   multi      qa     rag  reason    role    stem   write     AVG  |     pre_s  pre_min   pre_tps
------------------------------------------------------------------------------------------------------------------------------
    8k    97.8    95.6    88.3    88.3   102.4    96.5   122.5    98.0    98.9    98.1    98.6  |       4.9     0.08      1625
   32k    94.1    92.5    81.7    93.2    96.6    93.8   116.6    99.3   104.4    91.1    96.3  |      14.9     0.25      2110
   64k    93.2    88.2    92.6    92.5   104.6    89.6    96.5    91.6    97.1    94.4    94.0  |      21.9     0.36      2863
  128k    85.3    91.6    70.0    90.6   102.9    92.3   112.2   101.9    86.3    80.3    91.3  |      39.8     0.66      3144
  200k    87.0    93.3    75.1    84.5    97.6    88.1    93.6    99.6    99.2    85.9    90.4  |      60.8     1.01      3212
  256k    98.0    96.5    88.2    91.0    97.0    92.0   119.3    98.7    94.3    92.4    96.7  |      78.1     1.30      3195
------------------------------------------------------------------------------------------------------------------------------
   AVG    92.6    93.0    82.6    90.0   100.2    92.0   110.1    98.2    96.7    90.4    94.6
saved /home/xi/ubuntu_handoff/results/davidau_iq3_256k.json
```

- **decode AVG 94.6 = 本机历史最高**。同 ctx 对 0.1.43.1 的 256K 基线（[results/43_1_patch_1.json](results/43_1_patch_1.json)，AVG 85.0）**+11.3%**，
  256k 单档 81.8 → **96.7（+18.2%）**、200k 81.0 → 90.4、128k 85.4 → 91.3，**六档全涨**。
  跨 ctx 对 0.1.45 + YaRN×2 @512K 那版（AVG 79.7）是 **+18.7%**，但那一段里同时换了引擎、拆了 YaRN、上下文减半，不能全记在版本头上。
  口径要说清：43.1 基线是 **Swift IQ3_S + `PREFILL=8192`**，本轮是 **DavidAU IQ3_S + `PREFILL=16384`**，三件事一起动；要单拆版本收益得同模型重跑。
- **prefill**：256K 冷读 **78.1 s**（43.1 同 ctx 85.6 s，**-8.8%**；0.1.45 @512K 是 172.0 s），峰值 **3212 t/s @200k**、128k 3144、256k 3195。
  **8k/32k 两档 pre_tps 要留问号**：1625 / 2110 vs 基线 1608 / 2548，32k 反而慢 17%，这两档是引擎起来后第一批跑的（专家热层 / expert-profile 没热），0.1.45 那轮同现象；要准就重跑这两档。
- **停放 5 槽配不满**：`--conversation-cache-mib 20480 --conversation-cache-slots 5`，但 256K 每条约 **3.9 G**，64 G 物理 RAM 上常驻专家已吃 **32.48 GiB**，实测只停得住 4 条。
  引擎逐字日志：`conversation cache: skip parking (physical RAM admission; need 3754 MiB plus 2560 MiB floor, 4278 MiB available, source meminfo; evicted 4, 0 still parked)`，
  另有 `parked=4 bytes=15743201872 evictions=56`。→ 第 5 槽会 LRU 驱逐，稳态按 **4 条**算；要 5 条稳态就 `CONV_CACHE_SLOTS=4` 或加 RAM。
- **温度**：json 里 `sampling` = `1.0 / 0.95 / 20` + `force: true`（= 模型页 / Strata 内置卡片的 thinking 档；instruct 档是 `0.7 / 0.8 / 20 + presence_penalty 1.5`）。
  4090 那台钉死 0.9，本机这版**没钉**（客户端传什么就用什么）。`effort_position: "end"` 由 serve 翻成 `--tail-role-token 8678`，json 的 `args` 里看不到它，属正常。
- 口径：`tg=256`、`runs=1`、`exact_tg`、`flush_per_tier`、`enable_thinking=false`，`n_ctx=262144`、`model_name=qwen-flash-next`；各档 prompt token 8k 7912 / 32k 31352 / 64k 62606 / 128k 125125 / 200k 195454 / 256k 249668，
  与 43.1 基线那轮逐档差 <0.5%，可直接比。

### Strata 0.1.45（2026-10-18 官方源码编译 sm_86，只带 #1233 PLE 补丁；2026-10-21 已被 0.2.2 取代）

* **换法**：gh-proxy 拉 `v0.1.45.zip` 解到 `~/src/Strata-0.1.45` → **只打** `patches/ple-only-native-dense-v0.1.43.1.patch`（0.1.45 仍未修 #1233，不打启动即崩）
  → `cmake -G Ninja -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER=~/venvs/sglang-0519/…/cu13/bin/nvcc`
  `-DCMAKE_CUDA_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK -DSTRATA_GGML_DIR=~/src/Strata/third_party/llama.cpp` → `ninja -C build strata`（145 目标，约 2 分钟）
  + `build-vision`（275 目标，约 4 分钟）。官方 release 只有 Windows 包，Linux 只能源码编。
  换二进制要先 unlink 再拷（运行中的 inode 会「文本文件忙」），本机模型加载 **235 s**。
* **双卡 `--prompt-cache-tail` 补丁按决定不再打**：0.1.45 树 = 干净官方树 + PLE 补丁，launcher `PROMPT_CACHE_TAIL` 默认 **0**（引擎 cmdline 里已无该 flag，实测确认）。
* **script**：仍是 `launch_swift_abliterated_iq3s_q8ple_512k.sh`（ctx 524288 + YaRN×2 + 槽 3 / 池 14 + `PREFILL=8192`，单流 `total_slots=1`）。
* **回滚**：`systemctl --user stop strata-swift && mv -f ~/src/Strata/engine/strata.bak-0.1.44.1p ~/src/Strata/engine/strata && mv -f ~/src/Strata/engine/strata-vision.bak-0.1.44.1p ~/src/Strata/engine/strata-vision && systemctl --user start strata-swift`。

#### 0.1.45 + DavidAU IQ3_S + YaRN×2 @ 512K（换模型不换引擎，bench 2026-10-20 14:03，`results/davidau_iq3.json`；已被 0.2.2 @256K 取代）

* **model**：https://huggingface.co/DavidAU/Qwen3.8-Flash-Next-Abliterated-Uncensored-Turbo-Brilliance-50x-Reasoning-Instruct-GSQ-RCO-GGUF/tree/main
  （GSQ-RCO abliterated IQ3_S：分片1 `55063664928` + 分片2 `28800138432` + `mmproj-Qwen3.8-Flash-Next-BF16.gguf` `907543008`；本机 `~/models/DavidAU/…/`，`--native` 指分片1）
* **ngram（PLE，外挂 Q8_0，替换原生 IQ4_NL shard 2）**：还是那一个 https://huggingface.co/sleepyeldrazi/Qwen3.8-Flash-Next-GSQ-RCO-Q8_0-PLE-GGUF/tree/main
  （51 G；本机 `~/models/sleepyeldrazi/…-00002-of-00002.gguf`，`--ple-gguf`。同底座同 tokenizer，Swift 那版和 DavidAU 这版共用同一张 Q8_0 表，不用重下）
* **shard 2 必须落盘（本轮新踩，值得记）**：文件名是 `-00001-of-00002`，Strata `include/strata/artifact/gguf_split.hpp:45` 把 1..N 逐个 `ifstream` 打开，
  **打不开就抛 `missing model shard … (shard 2 of 2; is the download complete?)`**，`GgufModel` 构造还要求每片都是合法 GGUF（`split.count/no/tensors.count` 与分片1 对得上）。
  它不看内容：shard 2 里只有一个张量 `per_layer_token_embd.weight`（IQ4_NL n-gram 表，160×320001536×0.5625 B = 28800138432，逐字节对得上），
  我们已用 `--ple-gguf` 换成 Q8_0，**但那 28.8 G 还是得在**，否则引擎 `exit code 2`，视觉编码器也报 `cannot open the text model`。
  `tools/iq_pack.py` 同样硬要 shard 2 才肯跑。省法：先只下 shard1 + mmproj（53 G），pack 在另一台分片下全的机器上打好再 rsync 过来（1.5 G / 13 s @111 MB/s），
  shard 2 走局域网拷（28.8 G / 4 min 54 s @93 MB/s，2.5 GbE），比从镜像重下（9.4 MB/s ≈ 51 min）快一个数量级。
* **pack**：`tools/iq_pack.py --gguf <shard1> --out ~/src/Strata-data/packs/davidau-gsq-rco-iq3_s` → `dense.bin` 1538035200 + `tokenizer/chat_template.jinja`（从 GGUF 导出，与模型页 / Strata 内置卡片一致）。
* **MTP**：沿用 `~/src/Strata-data/mtp/rt`（Swift 官方 `mtp.*` → q2_0），换模型不用重打。
* **script**：`launch_davidau_iq3s_q8ple_512k.sh` = 512K 启动器的换模型版（`gpus 1,0` + `layer_split 26`、ctx 524288 + yarn×2、k8v4、`PREFILL=8192`、
  旗杆 `6×16384`、停放 `16384 MiB / 3 槽 / min-tokens 18432 / min-free 2560`、`--resident-experts`、`--pipeline-windows 2`、vision gpu、单流）。
  json 快照 `example-strata-davidau-iq3_s-q8ple.json`（含 `effort_position: "end"`）。**开机单元 drop-in 已指这份脚本**（2026-10-20，`~/.config/systemd/user/strata-swift.service.d/override.conf`）。
* **温度**：不钉。模型页 = Strata 卡片：thinking `1.0 / 0.95 / 20`，instruct `0.7 / 0.8 / 20 + presence_penalty 1.5`。
  本机 `serve/server.py` 不认 `sampling.force`（boot 日志逐字：`config sampling.force=True: unknown key, ignored`），config 里的 1.0/0.95/20 只是默认值，客户端传什么就用什么。

```
  tier      code   human    math   multi      qa     rag  reason    role    stem   write     AVG  |    pre_s  pre_min  pre_tps
------------------------------------------------------------------------------------------------------------------------------
    8k      88.6    84.9    75.5    78.2    93.7    78.0   105.1    88.9    90.4    92.0    87.5  |      6.7     0.11     1188
   32k      82.4    84.1    66.1    78.6    86.9    74.7   105.8    81.0    85.6    80.4    82.6  |     15.6     0.26     2012
   64k      84.3    83.1    76.0    81.9    87.2    77.7    91.1    84.4    83.7    81.9    83.1  |     22.7     0.38     2763
  128k      83.1    86.8    68.3    75.9    78.0    67.2    94.8    81.8    79.4    70.8    78.6  |     41.4     0.69     3020
  200k      70.3    67.6    68.2    72.8    87.4    77.1    99.0    82.0    75.9    73.5    77.4  |     65.7     1.09     2977
  256k      77.1    82.9    69.6    76.9    86.8    76.8   100.8    82.4    84.4    74.0    81.2  |     83.2     1.39     3007
  384k      72.3    69.7    65.6    70.4    66.5    73.7   103.6    78.2    74.2    76.0    75.0  |    130.7     2.18     2871
  512k      68.4    74.4    66.9    70.8    75.8    66.2    88.1    76.9    69.3    65.3    72.2  |    172.0     2.87     2906
------------------------------------------------------------------------------------------------------------------------------
   AVG      78.3    79.2    69.5    75.7    82.8    73.9    98.5    82.0    80.4    76.7    79.7
```

- **decode 全面高于同机 Swift 版**：AVG **75.3 → 79.7（+5.9%）**；64k +12.4%、256k +7.8%、384k +6.5%、200k +5.6%、512k 持平（72.2）。这是当时本机最高的一版（0.2.2 @256K 已到 94.6，见上）。
- **prefill**：512K 冷读 **190.7 s → 172.0 s（-9.8%，2906 t/s）**；64k 以上 +1.4~+10.8%。
  **但 8k/32k 反而慢**（1543→1188、2432→2012）：这两档是引擎重启后第一批跑的，专家热层 / expert-profile 还没热，别当结论；要准就重跑一轮。
- 口径：同机同引擎同 ctx 同 `PREFILL=8192` 同单流，只差 model + pack + n-gram 来源；两轮 prompt token 数几乎一致（512k 档 499721 vs 500187），`tg=256`、`runs=1`、`flush_per_tier`、thinking off。
- **2026-10-20 本机状态**：开机默认 = 这份 DavidAU 启动器；Swift 的 GGUF（79 G）和它的 pack（1.5 G）已从本机删除，`results/45_512k_1.json` 那套 Swift 数据只作历史对照。

#### 0.1.45 + YaRN×2 @ 512K（ctx 524288，PLE 补丁，槽 3 / 池 14，PREFILL=8192，单流；bench 2026-10-18 14:2x，`results/45_512k_1.json`）

```
  tier    code   human    math   multi      qa     rag  reason    role    stem   write     AVG  |     pre_s  pre_min   pre_tps
------------------------------------------------------------------------------------------------------------------------------
    8k    85.5    83.8    66.4    79.6    87.3    75.2   107.2    85.3    80.5    88.6    83.9  |       5.1     0.09      1543
   32k    75.2    83.3    67.8    76.7    80.3    73.9    83.7    81.9    80.3    75.9    77.9  |      12.9     0.22      2432
   64k    65.9    72.4    70.0    73.6    74.3    69.1    90.2    70.7    79.9    73.8    74.0  |      23.0     0.38      2724
  128k    74.5    71.7    64.6    72.3    78.3    71.2    83.0    77.0    77.8    80.2    75.1  |      43.6     0.73      2869
  200k    66.1    77.3    65.5    72.2    75.9    71.4    77.4    75.2    72.9    78.7    73.3  |      68.2     1.14      2870
  256k    72.8    75.1    64.2    70.3    80.6    72.4    88.8    80.5    74.9    73.5    75.3  |      88.6     1.48      2825
  384k    76.1    70.5    65.1    66.7    66.5    65.0    91.2    65.8    73.6    63.8    70.4  |     136.8     2.28      2745
  512k    76.9    71.2    60.7    65.6    72.3    67.2    84.3    79.5    74.1    69.7    72.2  |     190.7     3.18      2622
------------------------------------------------------------------------------------------------------------------------------
   AVG    74.1    75.7    65.5    72.1    76.9    70.7    88.2    77.0    76.8    75.5    75.3
```

- **本机目前最快的一版**：decode AVG **75.3**（0.1.43.1 同配置 72.9，**+3.3%**；0.1.44.1 当年全档 -5% 已回滚）。
  分档：8k 72.3→**83.9（+16%）**、256k 72.9→75.3、512k 69.8→**72.2（+3.4%）**；64k 77.3→74.0（-4.3%）是唯一明显回落档，384k 持平。
- **prefill 全面 +20~28%**：32k 2024→2432、128k 2234→2869、512k 2114→**2622**；512K 冷读 **236.6 s → 190.7 s**。
  口径要说清：基线那轮 `PREFILL=4096`，本轮 8192，所以这块收益里有一部分来自块大小而非版本（0.1.45 release notes 的方向正是「更大的默认 prompt 块 + 新 decode 内核」）。
- 模型 id 现为 **`qwen-flash-next`**（json `model_name`）；旧名 `swift-1.5-abliterated-iq3_s-q8ple` 打过来仍会跑，只是不再出现在 `/v1/models` 里。


### Strata 0.1.44.1（2026-10-15 实测后回滚；model / ngram / MTP 同 0.1.43.1，#1233 补丁同样必打）

* **script**：同一套 launcher；换引擎 = stop 单元 → cp 二进制 → start（备份 `engine/strata.bak-0.1.43.1p` 带补丁）。
  回滚一条命令：`systemctl --user stop strata-swift && cp ~/src/Strata/engine/strata.bak-0.1.43.1p ~/src/Strata/engine/strata && systemctl --user start strata-swift`。

#### 0.1.44.1 @ 256K（本版本只测到 256K，PREFILL=8192，单流；bench 2026-10-15 12:5x，`results/44_1_1.json`）

```
  tier    code   human    math   multi      qa     rag  reason    role    stem   write     AVG  |     pre_s  pre_min   pre_tps
------------------------------------------------------------------------------------------------------------------------------
    8k    82.0    79.7    73.0    81.4    96.8    87.8    98.2    91.8    84.3    93.4    86.8  |       5.0     0.08      1574
   32k    78.7    88.1    81.6    77.2    85.3    79.0    88.7    87.2    84.7    86.8    83.7  |      12.6     0.21      2492
   64k    76.8    78.4    75.8    79.7    78.6    78.2    97.0    76.2    86.7    83.9    81.1  |      22.4     0.37      2796
  128k    73.2    75.1    78.9    77.3    87.9    78.7    95.8    79.4    74.0    78.3    79.9  |      42.7     0.71      2932
  200k    74.8    70.9    67.6    71.3    80.9    76.1    83.2    74.8    76.8    78.8    75.5  |      66.4     1.11      2945
  256k    73.8    79.4    71.5    71.0    78.0    75.8    87.8    71.7    78.3    80.1    76.7  |      91.0     1.52      2747
------------------------------------------------------------------------------------------------------------------------------
   AVG    76.5    78.6    74.7    76.3    84.6    79.3    91.8    80.2    80.8    83.5    80.6
```

- **对比 v0.1.43.1 同套件（`results/43_1_patch_1.json`，AVG 85.0）：44.1 全档位一致回落 ~5%**
  （92.0→86.8 / 85.1→83.7 / 85.0→81.1 / 85.4→79.9 / 81.0→75.5 / 81.8→76.7），
  prefill 持平（2932-2945 中档，256k 2747 vs 2928 略降）。0.1.44.1 的修复项（老 CPU/CUDA 12.0/12.1/M2/setup 默认）与我们无关。
- 观察：44 系 resident 从 24.63 GiB 涨到 27.60 GiB（pinned RAM 多 ~3 G，挤停放预算），mtp 接受率 60-74%（43.1 同脚本 74-94%）。
- **本机建议：留 0.1.43.1**（回滚一条命令：`systemctl --user stop strata-swift && cp ~/src/Strata/engine/strata.bak-0.1.43.1p ~/src/Strata/engine/strata && systemctl --user start strata-swift`）。
- #1233 PLE 补丁在 44.1 仍必打（`patches/ple-only-native-dense-v0.1.43.1.patch` 原样可打到 44.1 树，行号偏移不影响 patch -p1 上下文匹配）。

### Strata 0.1.43.1（历史基准；本机长期主力引擎，sm_86 源码编译）

* **model**：https://huggingface.co/slider-meister-pub/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-GGUF
  （IQ3_S abliterated 2 分片 + mmproj BF16 + SHA256SUMS，79 G；本机 `~/models/slider-meister-pub/…/`，`--native` 指分片1）
* **ngram（PLE，外挂 Q8_0，替换原生 IQ4_NL shard 2）**：https://huggingface.co/sleepyeldrazi/Qwen3.8-Flash-Next-GSQ-RCO-Q8_0-PLE-GGUF/tree/main
  （51 G；本机 `~/models/sleepyeldrazi/…-00002-of-00002.gguf`，`--ple-gguf`）
* **MTP**：源 https://huggingface.co/ukisai/Swift-Qwen3.8-Flash-Next （BF16 `mtp.*` → `pack_official_mtp.sh` 打 q2_0 → `~/src/Strata-data/mtp/rt/`，0.79 G）
* **补丁（必打）**：`patches/ple-only-native-dense-v0.1.43.1.patch` — 上游 #1233 未修：外挂 PLE-only 单张量表撞 Swift shard 2 的 split.no，
  不打则启动即 `native dense: duplicate split shard number` 崩溃循环。44.1 树原样可打。
  Codex 插话 / 尾部分叉曾另打 `patches/prompt-cache-tail-layer-split.patch`（上游 `--prompt-cache-tail` 写死 `!multi_gpu`，0.1.45 仍是；双卡各 stage 在尾巴 chunk 存 part）。
  **2026-10-18 起撤掉**（收效甚微，判据见「双卡 --prompt-cache-tail」一节），launcher `PROMPT_CACHE_TAIL` 默认改 0；补丁存档在 git 历史 commit 16c3762。
* **script**：
  - `launch_swift_abliterated_iq3s_q8ple_512k.sh` = **当前在用 512K**：ctx 524288 + `--rope-scaling yarn --rope-scale 2`（`--yarn-orig-ctx` 默认 262144 免传）+ 停放槽 3 + 检查点池 14（停放实测：快照 **18.5-35 KB/token**，满长 512K ≈ **10-15 GiB/条**，19456 MiB 预算只够 ~1-2 条满长；纯 KV k8v4 9.2 KB/tok 的估算是低估，快照不止存 K/V）。
  - `launch_swift_abliterated_iq3s_q8ple.sh` = 256K 版（ctx 268288，无 rope flag，槽 6 池 28）。三 launcher 共用 `~/src/Strata/strata-swift-…-q8ple.json`，**谁最后启动谁赢**。
  - `launch_swift_abliterated_iq3s_q8ple_multi.sh` = 双路（PARALLEL=2，BATCH_MODE=draft|pipe，定案见下文 A/B）。
  - `systemd/strata-swift.service`：开机默认单元（ExecStart 现仍指 256K 脚本；**unit env 才是 PARALLEL/CONV_* 杠杆**，json 不是）。
    2026-10-16 状态：**已加 drop-in**（`~/.config/systemd/user/strata-swift.service.d/override.conf`，ExecStart→512K 启动器），下次开机自启即 512K；回 256K = 删 drop-in。
  - `JSON_ONLY=1 bash <launcher>`：只写 json 不启动、不 pkill（干跑验证用）。
  - YaRN 坑：缩放整进程固定（缓存 K 是 post-RoPE，"one run one scaling"），切 rope 模式必须重启；>262144 官方口径必须 yarn
    （上游 DETAILS.md 同款 IQ3_S：293K NLL yarn2 2.04 vs 裸 2.22 = 3-4× 噪声，≤32K 无退化，421K yarn2 QA 8/8；1M 才需要 yarn4）。
  - json 快照：`example-strata-swift-abliterated-iq3_s-q8ple.json`（512K 版，2026-10-16）。
  - 下载法：有梯子 `https_proxy=http://127.0.0.1:7897` + `aria2c -c` 走官方 resolve；无梯子换 hf-mirror（详见 03_downloads_mirrors.md）。

#### 双卡 `--prompt-cache-tail`（Codex 插话尾部分叉；2026-10-18，`results/prompt_cache_tail_split_20261018.txt`）

上游 `--prompt-cache-tail` 只给单卡（`generate.cpp` `!multi_gpu`，0.1.45 仍是，**v0.2.2 也仍是**，2026-10-22 复核：`src/program/generate.cpp:9771` 那个条件里还写着 `!multi_gpu`）。Codex 插话 = 掐断当前请求、丢掉助手半截、把新句子接在上一句完整话后面重发 → 尾部分叉，不是追加。`--prompt-cache-every 131072` 下分叉最多回退 128K。本机补丁：去掉该限制，layer-split 各 stage 在尾巴 chunk 也存 part。启动器默认 `PROMPT_CACHE_TAIL=1`。打法（该文件已从工作树删除，只剩 git 历史）：`git show 16c3762:patches/prompt-cache-tail-layer-split.patch > /tmp/tail.patch && patch -p1 -i /tmp/tail.patch`（Strata 根）。本次测在本机源码重编后的引擎上（boot 日志 `session is up (engine 0.1.44.1)`），512K launcher，`PREFILL=8192`。**插话这条线现在的有效解法是 same-chat rewind（见上面那节），tail 已不用。**

```
  run                      prompt  reused     read  wall_s  note
--------------------------------------------------------------------------------
  A  cold fill              22141       0    22141    11.5  log: tail ckpt @ 16384 (2*8192)
  B  tail fork              22141   16384     5757     5.0  last line changed; 74% prefix hit
  B  without patch (infer)  22141       0    22141   ~11.5  prompt < every 131072, no mid ckpt
```

- 同前缀、只改最后一句（模拟 Codex 插话）。A 打出 `prompt-cache-tail checkpoint at 16384 tokens (layer split)`；B 复用 16384，墙钟 11.5 s → 5.0 s。
- 无补丁时 22141 < 131072，没有周期旗杆，B 会再走整段（A 那行就是对照）。
- 旗杆约占 118 MB RAM/根，不占显存。`PROMPT_CACHE_TAIL=0` 关掉写入。
- **2026-10-18 撤掉这份补丁**：省时间的只有「prompt 短于 `--prompt-cache-every` 且正好拐在尾巴 chunk 边」这一格；真实 Codex 插话的 prompt 长度不固定，
  长 prompt 照样退到 128K 周期旗杆 → 收效甚微。0.1.45 树不再打，launcher `PROMPT_CACHE_TAIL` 默认 0，等官方支持多卡尾巴检测点再开回来。
  （4090 单卡官方就支持 tail，那边不受 `!multi_gpu` 限制，已在用。）


#### 埋点 NIAH @512K（`scripts/needle_512k.py`，5 针 × 5 深度，严格 KEY=<code> 判定）

```
prompt=493,734 tok (=1.88× 训练长 262,144)  单请求一次问 5 针   ->  5/5 PASS
  depth  10%  ALPHA    CFBU-1981   ✓        depth  70%  DELTA    KZQ8-2208   ✓
  depth  30%  BRAVO    2S6T-8747   ✓        depth  90%  ECHO     88UE-0921   ✓   (最深针 ~444K token 位)
  depth  50%  CHARLIE  HKCE-6908   ✓
复测 472,481 tok: 5/5 (`results/needle_512k_full.json` / `results/needle_512k.json`)；8k sanity: 5/5 (`results/needle_sanity.json`)
```

- **256K→512K 边际成本 ≈ 4-5%**：decode AVG 72.9→69.8，prefill 2226→2114 t/s；8k→512k 全程 decode 才 -3.4%
  （GDN 线性注意力 O(1)，decode 对长度近乎平坦——与上游 "decode is flat in context" 说法一致）。
- 512K 冷 prefill：500,183 tok **236.6 s（3.94 分钟）** @2114 t/s。该深度 decode 63-86 t/s（针测实测 63-76）。
- **512K 配置**（`launch_swift_abliterated_iq3s_q8ple_512k.sh`，快照 `example-strata-swift-abliterated-iq3_s-q8ple.json`）：
  `--max-context 524288 --rope-scaling yarn --rope-scale 2`（`--yarn-orig-ctx` 默认 262144，免传）；
  停放槽 6→**3**、检查点池 28→**14**（停放实测：快照 **18.5-35 KB/token**，满长 512K ≈ **10-15 GiB/条**，19456 MiB 预算只够 ~1-2 条满长；纯 KV k8v4 9.2 KB/tok 的估算是低估，快照不止存 K/V）。
- **为什么带 YaRN**：上游 `docs/DETAILS.md` 同款 IQ3_S 实测 293K 处 yarn2 NLL 2.04 vs 裸 2.22（3-4× 噪声），≤32K 无退化，421K yarn2 长文 QA 8/8；
  上游 [PR #1692](https://www.stratat.com/en/pulls/1692)：普通 RoPE 512K 埋点 3/3、普通 RoPE 1M 0/3、YaRN×4 1M 3/3。
  引擎口径：rope 缩放整进程固定（缓存 K 是 post-RoPE），无按请求切换；>262K 官方就是要 yarn。
- 本表低档位比同日早间 8192 基线低 10-15%：系 **PREFILL=4096 + 会话负载**（引擎内同时驻留 ~140K 工作会话 + 停放长文），
  **非 yarn 税**（yarn on/off 同负载对照未做）。
- 1M：本机未测。上游 YaRN×4 1M 3/3，但 llama.cpp 同族在 YaRN×4 有 ~2×orig（520K）prefill 崩溃先例
  （[llama.cpp#27090](https://github.com/ggml-org/llama.cpp/issues/27090)）；62 G 机器 1M 停放无望。

#### 0.1.43.1 + YaRN×2 @ 512K（ctx 524288，PLE 补丁，槽 3 / 池 14，PREFILL=4096，单流；bench 2026-10-16，原始数据 `results/512_yarn_1.json`）

```
  tier    code   human    math   multi      qa     rag  reason    role    stem   write     AVG  |     pre_s  pre_min   pre_tps
------------------------------------------------------------------------------------------------------------------------------
    8k    62.6    62.3    68.4    73.6    84.0    70.6    82.8    73.5    66.5    78.3    72.3  |       5.2     0.09      1513
   32k    66.6    68.3    62.5    70.4    73.3    72.6    87.8    71.2    72.7    78.1    72.3  |      15.5     0.26      2024
   64k    78.9    81.1    72.5    71.0    78.3    73.4    87.6    74.4    77.4    78.0    77.3  |      28.2     0.47      2222
  128k    71.7    74.5    62.7    69.2    76.2    73.8    92.2    80.5    76.2    73.2    75.0  |      56.0     0.93      2234
  200k    71.9    79.8    59.3    69.9    74.8    68.6    89.3    73.9    73.4    74.4    73.5  |      86.4     1.44      2263
  256k    66.8    75.5    63.5    65.0    74.1    69.3    92.5    72.8    70.3    79.3    72.9  |     112.5     1.87      2226
  384k    69.3    69.7    70.9    67.0    73.7    66.2    83.7    69.9    68.6    65.4    70.4  |     174.3     2.91      2154
  512k    66.2    72.5    68.1    62.7    69.9    65.8    86.1    67.8    66.9    71.8    69.8  |     236.6     3.94      2114
------------------------------------------------------------------------------------------------------------------------------
   AVG    69.2    73.0    66.0    68.6    75.5    70.0    87.8    73.0    71.5    74.8    72.9
```

#### 0.1.43.1 @ 256K（ctx 268288，PLE 补丁，PREFILL=8192，单流；bench 2026-10-13 20:23，原始数据 `results/43_1_patch_1.json`）

```
  tier    code   human    math   multi      qa     rag  reason    role    stem   write     AVG  |     pre_s  pre_min   pre_tps
------------------------------------------------------------------------------------------------------------------------------
    8k    90.3    97.1    74.9    83.2   103.6    93.3    96.1    90.6    94.5    96.2    92.0  |       4.9     0.08      1608
   32k    76.4    89.8    72.3    80.4    89.8    82.9    95.0   100.5    87.8    75.7    85.1  |      12.3     0.21      2548
   64k    81.3    80.4    80.0    75.7    88.3    91.8    97.8    97.3    77.1    80.5    85.0  |      22.2     0.37      2831
  128k    81.5    84.6    77.8    83.4    91.2    85.0    97.2    89.2    79.2    84.6    85.4  |      42.1     0.70      2977
  200k    77.1    82.9    66.0    78.5    87.6    80.7    87.9    86.5    86.8    75.7    81.0  |      66.4     1.11      2950
  256k    73.8    77.3    79.7    78.7    83.5    84.5    87.5    87.4    81.4    83.9    81.8  |      85.6     1.43      2928
------------------------------------------------------------------------------------------------------------------------------
   AVG    80.1    85.4    75.1    80.0    90.7    86.4    93.6    91.9    84.5    82.8    85.0
```

- decode 全档位真实文本 **AVG 85.0 t/s**（10 类测试各有，才测得出专家交换的负载；单流）。
- prefill：`PREFILL=8192` 后 64K+ **2831-2977 t/s ≈ RTX 4090 单卡（~3000）**；8K 档 1608（短 prompt 吃不满 8192 窗）。
- 版本对照（同一微基准，短 ctx）：decode 0.1.42 **93** → 0.1.43 **95.2** → **0.1.43.1 101.8/152.8 t/s**（draft 接受率 ~93% 不变，纯窗口速度）；
  prefill 25K 冷 ~1750（0.1.42，PREFILL=4096）→ 2062（0.1.43.1，PREFILL=4096）→ **2928-2977（0.1.43.1 + PREFILL=8192）**。
- 43.1 的 decode 增益来自**默认开启**的 `R4 hit path ON`（resident experts computed on GPU）+ `token graph hit path`（设备端决策）；
  release notes 的 "RTX 30 decode +5-11%" 是 Q2_0-pack 的 Q3_K kernel，**IQ3_S 不吃**；`STRATA_ADAPT_OVERLAP` 对本机无效
  （boot 日志：`is off: not on a layer split`）。
- 对照组：`results/iq3_4096_1024reserve.json`（2026-10-12 15:12，v0.1.43 + PREFILL=4096 / VRAM_RESERVE=1024）。


### 老版本速记（无全量十类表）

- decode 微基准（同一脚本短 ctx）：0.1.42 **93** → 0.1.43 **95.2** → 0.1.43.1 **101.8/152.8 t/s**（draft 接受率 ~93% 不变，纯窗口速度）。
- prefill 微基准：25K 冷 ~1750（0.1.42，PREFILL=4096）→ 2062（0.1.43.1，4096）→ **2928-2977（0.1.43.1，PREFILL=8192）**。
- 对照数据：`results/iq3_4096_1024reserve.json`（0.1.43 + PREFILL=4096 / VRAM_RESERVE=1024）；0.1.43.0 时代 parallel-2 双路各 9-11 t/s（slot 无 draft）。
- 43.1 的 decode 增益来自**默认开启**的 `R4 hit path ON`（resident experts 上 GPU 计算）+ `token graph hit path`；release notes 的 "RTX 30 decode +5-11%" 是 Q2_0-pack Q3_K kernel，**IQ3_S 不吃**；`STRATA_ADAPT_OVERLAP` 对本机无效（boot 日志 `is off: not on a layer split`）。


## Goal

Bring up **Strata v0.1.38** serving **Swift 1.5 Qwen3.8-Flash-Next**:

- **IQ3_XXS** (GSQ-RCO): official Swift MTP packed q2_0, adaptive MTP, GPU vision, dual RTX 3080 layer-split, OpenAI API on **0.0.0.0:8005**.
- **IQ4_XS** (bartowski): same engine after `patches/resident-on-split-v0.1.38.patch` — `--resident-experts` on the split (GPU caches, then RAM complement).

Do **not** use ISTA IQ3_S, original-Qwen MTP (`mtp-qwen`), Unsloth BF16/Q8 MTP, or the GitHub **sm_120** prebuilt engine.

## Upstream (this is not a Strata fork)

This repo is **launch/compile recipe only**. Engine source is pulled at setup time. Do not vendor Niko's tree.

| used | url | pin / note |
| --- | --- | --- |
| Strata engine (ONLY tree used) | https://github.com/Niko1221/Strata | tag **`v0.1.38`** = https://github.com/Niko1221/Strata/releases/tag/v0.1.38 |
| clone in this bundle | `clone_strata.sh` | `gh-proxy.com/https://github.com/Niko1221/Strata/archive/refs/tags/v0.1.38.zip` |
| llama.cpp (via Strata setup.py) | https://github.com/ggml-org/llama.cpp | commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` |
| GGUF | https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF | IQ3_XXS two shards + mmproj |
| MTP source | https://huggingface.co/ukisai/Swift-Qwen3.8-Flash-Next | BF16 `mtp.*`, then pack q2_0 |
| this recipe repo | https://github.com/Tlyer233/strata-dual-3080-qwen-flash | dual RTX 3080 20GB sm86 |

**There is no separate "Strata-3090" git project in this stack.** Public 3090 numbers and the "gaming PC" install path are the **same** [Niko1221/Strata](https://github.com/Niko1221/Strata) repo (Ampere 24GB is their headline card). RTX 3080 20GB and RTX 3090 24GB are both **sm_86**. Official Linux prebuilt `strata-linux-x64.zip` is **sm_120** (50-series); 30-series must compile locally (`compile_engine.sh`). Do not confuse with `ninfer-3090` / `iamwavecut/ninfer` — different engine, not used here.

Local patches on top of v0.1.38 (do not upstream unless asked): `patches/apply_local.py` (`STRATA_SKIP_BUILD`, `sampling.force`, **dual-GPU `--resident-experts`**). See [RESIDENT_ON_SPLIT.md](RESIDENT_ON_SPLIT.md).

## Do not put in git

- GGUF shards (~72 GiB)
- `mtp-q2_0.gguf` / `mtp/rt/*.bin` (~0.8 GiB after pack; source BF16 mtp.* is larger)
- compiled `engine/strata` / `strata-vision` (~108 MiB ELF). **Do not commit.** They are sm_86 + dynamically linked to pip `libcudart.so.13` / `libcublas.so.13` under `~/venvs/sglang-0519/.../nvidia/cu13/lib`. Another clone still needs that CUDA 13 and `compile_engine.sh`. Optional later: GitHub **Release** asset, never the git tree.
- `.venv`

This folder is scripts + tokenizer + expert-profile + agent recipe only. Copy, never move, from the original machine.

## Reference machine

```
host:        this ubuntu_handoff box
cpu:         AMD Ryzen 7 5700X 8-Core, x86_64
ram:         64 GiB
gpu0:        NVIDIA GeForce RTX 3080 20480 MiB  compute 8.6  (desktop display)
gpu1:        NVIDIA GeForce RTX 3080 20480 MiB  compute 8.6  (cleaner, first stage)
driver:      610.57.04
cuda nvcc:   ~/venvs/sglang-0519/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc
             (pip CUDA 13.4, NOT /usr/local/cuda)
os:          Ubuntu, kernel 7.0.0-34-generic, python 3.12.3
clash:       mixed-port 127.0.0.1:7897 when VPN is up
```

Engine must be compiled **sm_86**. CMakeLists default arch is 120. Prebuilt `strata-linux-x64.zip` is sm_120 and will not run these cards.

Physical GPU mapping used here: `--gpus 1,0` + `layer_split=26` means CUDA device 1 (physical GPU1) holds layers 0-25; CUDA device 0 (physical GPU0 / display) holds later layers + head + MTP. Last stage also needs extra VRAM (`kWindowMib=96` + `kDrafterMib=1000` inside engine). `--vram-reserve-mib 800` is the working value: 256 OOMs the MTP draft head; 1600 wastes expert cache.

## Canonical paths on this PC

| what | path | bytes |
| --- | --- | --- |
| Strata tree | `~/src/Strata` | tag `v0.1.38` |
| engine | `~/src/Strata/engine/strata` + `strata-vision` | local compile 0.1.38 archs `[86]` |
| BUILD.json | `~/src/Strata/engine/BUILD.json` | see `example-BUILD.json` |
| data dir | `~/src/Strata-data` (sibling of Strata, not inside it) | |
| GGUF dir | `~/models/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/` | |
| shard1 | `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` | **39785790560** |
| shard2 | `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` | **36180282560** |
| mmproj | `mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf` | **907543520** |
| pack | `~/src/Strata-data/packs/swift-iq3_xxs` | setup.py output |
| MTP runtime | `~/src/Strata-data/mtp/rt` | `experts.bin` + `dense.bin` |
| json | `~/src/Strata/strata-swift-iq3_xxs.json` | snapshot: `example-strata-swift-iq3_xxs.json` |
| run wrapper | `~/src/Strata/run-swift-iq3_xxs.sh` | |
| this bundle | `ubuntu_handoff/strata-swift-iq3xxs/` | HAND; all scripts `cd` to dirname of $0 |

`--native` and `--ple-gguf` both point at **shard1** of Swift IQ3_XXS (not shard2, not ISTA).

## Which MTP (do not guess)

**Use Swift Flash's own official MTP, packed q2_0.**

- HF repo: `ukisai/Swift-Qwen3.8-Flash-Next` (125B Flash, **not** the 27B dense Swift)
- tensors: BF16 `mtp.*` from that checkpoint (`STRATA_MTP_REPO=swift` in `pack_official_mtp.sh` / `mtp_aria2_fetch.py`)
- pack: `tools/mtp_pack.py --experts q2_0` → `mtp-q2_0.gguf`
- runtime: `tools/mtp_rt.py` → `~/src/Strata-data/mtp/rt/{experts.bin,dense.bin}`
- fastest Strata draft on this stack is **q2_0**, not BF16 sidecar

**Do not use:**

- Unsloth shared-Q8_0 / Unsloth GGUF MTP (llama.cpp-only, trained on original Qwen, Strata will not load it)
- `Qwen/Qwen3.8-Flash-Next` MTP → that is `~/src/Strata-data/mtp-qwen/rt` for **ISTA / family=qwen**, not this launch
- DFlash (Strata has no DFlash runtime)

`pack_official_mtp.sh` default `STRATA_MTP_REPO=swift`. Leave it.

## Chat template / thinking / sampling

Thinking is **not disabled**. Official default effort is **xhigh**, not medium. Client may pass `reasoning_effort` (`xhigh`/`medium`/`low`). Do **not** hardcode effort in the server.

Template lives in the pack tokenizer after setup.py:

`~/src/Strata-data/packs/swift-iq3_xxs/tokenizer/chat_template.jinja`

Copy in this bundle: `tokenizer/chat_template.jinja` (extracted from the Swift pack). Key line:

`resolved_reasoning_effort = reasoning_effort|default('xhigh')`

`launch_swift_iq3xxs.sh` does **not** pass `--chat-template`; Strata uses the pack jinja. `launch_ista_iq3s.sh` is a different family and copies a different jinja — ignore it here.

Official **thinking-mode** sampling (forced over the client via json `sampling.force`):

```
temperature=1.0
top_p=0.95
top_k=20
min_p=0.0
presence_penalty=0.0
repetition_penalty=1.0
force=true
```

Stock Strata **ignores** unknown key `force`. Apply `patches/apply_local.py` so `serve/server.py` honors it (`svc.sampling_force`). Without the patch, clients (Open WebUI, speedtest) overwrite temp/top_p.
On **v0.2.2** the serve code moved and those anchors no longer match: use `patches/port_pin_022.py` (same pin, 0.2.2's anchors; json keys are top-level `sampling` + a `sampling_force` list).

Adaptive MTP knobs (written by `apply_adaptive_mtp.py` into json before start):

```
--spec 6 --mtp-max-t 4 --suffix-draft 3 --spec-min-p 0.5
--kv k8v4
--max-context 206848
--prefill 2048
--vram-reserve-mib 800
--expert-cache auto
```

`model_name` in json: `swift-1.5-iq3_xxs`. API `host=0.0.0.0` `port=8005` `fit_max_tokens=true`.

## China download (mirrors)

Full recipe: `03_downloads_mirrors.md`. Summary:

**Big files (>64MB: GGUF, mmproj, mtp tensors):** Clash 7897 UP → official huggingface.co + aria2 `-c -x 4 -s 4 -k 8M`. Do **not** use hf-mirror as first choice for large files (cas-bridge ~1MB/s). Do **not** use `gh-proxy.com/https://huggingface.co/...` (404).

```bash
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897
ss -ltn | grep 7897   # must listen
aria2c -c -x 4 -s 4 -k 8M --file-allocation=none --max-tries=0 --retry-wait=8 \
  --timeout=120 --connect-timeout=30 --disable-ipv6=true \
  --allow-overwrite=true --auto-file-renaming=false \
  --http-proxy="$https_proxy" \
  -o <file> \
  "https://huggingface.co/<org>/<repo>/resolve/main/<file>"
```

7897 down: unset proxy, `HF_ENDPOINT=https://hf-mirror.com`, same aria2 flags. `pack_official_mtp.sh` already probes 7897 and falls back.

**GitHub (Strata zip, llama.cpp pin, get-pip):** `https://gh-proxy.com/https://github.com/...`

**pip:** `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`

HF repos:

- model: `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF`
- MTP source: `ukisai/Swift-Qwen3.8-Flash-Next` (mtp.* only)
- Strata: `Niko1221/Strata` tag `v0.1.38`
- IQ4 GGUF: `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF` (`IQ4_XS/` shards)
- llama.cpp pin: `3cf03257f219afbe7334045ff7c6a06ac68c627d` (`LLAMA_CPP_COMMIT` in setup.py)

## Compile Strata (sm_86)

```bash
export STRATA_ROOT=$HOME/src/Strata
bash clone_strata.sh          # gh-proxy zip v0.1.38 → ~/src/Strata if missing
python3 patches/apply_local.py "$STRATA_ROOT"
bash compile_engine.sh        # nvcc from sglang-0519 cu13, arch=86, llama.cpp pin, vision GPU
```

`compile_engine.sh` facts:

- `CMAKE_CUDA_ARCHITECTURES=86` (override CMakeLists 120)
- `CCCL_DISABLE_CTK_COMPATIBILITY_CHECK=1` required for pip CUDA 13.4 nvcc
- llama.cpp zip via gh-proxy at pin above; `STRATA_GGML_DIR=$ROOT/third_party/llama.cpp`
- targets: `strata` then `strata-vision` (`STRATA_VISION_CUDA=ON`)
- copies `data/expert-profile.bin` and `data/draft_vocab.bin` from this bundle into `$STRATA_ROOT/data/`
- writes `engine/BUILD.json` `source=local version=0.1.38 archs=[86] vision=gpu`
- **always** set `STRATA_SKIP_BUILD=1` for later `setup.py` so it does not fetch sm_120 prebuilt or rebuild for hash mismatch

Do not `apt install cuda-toolkit` on this PC; use the pip nvcc path. Override with `NVCC=` / `CUDA_PIP=` if the venv moves.

After compile: `STRATA_SKIP_BUILD=1` is already exported in `setup_swift_iq3xxs.sh`.

## Pack MTP

```bash
# needs Strata .venv + tools/mtp_pack.py from the tree
bash pack_official_mtp.sh     # STRATA_MTP_REPO=swift → ~/src/Strata-data/mtp/rt
```

Exists-skip: `mtp-manifest.json` / `mtp-q2_0.gguf` / `rt/experts.bin`. Ready when `rt/experts.bin` and `rt/dense.bin` exist.

## Download GGUF

```bash
bash scripts/download_swift15_flash_gsq_iq3.sh
# dest: ~/models/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/
# verify sizes 39785790560 / 36180282560 / 907543520 then touch *.done
```

setup.py HEAD to huggingface.co hangs if a file exists without `.done` and 7897 is down. After a finished download, write sibling `*.gguf.done`.

## Pack model + json

```bash
export STRATA_SKIP_BUILD=1 HF_ENDPOINT=https://hf-mirror.com
bash setup_swift_iq3xxs.sh
# family=swift model=IQ3_XXS vision=gpu kv=k8v4 gpus=1,0
# setup.py may clamp ctx to 128K on 64GB; launch script raises to 206848 via apply_adaptive_mtp.py
```

Produces `~/src/Strata/strata-swift-iq3_xxs.json` and `run-swift-iq3_xxs.sh`.

## Launch

```bash
bash launch_swift_iq3xxs.sh
# env knobs: CTX=206848 PREFILL=2048 VRAM_RESERVE=800 KV=k8v4 PORT=8005 STRATA_ROOT
# kills 8000/8001/8080/$PORT, stops user unit sglang-dspark, apply_adaptive_mtp.py, exec run-*.sh
```

CUDA libs: `LD_LIBRARY_PATH=$HOME/venvs/sglang-0519/lib/python3.12/site-packages/nvidia/cu13/lib`

Ready log: `~/src/Strata/strata-swift-iq3_xxs.log`. GPU vs CPU experts:

```bash
python3 scripts/print_expert_counts.py ~/src/Strata/strata-swift-iq3_xxs.json
```

Target on this box: as many MoE experts on GPU as VRAM allows (was 18/20; 800 MiB reserve aims ≥19/20). If `mtp: the draft head does not fit`, raise `VRAM_RESERVE` (700–1000), do not drop MTP.

## Order on a new clone of this folder

1. `clone_strata.sh`
2. `python3 patches/apply_local.py`   # SKIP_BUILD, sampling.force, dual-GPU resident-on-split
3. `compile_engine.sh`
4. `scripts/download_swift15_flash_gsq_iq3.sh` (or skip if sizes already match)
5. `pack_official_mtp.sh`
6. `setup_swift_iq3xxs.sh`
7. `launch_swift_iq3xxs.sh`

IQ4 (VRAM then RAM; [RESIDENT_ON_SPLIT.md](RESIDENT_ON_SPLIT.md)): after 1–3 and 5, `scripts/download_swift15_flash_iq4xs.sh` → `setup_swift_iq4xs.sh` → `launch_swift_iq4xs.sh`.

Remap `STRATA_ROOT`, `GGUF_DIR`, `CUDA_PIP`, `NVCC`, `STRATA_DATA` if not this PC. Keep arch=86 for 3080. Dual-GPU split 26 / gpus=1,0 is this machine's layout (GPU1 first).

## Files in this folder

| file | role |
| --- | --- |
| `launch_swift_iq3xxs.sh` | IQ3 start (RAM arena) |
| `launch_swift_iq4xs.sh` | IQ4 start (`--resident-experts`, GPU then RAM) |
| `setup_swift_iq3xxs.sh` | setup.py pack + first apply_adaptive_mtp |
| `setup_swift_iq4xs.sh` | IQ4 pack (`iq_pack --compat-bf16`) + json |
| `apply_adaptive_mtp.py` | upsert spec/kv/ctx/prefill/reserve + sampling.force; `--resident-experts` |
| `compile_engine.sh` | sm_86 engine+vision |
| `clone_strata.sh` | v0.1.38 via gh-proxy zip |
| `pack_official_mtp.sh` | Swift mtp.* → q2_0 → mtp/rt |
| `mtp_aria2_fetch.py` | fetch mtp tensors (7897 or hf-mirror) |
| `scripts/download_swift15_flash_gsq_iq3.sh` | IQ3 GGUF+mmproj |
| `scripts/download_swift15_flash_iq4xs.sh` | IQ4 GGUF shards |
| `scripts/print_expert_counts.py` | parse last loaded chunk in log |
| `patches/apply_local.py` | STRATA_SKIP_BUILD + sampling.force + resident-on-split；v0.2.2 起还提供 `apply_same_chat_rewind`（插话不占停放槽的引擎补丁） |
| `patches/apply_codex_side_request.py` | 2026-10-24 Codex 旁路 / 同线程替换：`conv=` + `drop_replaced`。幂等，不编译。运行目录的 `serve/server.py` 作第二个参数 |
| `patches/port_pin_022.py` | 把「钉 sampling」补丁搬到 v0.2.2 的 `serve/server.py`（4 个精确锚点、幂等、py_compile 自检） |
| `patches/resident-on-split-v0.1.38.patch` | dual-GPU GPU-then-RAM (see RESIDENT_ON_SPLIT.md) |
| `RESIDENT_ON_SPLIT.md` | method + Strata issues/PRs |
| `tokenizer/chat_template.jinja` | Swift pack template, default xhigh |
| `data/expert-profile.bin` | required by engine `--expert-profile` |
| `data/draft_vocab.bin` | copied onto mtp/rt |
| `example-strata-swift-iq3_xxs.json` | working json snapshot this PC |
| `example-run-swift-iq3_xxs.sh` | working run wrapper snapshot |
| `example-BUILD.json` | working BUILD.json snapshot |
| `03_downloads_mirrors.md` | aria2 7897 recipe |
| `scripts/decode_ctx_tps.py` | bench：8k→512k 八档十类 sweep，写 results/*.json |
| `scripts/decode_bench.py` | bench：单次测量（服务端 timings 优先） |
| `scripts/needle_512k.py` | **512K NIAH 埋点**：5 针 × 5 深度，单请求，严格 `KEY=<code>` 判分（复用 decode_bench 原语） |
| `launch_swift_abliterated_iq3s_q8ple.sh` | 256K IQ3_S+PLE 启动器（单流订死；JSON_ONLY=1 只写 json 不启动） |
| `launch_swift_abliterated_iq3s_q8ple_512k.sh` | 512K 启动器（Swift 版，历史）：ctx 524288 + yarn×2 + 旗杆 `6×16384` + 停放 `16384/3/18432/2560`，`PROMPT_CACHE_TAIL` 默认 0 |
| `launch_davidau_iq3s_q8ple_256k.sh` | **当前开机在用（0.2.2 @256K）**：DavidAU IQ3_S + Q8_0 PLE，ctx 262144 无 YaRN、`PREFILL=16384`、停放 `20480/5`、旗杆 `6×16384` |
| `example-strata-davidau-iq3_s-q8ple-256k.json` | 上面那份的 live json 快照（2026-10-22，含 `effort_position: "end"`） |
| `results/davidau_iq3_256k.json` | **0.2.2 @256K 六档十类原始数据（2026-10-22，decode AVG 94.6 / 256k 冷读 78.1 s）** |
| `launch_davidau_iq3s_q8ple_512k.sh` | DavidAU 512K 启动器（0.1.45 时代的开机在用，YaRN×2） |
| `example-strata-davidau-iq3_s-q8ple.json` | 512K live json 快照（2026-10-20，含 `effort_position: "end"`） |
| `results/davidau_iq3.json` | DavidAU 八档十类原始数据（2026-10-20，decode AVG 79.7 / 512k 冷读 172.0 s） |
| `launch_swift_abliterated_iq3s_q8ple_multi.sh` | multi-slot 入口（PARALLEL / BATCH_MODE=draft|pipe） |
| `systemd/strata-swift.service` | user unit；ExecStart=256K 启动器；**unit env 才是 PARALLEL/CONV_* 杠杆** |
| `patches/ple-only-native-dense-v0.1.43.1.patch` | #1233 外挂 PLE 启动修复（**必打**；44.1 树原样可打） |
| `patches/prompt-cache-tail-layer-split.patch` | 已从工作树删除（`459f0c6`），存档在 git 历史 `16c3762`；双卡 `--prompt-cache-tail`，现由 same-chat rewind 取代 |
| `results/prompt_cache_tail_split_20261018.txt` | 2026-10-18 同前缀分叉：B 16384 reused / 5.0 s vs A 0 reused / 11.5 s |
| `results/43_1_patch_1.json` | 256K 基线原始数据（43.1+PREFILL 8192） |
| `results/44_1_1.json` / `results/iq3_4096_1024reserve.json` | 44.1 全量（回滚依据）/ 43.0 对照 |
| `results/512_yarn_1.json` + `results/needle_512k(_full).json` + `results/needle_sanity.json` | **512K 八档测速 + 埋点 5/5 原始数据（2026-10-16）** |
| `example-strata-swift-abliterated-iq3_s-q8ple.json` | **512K live json 快照（2026-10-16）** |

Not this launch: `launch_ista_iq3s.sh`, `pack_qwen_mtp.sh`, ISTA GGUF, Unsloth GGUF.


## 多 slot ≠ 并发（multi-slot 须知）


- **停放槽（conversation-cache slots）≠ 并发**：槽 6 = 6 条长对话的 KV 常驻 RAM 停放（240-256K 每条约 3.5-4.6 GiB），
  回访串行取用。本机实测 256K 回访 **4.9 / 5.9 s**（prompt 全量命中缓存）；4090 5×240K 回访 0.7-2.1 s。
- **真并发** = json `parallel: N`（serve 翻译成 `--batch N`，每 slot 一个独立会话，含各自 GDN/QSA-KV/indexer/PLE 历史）。
  0.1.43.0 时代实测（parallel 2，双 3080）：两路**同时打字**各仅 **9-11 t/s**（batch slot 无 MTP draft，每窗 1 token）；
  被 slot 占用时单请求 66-74（纯单流 93）；每 slot 吃 ~1.55 GiB 显存/stage（专家热层 -22%）+ ~11.6 GiB pinned RAM。
- v0.1.43.1 新行为（[#2038](https://github.com/Niko1221/Strata/issues/2038) 已修）：layer-split `--batch` 默认 pipelined 组
  （官方 4 流/2 GPU +25-30%，无 slot draft）；`--batch-mtp` + `--batch-groups 1` = slot 恢复 MTP draft（串行窗），
  官方原话"流少时或 prompts 稀疏时可能更快"——本机 2 流 **待 A/B 实测**。
- 未修风险（至 v0.1.43.1）：#1503 slot+停放可能串上下文（抽查各会话 marker）；#1603 双长上下文会话并发可能挂死
  （判据：日志 `running:0 queued:0` 且两分钟不动）；0.1.43.1 修的是 #1994（slot 400 "no such free slot" → 503 retry）。


## Benchmark 脚本（产出上面各表）


`scripts/decode_ctx_tps.py`（sweep 8k→512k 八档，10 类 scenario）+ `scripts/needle_512k.py`（512K 埋点：NIAH 5 针×5 深度，一条请求，严格子串判分，产出上面埋点小节）+ `scripts/decode_bench.py`（单次测量：
读服务端 `timings` 的 `prompt_per_second`/`predicted_per_second`/`cache_n`，TTFT 与 decode 分开计，纯 stdlib）。
跑法：`python3 scripts/decode_ctx_tps.py`（会问结果名，JSON 写到 `~/ubuntu_handoff/results/<名>.json`，就是表旁那两个文件）。
8k 档 pre 1608 → 256k 档 2928 的差别是短 prompt 吃不满 `PREFILL=8192` 大窗，档位越高越吃满。


## Parallel-2 slot 实测（strata **v0.1.43.1** + `--batch-mtp` draft 模式，2026-10-13 21:08 boot，A 局）


启动：`systemctl --user set-environment PARALLEL=2 BATCH_MODE=draft && systemctl --user restart strata-swift`
（unit env 是杠杆——launcher 只认 env；boot 日志出现 `--pipeline-windows 2 is off: not with --batch slots` = slot 生效）。
测试：每请求 300 tok 流式，读服务端 timings。

| 场景 | t/s | draft 接受 | 说明 |
| --- | --- | --- | --- |
| 单打（另一 slot 空闲） | **33.5 / 39.5** | solo2 170/181=94% | draft→verify 串行窗，比纯单流（101-152）慢 |
| 两路同时打 | **47.2 + 54.0（合计 ~101）** | 0/0（窗互扣） | 两 slot 的 draft/verify 窗互相填缝 |

- 结论：draft slot 下单打最慢；两路都在时总吞吐≈纯单流。**两路同时打体验=单流水平（每路 ~50），代价是单打 150→35**。
- 对照 0.1.43.0（无 draft slot）：两路各仅 9-11 t/s；draft 模式把双路并发提了 ~5×。
- B 局 `BATCH_MODE=pipe`（43.1 默认 pipelined 组，无 slot draft）待测。
- 回单流：`systemctl --user unset-environment PARALLEL BATCH_MODE && systemctl --user restart strata-swift`。

## B 局 pipe（pipelined groups）实测 + A/B 定案（strata **v0.1.43.1**，2026-10-13 21:43，同一 300-tok 脚本）

| mode | 单打 t/s | 两路同时打 t/s | draft |
| --- | --- | --- | --- |
| A `draft`（--batch-mtp 串行窗） | 33.5 / 39.5 | **47.2 + 54.0（合计 ~101）** | 串行窗内 94% 接受 |
| B `pipe`（默认 pipelined 组） | 48.4 / 71.1 | 43.4 + 38.3（合计 ~82） | 两路时 0/0（pipelined 组不跑 slot draft） |
| 参考：纯单流（parallel 关） | **101.8 / 152.8** | — | 95% |

- **定案：2 流用 A `draft`**（并发 101 > 82）——官方"流少/prompts 稀疏时串行 MTP 窗更快"在双 3080 复现；
  官方 +25-30% 的 pipelined 优势是 4 流场景。`multi.sh` 默认保持 draft。
- 两种 slot 模式单打都远慢于纯单流（150 → 33-71）：**没人在打字时才用单流**；回单流一条命令（见上节）。
- 0.1.43.0 时代 9-11 t/s/路 → 43.1 的 38-54 t/s/路：slot 并发从"不可用"到"能用"。
