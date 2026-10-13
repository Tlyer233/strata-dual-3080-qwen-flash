# AGENT: strata-dual-3080-qwen-flash

Reproduce the dual RTX 3080 20GB + 64 GB Strata launches in this folder. This file is for an agent, not a human. Paths below are the reference PC unless remapped.

**IQ3_XXS** (`launch_swift_iq3xxs.sh`): full RAM arena, ~70–80 tok/s decode on this box. Quant fits 64 GB.

**IQ4_XS** (`launch_swift_iq4xs.sh`): dual-GPU **VRAM first, then RAM**. Upstream Strata v0.1.38 does not merge this yet. Method, patch, and citations: **[RESIDENT_ON_SPLIT.md](RESIDENT_ON_SPLIT.md)** ([Niko1221/Strata#604](https://github.com/Niko1221/Strata/issues/604), [#364](https://github.com/Niko1221/Strata/issues/364), [#384](https://github.com/Niko1221/Strata/issues/384), [PR #639](https://github.com/Niko1221/Strata/pull/639), [PR #640](https://github.com/Niko1221/Strata/pull/640), [PR #731](https://github.com/Niko1221/Strata/pull/731)).

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
| `patches/apply_local.py` | STRATA_SKIP_BUILD + sampling.force + resident-on-split |
| `patches/resident-on-split-v0.1.38.patch` | dual-GPU GPU-then-RAM (see RESIDENT_ON_SPLIT.md) |
| `RESIDENT_ON_SPLIT.md` | method + Strata issues/PRs |
| `tokenizer/chat_template.jinja` | Swift pack template, default xhigh |
| `data/expert-profile.bin` | required by engine `--expert-profile` |
| `data/draft_vocab.bin` | copied onto mtp/rt |
| `example-strata-swift-iq3_xxs.json` | working json snapshot this PC |
| `example-run-swift-iq3_xxs.sh` | working run wrapper snapshot |
| `example-BUILD.json` | working BUILD.json snapshot |
| `03_downloads_mirrors.md` | aria2 7897 recipe |
| `scripts/decode_ctx_tps.py` | bench：8k-256k 十类 sweep，写 results/*.json |
| `scripts/decode_bench.py` | bench：单次测量（服务端 timings 优先） |

Not this launch: `launch_ista_iq3s.sh`, `pack_qwen_mtp.sh`, ISTA GGUF, Unsloth GGUF.

## v0.1.43.1 — Swift 1.5 abliterated IQ3_S + sleepyeldrazi Q8_0 PLE（当前配置，2026-10-13）

引擎 **strata v0.1.43.1**（本机源码编译 sm_86，pip CUDA 13 nvcc），**单流**（parallel 关），`--serve` :8005，
ctx **268288**，kv `k8v4`，**prefill 8192**（launcher 里 `PREFILL=8192`，prefill 最大提速杠杆），
双 3080 layer-split gpus=1,0 split=26，`--pipeline-windows 2`，`--resident-experts`（GPU 先、RAM 补 24.63 GiB pinned），
adaptive MTP（spec 6 / mtp-max-t 4 / suffix-draft 3 / min_p 0.5），conversation-cache 停放默认
（预算 19456 MiB / 槽 6 / RAM 地板 4096 MiB / 短请求门槛 4096 tok / 检查点 28×every 131072 tok）。

- **必打本地补丁**（上游 [Niko1221/Strata#1233](https://github.com/Niko1221/Strata/issues/1233)，**v0.1.43.1 未修**）：
  `patches/ple-only-native-dense-v0.1.43.1.patch` — 外挂单张量 PLE-only 表（`per_layer_token_embd.weight`）不加入
  native dense；否则启动即 `native dense: duplicate split shard number` 崩溃循环（PLE 的 split.no 与 Swift shard 2 撞号）。
  编译前 `git apply`/`patch -p1` 到 generate.cpp。
- 默认开机 = systemd user unit `systemd/strata-swift.service` → `launch_swift_abliterated_iq3s_q8ple.sh`
  （launcher **每次启动强制清 json 里的 parallel/--batch-groups/--batch = 开机订死单流**；停放参数走 env
  `CONV_CACHE_MIB` 等，`systemctl --user set-environment` 覆盖，json 不是杠杆）。
- 多流（multi-slot）另开：`launch_swift_abliterated_iq3s_q8ple_multi.sh`（默认 PARALLEL=2；`JSON_ONLY=1` 只写 json 不启动）。
- json 快照：`example-strata-swift-abliterated-iq3_s-q8ple.json`。

## Benchmark（strata **v0.1.43.1** + PLE 补丁，PREFILL=8192，单流；bench 2026-10-13 20:23，原始数据 `results/43_1_patch_1.json`）

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

## Benchmark 脚本（产出上面那张表）

`scripts/decode_ctx_tps.py`（sweep 8k→256k，10 类 scenario）+ `scripts/decode_bench.py`（单次测量：
读服务端 `timings` 的 `prompt_per_second`/`predicted_per_second`/`cache_n`，TTFT 与 decode 分开计，纯 stdlib）。
跑法：`python3 scripts/decode_ctx_tps.py`（会问结果名，JSON 写到 `~/ubuntu_handoff/results/<名>.json`，就是表旁那两个文件）。
8k 档 pre 1608 → 256k 档 2928 的差别是短 prompt 吃不满 `PREFILL=8192` 大窗，档位越高越吃满。

## Model / PLE / MTP assets（v0.1.43.1 实测这套；二进制不进 git，按 03_downloads_mirrors.md 的 aria2/7897 下法）

| 资产 | HuggingFace repo | 本地路径（参考机） | 大小 |
| --- | --- | --- | --- |
| 主模型 IQ3_S（abliterated）2 分片 + mmproj BF16 + SHA256SUMS | `slider-meister-pub/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-GGUF` | `~/models/slider-meister-pub/Swift-1.5-…-IQ3_S-GGUF/` | 79 G |
| **PLE（n-gram 表，外挂）** Q8_0 分片 | `sleepyeldrazi/Qwen3.8-Flash-Next-GSQ-RCO-Q8_0-PLE-GGUF` | `~/models/sleepyeldrazi/…-Q8_0-PLE-GGUF/`（本机现存 `-00002-of-00002.gguf`） | 51 G |
| MTP（官方 Swift mtp.* → q2_0 packed） | 源 `ukisai/Swift-Qwen3.8-Flash-Next`，用本仓 `pack_official_mtp.sh` 打包 | `~/src/Strata-data/mtp/rt/` | 0.79 G |
| Strata pack（iq_pack `--compat-bf16` 产物：dense.bin/conversions/index/native_experts） | 由 setup 生成，非下载 | `~/src/Strata-data/packs/swift-abliterated-iq3_s/` | 1.5 G |

- `--native` = IQ3_S 主分片（split=26 靠它），`--ple-gguf` = sleepyeldrazi Q8_0 PLE（**必打 #1233 补丁**，见上）。
- 下载：`https://huggingface.co/<repo>/resolve/main/<file>`，有梯子 `https_proxy=http://127.0.0.1:7897` + aria2c `-c`；
  无梯子换 `https://hf-mirror.com/<repo>/resolve/main/<file>`。校验 SHA256SUMS（IQ3_S 仓自带）。

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
