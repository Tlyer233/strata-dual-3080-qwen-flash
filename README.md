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

Not this launch: `launch_ista_iq3s.sh`, `pack_qwen_mtp.sh`, ISTA GGUF, Unsloth GGUF.
