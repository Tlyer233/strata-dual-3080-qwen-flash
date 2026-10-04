# Dual-GPU GPU-then-RAM (not merged upstream)

Strata already has `--resident-experts` on **one GPU**: fill the VRAM expert cache, copy only the experts that card does not hold into RAM, leave PLE / n-gram on the GGUF. On **several GPUs** (a layer split) v0.1.38 still does **not** do that. Setup rewrites the flag to `--mmap-experts` (RAM stays empty; experts stream from SSD). The engine used to refuse `--resident-cpu-experts` with `--layer-split`. The only other dual-GPU path is the full RAM arena (copy every expert into RAM first), which on 64 GB + a ~61 GiB IQ4 expert set OOMs the desktop GPU.

This recipe patches v0.1.38 so `--resident-experts` plus `--gpus 1,0` / `layer_split=26` is the middle path:

1. Each GPU fills its own expert cache (hottest experts of **its** layers).
2. `pin_cache_complement` is told about later GPUs (`additional_gpu_pairs`; the hook already existed, generate.cpp passed `{}`).
3. RAM page-locks only the experts **no card holds**.
4. PLE / n-gram stay mapped from the GGUF (disk + OS cache).
5. If the whole complement does not fit, the soft path keeps the hottest slice (`kResidentWhatFits`); the rest stay on the files. It does **not** pin the full arena.

Measured on this box (2× RTX 3080 20 GB + 64 GB DDR4, Swift bartowski IQ4_XS, engine 0.1.38 + this patch):

| tier | what landed |
| --- | --- |
| GPU caches | 10585 slots (CUDA0 5500 + CUDA1 5085), ~26 GiB |
| RAM complement | **34.69 GiB** page-locked |
| omitted (already on GPU1) | 5085 experts |
| disk | PLE / n-gram + mmap fallback for those 5085 |

Log lines:

```
strata generate: resident RAM mode with a layer split: ...
strata generate: resident complement omits 5085 experts already on later GPUs
FileExpertSource: allocating 34.69 GiB page-locked cache complement
strata generate: resident RAM mode: 34.69 GiB of experts in RAM (page-locked), 10585 in GPU caches
```

Leave `--vram-reserve-mib` ≥ 1024 (1100 here). 500 left 188 MiB free; the first prefill died with `cublasGemmEx: cuBLAS status 14`.

IQ3_XXS on this box still uses the **full RAM arena** (`launch_swift_iq3xxs.sh`, no `--resident-experts`). That quant fits 64 GB. Do not force this patch's flag onto IQ3.

## Apply

```
python3 patches/apply_local.py "$HOME/src/Strata"   # includes resident-on-split-v0.1.38.patch
bash compile_engine.sh                              # rebuild strata after the patch
```

Idempotent. Against **v0.1.38** only.

## Upstream (not merged)

`--resident-experts` on one GPU is first-class. The split case is explicitly deferred.

| | |
| --- | --- |
| [#604](https://github.com/Niko1221/Strata/issues/604) | Resident expert placement on a layer split — still on the list. Niko: a real resident mode across a split is planned. `additional_gpu_pairs` is already in `pin_cache_complement`. |
| [#364](https://github.com/Niko1221/Strata/issues/364) | 32 GB RAM + 2×3090: dual GPU only after forcing a split; later setup maps that to mmap. |
| [#384](https://github.com/Niko1221/Strata/issues/384) | Layer split + low-RAM resident: setup/engine fall back to `--mmap-experts`. |
| [#498](https://github.com/Niko1221/Strata/issues/498) | UD-Q4_K_XL `--resident-budget-gib` has no layer split; without it dual GPU wants ~135 GB RAM. |
| [#737](https://github.com/Niko1221/Strata/issues/737) | #498 follow-up: allow confirm/override just below 135 GB. |
| [#730](https://github.com/Niko1221/Strata/issues/730) | `--resident-budget-gib` alloc fail → warning, RAM empty, experts from SSD. |

Related PRs (VRAM / arena / disjoint slots — not this patch, but the same stack):

| | |
| --- | --- |
| [#639](https://github.com/Niko1221/Strata/pull/639) | Layer split: each GPU loads only its own layers' dense weights (more room for expert cache). |
| [#640](https://github.com/Niko1221/Strata/pull/640) | `STRATA_ARENA_MMAP=1`: expert arena as a read-only mapped file (small-RAM machines). |
| [#731](https://github.com/Niko1221/Strata/pull/731) | Avoid duplicate expert slots across helper and primary GPUs (`STRATA_DISJOINT_ADAPT=1`). |

Single **48 GB** card + 64 GB RAM does **not** need this patch. Stock `--resident-experts` is GPU then RAM. A ~70–80 GiB Q4 expert set fits that box; a ~100 GiB expert set still needs an SSD tail (`--resident-budget-gib` / soft mmap).

## Launch (IQ4 on this recipe)

```
bash scripts/download_swift15_flash_iq4xs.sh
bash setup_swift_iq4xs.sh
bash launch_swift_iq4xs.sh    # --resident-experts, VRAM_RESERVE=1100
```
