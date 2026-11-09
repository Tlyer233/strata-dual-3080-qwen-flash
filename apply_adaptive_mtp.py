#!/usr/bin/env python3
"""Patch a Strata serve JSON with engine 0.1.7+ adaptive MTP.

Strata has no llama.cpp-style draft-mtp-adaptive flag. Adaptive decode is:

  --spec 6 --mtp-max-t 4 --suffix-draft 3 --spec-min-p 0.5 --draft-vocab cjk

That is setup.py's --spec 4 --spec-min-p 0.5 plus the engine default suffix
drafter (generate.cpp: suffix_draft=3 raises spec to min(spec+2,8) and caps
MTP at the original spec). DraftPolicy then picks MTP vs prompt-lookup per
verify round from measured E(tokens)/ms.

Also: 200k context on 2x20GB with --kv k8v4 (KV stays in VRAM; setup would
otherwise clamp IQ3 262K to 128K on 64 GB RAM). Vision stays on GPU.
n_ctx is 206848 so the 200k tier (204800) plus generate slack still fits.
Prefill 2048: dual-GPU cannot borrow cache slots. Remaining VRAM goes to experts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def upsert(args: list[str], flag: str, *vals: str) -> list[str]:
    out: list[str] = []
    skip = 0
    i = 0
    while i < len(args):
        if skip:
            skip -= 1
            i += 1
            continue
        if args[i] == flag:
            skip = len(vals)
            i += 1
            continue
        out.append(args[i])
        i += 1
    out.append(flag)
    out.extend(vals)
    return out


def drop(args: list[str], flag: str, nvals: int = 1) -> list[str]:
    out: list[str] = []
    skip = 0
    i = 0
    while i < len(args):
        if skip:
            skip -= 1
            i += 1
            continue
        if args[i] == flag:
            skip = nvals
            i += 1
            continue
        out.append(args[i])
        i += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--context", type=int, default=206848)
    ap.add_argument("--kv", default="k8v4")
    ap.add_argument("--spec", default="6")
    ap.add_argument("--mtp-max-t", default="4")
    ap.add_argument("--suffix-draft", default="3")
    ap.add_argument("--spec-min-p", default="0.5")
    # Empty: leave the engine's PCIe probe. Dual-GPU cannot borrow cache
    # slots, so keep an explicit 2048 and spend the rest of VRAM on experts.
    ap.add_argument("--pcie-frac", default="")
    ap.add_argument("--prefill", default="2048")
    ap.add_argument("--expert-cache", default="auto")
    ap.add_argument("--vram-reserve-mib", default="1600")
    ap.add_argument("--pipeline-windows", default="",
                    help="0.1.40 dual-GPU overlap (1=short prompt, 2=decode windows). Empty: leave/omit.")
    ap.add_argument("--layer-split", default="26")
    ap.add_argument("--gpus", default="1,0",
                    help="CUDA indices, comma-separated.")
    ap.add_argument("--mtp", default="",
                    help="engine --mtp dir (empty: leave whatever setup wrote)")
    # Unsloth-only. launch_swift_iq3xxs.sh must not pass these — Swift keeps
    # packed experts.bin in the RAM arena, not GGUF mmap / resident-budget.
    ap.add_argument("--resident-budget-gib", default="",
                    help="Unsloth-only. Do not pass from launch_swift_iq3xxs.sh.")
    ap.add_argument("--mmap-experts", action="store_true",
                    help="Map experts from GGUF; RAM stays empty. Do not pass from IQ3/IQ4 RAM paths.")
    ap.add_argument("--resident-experts", action="store_true",
                    help="GPU expert cache first, then copy the complement into RAM. IQ4 dual-GPU path.")
    ap.add_argument("--chat-template", default="",
                    help="copy this jinja over the pack tokenizer chat_template.jinja")
    ap.add_argument("--ple-gguf", default="",
                    help="override --ple-gguf (Q8_0 companion shard, etc.)")
    a = ap.parse_args()
    path = Path(a.config)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    args = list(cfg.get("args") or [])
    args = drop(args, "--kv-resident", 1)
    args = upsert(args, "--max-context", str(a.context))
    args = upsert(args, "--kv", a.kv)
    args = upsert(args, "--spec", a.spec)
    args = upsert(args, "--mtp-max-t", a.mtp_max_t)
    args = upsert(args, "--suffix-draft", a.suffix_draft)
    args = upsert(args, "--spec-min-p", a.spec_min_p)
    if a.pcie_frac:
        args = upsert(args, "--pcie-frac", a.pcie_frac)
    else:
        args = drop(args, "--pcie-frac", 1)
    args = upsert(args, "--prefill", a.prefill)
    args = upsert(args, "--expert-cache", a.expert_cache)
    args = upsert(args, "--vram-reserve-mib", a.vram_reserve_mib)
    if a.mtp:
        args = upsert(args, "--mtp", a.mtp)
    if a.ple_gguf:
        args = upsert(args, "--ple-gguf", a.ple_gguf)
    if a.pipeline_windows:
        args = upsert(args, "--pipeline-windows", a.pipeline_windows)
        # 0.1.41: --remote-expert-opt turns --pipeline-windows off. Keep the
        # decode overlap; helper-cache is not the GPU-then-RAM split path.
        args = drop(args, "--remote-expert-opt", 0)
    gpus = [int(x) for x in a.gpus.split(",") if x.strip() != ""]
    if not gpus:
        raise SystemExit("--gpus is empty")
    # Opt-in only. Swift IQ3 launch does not pass these, so its packed
    # experts.bin arena and dual-GPU layer split stay as they are.
    if a.mmap_experts:
        args = drop(args, "--resident-budget-gib", 1)
        args = drop(args, "--resident-experts", 0)
        args = drop(args, "--mmap-experts", 0)
        args.append("--mmap-experts")
    elif a.resident_experts:
        args = drop(args, "--mmap-experts", 0)
        args = drop(args, "--resident-experts", 0)
        args.append("--resident-experts")
        if a.resident_budget_gib:
            args = upsert(args, "--resident-budget-gib", a.resident_budget_gib)
        else:
            args = drop(args, "--resident-budget-gib", 1)
        # Do not add --remote-expert-opt: with --pipeline-windows it disables
        # the overlap; without it the split already fills both GPU caches.
    elif a.resident_budget_gib:
        args = drop(args, "--mmap-experts", 0)
        args = drop(args, "--resident-experts", 0)
        args = upsert(args, "--resident-budget-gib", a.resident_budget_gib)
    else:
        # Swift IQ3: full RAM arena. Strip leftovers.
        args = drop(args, "--mmap-experts", 0)
        args = drop(args, "--resident-experts", 0)
        args = drop(args, "--resident-budget-gib", 1)
        args = drop(args, "--remote-expert-opt", 0)
    cfg["args"] = args
    cfg["gpu"] = gpus if len(gpus) > 1 else gpus[0]
    cfg["host"] = "0.0.0.0"
    cfg.pop("api_key", None)
    cfg["allowed_hosts"] = "*"
    cfg["fit_max_tokens"] = True
    if len(gpus) > 1:
        cfg["layer_split"] = a.layer_split
    else:
        cfg.pop("layer_split", None)
    cfg["draft_vocab"] = cfg.get("draft_vocab") or "cjk"
    # 采样整块钉死: 客户端 temperature/top_p/top_k/... 一律不认。
    # 靠本地补丁 serve/server.py 的 sampling_force 列表 (官方优先级 request 最高)。
    cfg["sampling"] = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    cfg["sampling_force"] = [
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
    ]
    # --- effort 放末尾 (2026-10-19, DETAILS.md:903-908 / #458) ---
    # effort 指令默认在 prompt 最前, 只改 reasoning_effort 的请求会把整段对话重读一遍。
    # "end" 把 effort 渲染成答案前的一个短 system turn, checkpoint 停在它前面 -> 下一条请求复用会话。
    # 官方提醒: 模型在训练时没放过的位置看到 effort, 跟得如何没测过。回旧行为: EFFORT_POSITION=start
    pos = os.environ.get("EFFORT_POSITION", "end")
    if pos == "start":
        cfg.pop("effort_position", None)
    else:
        cfg["effort_position"] = pos
    tpl = Path(a.chat_template) if a.chat_template else None
    if tpl:
        if not tpl.is_file():
            raise SystemExit(f"chat template not found: {tpl}")
        dest = Path(cfg["tokenizer"]) / "chat_template.jinja"
        dest.write_text(tpl.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        print(f"chat template: {tpl} -> {dest}")
    path.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
    print(f"adaptive MTP written: {path}")
    print("  --spec", a.spec, "--mtp-max-t", a.mtp_max_t,
          "--suffix-draft", a.suffix_draft, "--spec-min-p", a.spec_min_p,
          "--pcie-frac", a.pcie_frac or "(probe)")
    print("  --max-context", a.context, "--kv", a.kv,
          "--prefill", a.prefill, "--expert-cache", a.expert_cache,
          "--vram-reserve-mib", a.vram_reserve_mib,
          "--mtp", a.mtp or "(unchanged)")
    if a.mmap_experts or a.resident_budget_gib:
        print("  --mmap-experts", "yes" if a.mmap_experts else "no",
              "--resident-budget-gib", a.resident_budget_gib or "(none)")
    if a.resident_experts:
        print("  --resident-experts yes  (no --remote-expert-opt; pipeline-windows stays on)")
    if a.pipeline_windows:
        print("  --pipeline-windows", a.pipeline_windows)
    print("  gpus", gpus, "layer-split",
          (a.layer_split if len(gpus) > 1 else "(none, single GPU)"))
    print("  host 0.0.0.0  (Tailscale / LAN)")
    print("  no api_key  (Bearer not required; allowed_hosts=* so Host check is off)")
    print("  fit_max_tokens=true  (clamp max_tokens instead of 400)")
    print("  sampling: " + (json.dumps(cfg.get("sampling"), ensure_ascii=False) +
                            " 钉住 " + json.dumps(cfg.get("sampling_force", []))
                            if "sampling" in cfg else "off (全按 model card / 客户端)"))
    print("  effort_position: " + str(cfg.get("effort_position", "start (默认)")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
