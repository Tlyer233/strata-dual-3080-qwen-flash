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
    ap.add_argument("--layer-split", default="26")
    ap.add_argument("--mtp", default="",
                    help="engine --mtp dir (empty: leave whatever setup wrote)")
    ap.add_argument("--chat-template", default="",
                    help="copy this jinja over the pack tokenizer chat_template.jinja")
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
    cfg["args"] = args
    cfg["gpu"] = [1, 0]
    cfg["host"] = "0.0.0.0"
    cfg.pop("api_key", None)
    cfg["fit_max_tokens"] = True
    cfg["layer_split"] = a.layer_split
    cfg["draft_vocab"] = cfg.get("draft_vocab") or "cjk"
    # Qwen3.8-Flash-Next thinking-mode card: temp=1.0 top_p=0.95 top_k=20 min_p=0
    # presence_penalty=0 repetition_penalty=1. force=true: client sampling ignored.
    # reasoning_effort is NOT set here — client passes xhigh/medium/low.
    cfg["sampling"] = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "force": True,
    }
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
    print("  gpus 1,0  layer-split", a.layer_split, "(GPU0 desktop is the later stage)")
    print("  host 0.0.0.0  (Tailscale / LAN)")
    print("  fit_max_tokens=true  (clamp max_tokens instead of 400)")
    print("  sampling temp=1.0 top_p=0.95 top_k=20 min_p=0 (force, client ignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
