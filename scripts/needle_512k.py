#!/usr/bin/env python3
"""512K NIAH 埋点: 一根 496K token 的 " x" 干草堆, 5 个深度(10/30/50/70/90%)各插一把针,
末尾一问收 5 个码. 一次请求, 打分=严格子串. 复用 decode_bench 的填充/请求原语.

用法: python3 needle_512k.py [--url http://127.0.0.1:8005] [--target 496000] [--name needle_512k]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
import uuid
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decode_bench as db  # noqa: E402

CALIB_TOKENS = 2048  # 与 decode_bench 校准同级别, 500K 下 " x" 分词线性, 误差 <2%
CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  # 去 O/0/I/1 防抄错
NAMES = ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO"]
DEPTHS = [0.10, 0.30, 0.50, 0.70, 0.90]


def make_code(rng: random.Random) -> str:
    return "".join(rng.choice(CODE_ALPHABET) for _ in range(4)) + "-" + "".join(rng.choice("0123456789") for _ in range(4))


def build_haystack(target_tokens: int, ratio: float, needles: list[tuple[str, str]]) -> str:
    rng = random.Random(0xC0FFEE)
    head = f"id={uuid.uuid4().hex}\nThis is padding. Ignore it. Answer the last user message only.\n"
    # 每根针句约 30 token; 问题约 120 token; 预留 2K 安全边距
    budget_units = int((target_tokens - 2000) * ratio)
    seg = budget_units // (len(needles) + 1)
    parts = [head]
    for i, (name, code) in enumerate(needles):
        parts.append(" x" * seg)
        pos = rng.randint(100, 999)
        parts.append(
            f"\n=== Important fact #{i + 1} (memorize it): the launch code of Project {name} "
            f"is {code}, recorded on day {pos} of the Kestrel log. ===\n"
        )
    parts.append(" x" * (budget_units - seg * len(needles)))
    return "".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8005")
    ap.add_argument("--target", type=int, default=496000)
    ap.add_argument("--tg", type=int, default=512)
    ap.add_argument("--name", default="needle_512k")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/ubuntu_handoff/results"))
    args = ap.parse_args()

    base = args.url.rstrip("/")
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=10) as r:
            engines = json.load(r).get("data") or []
        model = str(engines[0].get("root") or engines[0].get("id")) if engines else None
    except Exception:
        model = None
    extra = db.build_extra(True, "", strata=True)

    ratio = db.calibrate(base, model, extra)
    db.log(f"padding ratio {ratio:.3f} units/token")

    rng = random.Random(time.time_ns())
    needles = [(n, make_code(rng)) for n in NAMES]
    hay = build_haystack(args.target, ratio, needles)
    question = (
        "List the launch codes of the five projects in the order "
        + ", ".join(NAMES)
        + ". Reply only as "
        + "; ".join(f"{n}=<code>" for n in NAMES)
        + "."
    )
    messages = [{"role": "user", "content": hay + "\n\n" + question}]

    t0 = time.perf_counter()
    data, wall = db.chat_blocking(base, model, messages, args.tg, 900.0, extra)
    el = time.perf_counter() - t0

    msg = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    timing = data.get("timings") if isinstance(data.get("timings"), dict) else {}
    usage = data.get("usage") or {}

    scores = {}
    for name, code in needles:
        m = re.search(rf"{name}\s*[:=]\s*([A-Z0-9\-]+)", msg, re.I)
        got = m.group(1).upper() if m else ""
        scores[name] = {"want": code, "got": got, "ok": got == code,
                        "loose": code in msg.upper()}

    passed = sum(1 for s in scores.values() if s["ok"])
    row = {
        "when": datetime.now().isoformat(timespec="seconds"),
        "url": base,
        "engine": (data.get("timings") or {}).get("version") if isinstance(data.get("timings"), dict) else None,
        "target_tokens": args.target,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cache_n": timing.get("cache_n"),
        "predicted_n": timing.get("predicted_n"),
        "decode_tps": timing.get("predicted_per_second"),
        "draft_n": timing.get("draft_n"),
        "draft_n_accepted": timing.get("draft_n_accepted"),
        "http_wall_s": wall,
        "wall_s": el,
        "answer": msg,
        "needles": needles,
        "depths": DEPTHS,
        "scores": scores,
        "passed": f"{passed}/{len(NAMES)}",
    }
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, args.name + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=1)

    print(f"prompt_tokens={usage.get('prompt_tokens')} cache_n={timing.get('cache_n')} "
          f"decode={timing.get('predicted_per_second')} t/s wall={el:.1f}s -> {row['passed']}")
    for (name, code), depth in zip(needles, DEPTHS):
        s = scores[name]
        mark = "PASS" if s["ok"] else ("LOOSE" if s["loose"] else "FAIL")
        print(f"  depth {int(depth * 100):>3}%  {name:<8} want={code} got={s['got'] or '-'}  [{mark}]")
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
