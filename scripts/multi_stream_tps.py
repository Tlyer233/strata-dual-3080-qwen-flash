#!/usr/bin/env python3
"""多流（Strata parallel N）并发测速：每条流各自的 prefill / decode t/s。

每条流是一段独立会话（各自的 nonce），一轮里所有流同时开跑：
  run 1  各流冷 prefill 一个 depth 的 prompt  -> 并发下的 prefill 带宽
  run 2+ 同一段会话续问（前缀命中）        -> 并发下的 decode
--streams 1 就是单流基线，同一张表直接对比。

用法（README 口径）：
  python ~/ubuntu_handoff/scripts_3080/multi_stream_tps.py --url http://127.0.0.1:8005 \
      --streams 2 --depth 32k --tg 256 --runs 3 --name dual_4090_1
JSON 落在 ~/ubuntu_handoff/results/<name>.json，--show <name> 重打表。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import uuid
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decode_bench as db  # noqa: E402
import decode_ctx_tps as dc  # noqa: E402

log = db.log
QUESTIONS = (
    "Reply with a long plain paragraph about rivers.",
    "Reply with a long plain paragraph about lighthouses.",
    "Reply with a long plain paragraph about the history of cartography.",
    "Reply with a long plain paragraph about beekeeping.",
    "Reply with a long plain paragraph about harbor cranes.",
)


def parse_depth(raw: str) -> list[tuple[str, int]]:
    """--depth 32k 或 32k,128k（逗号 = 每条流各自的深度）。"""
    out: list[tuple[str, int]] = []
    for tok in str(raw or "").split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok.endswith("k") and tok[:-1].isdigit():
            out.append((tok, int(tok[:-1]) * 1024))
        elif tok.endswith("m") and tok[:-1].rstrip(".").isdigit():
            out.append((tok, int(float(tok[:-1]) * 1024 * 1024)))
        elif tok.isdigit():
            out.append((tok + "t", int(tok)))
        else:
            hit = [(n, v) for n, v in dc.TIERS if n.lower() == tok]
            if not hit:
                raise SystemExit(f"--depth 认不出 {tok!r}（可用 8k/32k/128k/256k/384k/512k 或纯 token 数）")
            out.append(hit[0])
    if not out:
        out = [("32k", 32 * 1024)]
    return out


def slots_info(base: str) -> dict:
    """/props 的 total_slots（= 引擎真开出来的并发流数）+ /v1/status 的停放计数。"""
    out: dict = {}
    try:
        props = db.get_json(base.rstrip("/") + "/props", timeout=10)
        for k in ("total_slots", "slots", "parallel"):
            if props.get(k) is not None:
                out[k] = props[k]
    except Exception:
        pass
    try:
        st = db.get_json(base.rstrip("/") + "/v1/status", timeout=10)
        act = st.get("activity") or {}
        for k in ("in_flight", "parked", "evictions"):
            if act.get(k) is not None:
                out[k] = act[k]
    except Exception:
        pass
    return out


def run_round(base, model, n_streams, depth, tg, ratio, extra, stagger, run_i, convs):
    """一轮：n_streams 条会话同时各发一问。convs 保存每条流的对话历史。"""
    rows: list[dict] = [None] * n_streams  # type: ignore
    barrier = threading.Barrier(n_streams + 1)

    def worker(i: int) -> None:
        if run_i == 1:
            nonce = uuid.uuid4().hex
            prompt = db.filler(int(depth * ratio), nonce) + "\n" + QUESTIONS[0]
            convs[i] = [{"role": "user", "content": prompt}]
        else:
            convs[i] = convs[i] + [{"role": "user", "content": QUESTIONS[(run_i - 1) % len(QUESTIONS)]}]
        barrier.wait()
        time.sleep(stagger * i)
        t_start = time.perf_counter()
        try:
            res = db.chat_stream(base, model, convs[i], tg, db.req_timeout_for(depth), extra, True)
        except Exception as exc:  # 一条流炸了别拖着别人
            res = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        res["stream"] = i + 1
        res["run"] = run_i
        res["start_rel_s"] = t_start
        if res.get("ok") and res.get("out_tokens"):
            # 续问：把回复留在历史里，下一轮前缀才接着命中
            convs[i] = convs[i] + [{"role": "assistant", "content": f"reply-{run_i}-s{i + 1}"}]
        rows[i] = res

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n_streams)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    barrier.wait()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    ok = [r for r in rows if r and r.get("ok")]
    dec = [r["decode_tps"] for r in ok if r.get("decode_tps")]
    pre = [r["prefill_tps"] for r in ok if r.get("prefill_tps")]
    agg = {
        "run": run_i,
        "streams": n_streams,
        "ok": len(ok),
        "round_wall_s": round(wall, 2),
        "decode_sum_tps": round(sum(dec), 1) if dec else None,
        "decode_min_tps": round(min(dec), 1) if dec else None,
        "decode_max_tps": round(max(dec), 1) if dec else None,
        "prefill_mean_tps": round(statistics.fmean(pre), 1) if pre else None,
        "prefill_min_tps": round(min(pre), 1) if pre else None,
    }
    for r in ok:
        log(
            f"    s{r['stream']} run {run_i}  in={r.get('prompt_tokens')} (cached {r.get('cached_tokens')}) "
            f"out={r.get('out_tokens')}  prefill={r.get('prefill_tps') or 0:.0f} t/s  "
            f"ttft={r.get('ttft_s') or 0:.2f}s  decode={r.get('decode_tps') or 0:.1f} t/s"
        )
    for r in rows:
        if r and not r.get("ok"):
            log(f"    s{r['stream']} run {run_i}  失败: {r.get('error')}")
    log(
        f"  == run {run_i}: {agg['ok']}/{n_streams} 路  decode 合计 {agg['decode_sum_tps']} t/s "
        f"(最慢 {agg['decode_min_tps']})  prefill 均值 {agg['prefill_mean_tps']} t/s  轮墙钟 {agg['round_wall_s']} s"
    )
    return rows, agg


def table(rows: list[dict], aggs: list[dict]) -> str:
    head = (
        f"{'str':>4} {'run':>4} {'prompt':>7} {'cached':>7} {'out':>4} "
        f"{'pre_s':>7} {'pre_tps':>8} {'ttft_s':>7} {'dec_tps':>8} {'e2e_tps':>8}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        if not r:
            continue
        if not r.get("ok"):
            lines.append(f"{r.get('stream', '?'):>4} {r.get('run', '?'):>4}  失败: {r.get('error')}")
            continue
        f = lambda k, d=1: (f"{r[k]:.{d}f}" if isinstance(r.get(k), (int, float)) else "-")  # noqa: E731
        lines.append(
            f"{r['stream']:>4} {r['run']:>4} {r.get('prompt_tokens') or 0:>7} {r.get('cached_tokens') or 0:>7} "
            f"{r.get('out_tokens') or 0:>4} {f('prefill_s'):>7} {f('prefill_tps', 0):>8} {f('ttft_s', 2):>7} "
            f"{f('decode_tps', 1):>8} {f('e2e_tps', 1):>8}"
        )
    lines.append("-" * len(head))
    for a in aggs:
        lines.append(
            f"run {a['run']}: {a['ok']}/{a['streams']} 路  decode 合计 {a['decode_sum_tps']} t/s  "
            f"最慢单路 {a['decode_min_tps']} t/s  prefill 均值 {a['prefill_mean_tps']} t/s  轮墙钟 {a['round_wall_s']} s"
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="多流并发 decode/prefill 测速")
    p.add_argument("--url", default="http://127.0.0.1:8005")
    p.add_argument("--streams", default="1,2", help="并发流数列表，如 1,2 或 2；1 = 单流基线")
    p.add_argument("--depth", default="32k", help="每条流的 prompt 深度：32k，或 32k,128k 分给各流")
    p.add_argument("--tg", type=int, default=256)
    p.add_argument("--runs", type=int, default=3, help="每条会话的轮数（run1 冷 prefill，之后 decode）")
    p.add_argument("--stagger", type=float, default=0.0, help="每条流错开启动的秒数（>0 可测 prefill 撞 decode）")
    p.add_argument("--model", default="")
    p.add_argument("--name", default="", help="results/ 下的 json 名")
    p.add_argument("--show", default="", help="重打已存 json 的表")
    p.add_argument("--think", action="store_true", help="允许 thinking（默认关）")
    p.add_argument("--extra-body", default="")
    p.add_argument("--keep-going", action="store_true", help="一条流失败也继续下一轮")
    args = p.parse_args()

    base = args.url.rstrip("/")
    if args.show:
        path = args.show if args.show.endswith(".json") else os.path.join(dc.RESULTS, args.show + ".json")
        saved = json.load(open(path, encoding="utf-8"))
        for blk in saved.get("blocks", []):
            log(f"streams={blk['streams']} depth={blk['depth_name']} tg={blk['tg']}  slots={blk.get('slots')}")
            log(table([r for b in blk["rounds"] for r in b["rows"]], [b["agg"] for b in blk["rounds"]]))
        return 0

    out_path = dc.ask_name(args.name)
    cap, src = dc.probe_cap(base)
    engines = dc.load_engines(base)
    model = dc.pick_engine(engines, args.model, None, sys.stdin.isatty())["id"]
    extra = db.build_extra(not args.think, args.extra_body, strata=db.endpoint_is_strata(base))
    ratio = db.calibrate(base, model, extra)
    depths = parse_depth(args.depth)
    plans = [int(s.strip()) for s in str(args.streams).split(",") if s.strip()]
    log(f"engine={model}  n_ctx={cap} ({src})  tg={args.tg}  runs={args.runs}  depth={depths}")

    blocks: list[dict] = []
    for n_streams in plans:
        depth_name, depth = depths[0] if len(depths) == 1 else depths[min(n_streams, len(depths)) - 1]
        budget = cap - args.tg - dc.SLACK
        if depth > budget:
            log(f"skip streams={n_streams} depth={depth}（n_ctx {cap} 装不下）")
            continue
        dc.wait_idle(base)
        log(f"--- streams={n_streams}  depth={depth_name}  slots={slots_info(base)} ---")
        convs: list[list[dict]] = [[] for _ in range(n_streams)]
        rounds = []
        for run_i in range(1, args.runs + 1):
            rows, agg = run_round(base, model, n_streams, depth, args.tg, ratio, extra, args.stagger, run_i, convs)
            rounds.append({"rows": rows, "agg": agg})
            if agg["ok"] < n_streams and not args.keep_going:
                log("有流失败，停（--keep-going 可继续）")
                break
        blocks.append({
            "streams": n_streams,
            "depth_name": depth_name,
            "depth": depth,
            "tg": args.tg,
            "slots": slots_info(base),
            "rounds": rounds,
        })
        log(table([r for b in [blocks[-1]] for rr in b["rounds"] for r in rr["rows"]],
                  [rr["agg"] for rr in blocks[-1]["rounds"]]))

    out = {
        "name": os.path.basename(out_path)[:-5],
        "created": datetime.now().isoformat(timespec="seconds"),
        "url": base,
        "model": model,
        "n_ctx": cap,
        "tg": args.tg,
        "runs": args.runs,
        "stagger": args.stagger,
        "engine_version": None,
        "blocks": blocks,
    }
    try:
        props = db.get_json(base + "/props", timeout=10)
        out["engine_version"] = props.get("version")
    except Exception:
        pass
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log(f"json: {out_path}")
    # 各 streams 的 decode 合计对比（run>=2 的中位数）
    for b in blocks:
        steady = [rr["agg"]["decode_sum_tps"] for rr in b["rounds"][1:] if rr["agg"]["decode_sum_tps"]]
        cold = [rr["agg"]["prefill_mean_tps"] for rr in b["rounds"][:1] if rr["agg"]["prefill_mean_tps"]]
        log(f"streams={b['streams']} depth={b['depth_name']}: decode 合计(稳态中位, run>=2) "
            f"{statistics.median(steady) if steady else '-'} t/s   冷 prefill 均值(run1) "
            f"{statistics.median(cold) if cold else '-'} t/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
