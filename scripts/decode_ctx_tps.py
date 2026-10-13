#!/usr/bin/env python3
"""Decode t/s sweep over context depth on an OpenAI /v1 endpoint.

Tiers: 8k 32k 64k 128k 200k 256k, skipping what n_ctx cannot fit.
Measurement lives in decode_bench.py. Asks for a result name, writes JSON
under ~/ubuntu_handoff/results.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.request
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decode_bench as db  # noqa: E402

TIERS = (
    ("8k", 8 * 1024),
    ("32k", 32 * 1024),
    ("64k", 64 * 1024),
    ("128k", 128 * 1024),
    ("200k", 200 * 1024),
    ("256k", 256 * 1024),
)
TG = 256
SLACK = 256
RESULTS = os.path.expanduser("~/ubuntu_handoff/results")
SAFE_NAME = re.compile(r"^[\w.\-]+$")

log = db.log


def _ints(*vals) -> list[int]:
    out = []
    for v in vals:
        try:
            n = int(v)
        except (TypeError, ValueError):
            continue
        if n > 0:
            out.append(n)
    return out


def load_engines(base: str) -> list[dict]:
    """Unique served names from GET /v1/models. Aliases of the same slot collapse."""
    data = db.get_json(base.rstrip("/") + "/v1/models")
    engines: list[dict] = []
    by_id: dict[str, dict] = {}
    for m in data.get("data") or []:
        if not isinstance(m, dict):
            continue
        served = str(m.get("root") or m.get("id") or "")
        if not served:
            continue
        aid = str(m.get("id") or served)
        meta = m.get("meta") if isinstance(m.get("meta"), dict) else {}
        n = _ints(
            m.get("max_model_len"),
            m.get("context_length"),
            meta.get("n_ctx"),
            m.get("n_ctx"),
        )
        if served not in by_id:
            row = {
                "id": served,
                "slot": str(m.get("slot") or served),
                "owned_by": str(m.get("owned_by") or ""),
                "n_ctx": min(n) if n else 0,
                "aliases": [],
            }
            by_id[served] = row
            engines.append(row)
        if aid != served and aid not in by_id[served]["aliases"]:
            by_id[served]["aliases"].append(aid)
        if n and (not by_id[served]["n_ctx"] or min(n) < by_id[served]["n_ctx"]):
            by_id[served]["n_ctx"] = min(n)
    return engines


def format_engines(engines: list[dict]) -> str:
    lines = ["engines:"]
    for i, e in enumerate(engines, 1):
        extra = f"  ({', '.join(e['aliases'])})" if e["aliases"] else ""
        lines.append(f"  {i}  {e['id']:<36} n_ctx={e['n_ctx']}{extra}")
    return "\n".join(lines)


def pick_engine(
    engines: list[dict], model: str, engine_n: int | None, tty: bool
) -> dict:
    if not engines:
        raise SystemExit("GET /v1/models returned no models")
    if model:
        q = model.strip().lower()
        for e in engines:
            names = [e["id"], e["slot"], *e["aliases"]]
            if q in {n.lower() for n in names}:
                return e
        raise SystemExit(f"unknown model {model!r}\n{format_engines(engines)}")
    if engine_n is not None:
        if engine_n < 1 or engine_n > len(engines):
            raise SystemExit(f"--engine {engine_n} out of range\n{format_engines(engines)}")
        return engines[engine_n - 1]
    if len(engines) == 1:
        return engines[0]
    menu = format_engines(engines)
    if not tty:
        raise SystemExit(menu + "\npass --engine N or --model NAME")
    print(menu)
    raw = input(f"which engine? [1-{len(engines)}]: ").strip()
    if not raw.isdigit():
        raise SystemExit(f"need a number 1-{len(engines)}")
    return pick_engine(engines, "", int(raw), False)


def probe_cap(base: str, n_ctx: int = 0) -> tuple[int, str]:
    """n_ctx of the chosen engine; fall back to /props /server_info."""
    if n_ctx > 0:
        return n_ctx, "openai"
    base = base.rstrip("/")
    errors: list[str] = []

    try:
        engines = load_engines(base)
        if len(engines) == 1 and engines[0]["n_ctx"] > 0:
            return engines[0]["n_ctx"], "openai"
        errors.append("/v1/models has multiple engines; pick --engine N")
    except Exception as exc:
        errors.append(f"/v1/models {type(exc).__name__}: {exc}")

    try:
        props = db.get_json(base + "/props")
        dgs = props.get("default_generation_settings") or {}
        n = _ints(props.get("n_ctx"), dgs.get("n_ctx"))
        if n:
            return min(n), "llamacpp"
    except Exception as exc:
        errors.append(f"/props {type(exc).__name__}: {exc}")

    try:
        info = db.get_json(base + "/server_info", timeout=20)
        args = info.get("server_args") if isinstance(info.get("server_args"), dict) else {}
        n = _ints(
            info.get("context_length"),
            info.get("max_model_len"),
            args.get("context_length"),
        )
        if n:
            return min(n), "sglang"
    except Exception as exc:
        errors.append(f"/server_info {type(exc).__name__}: {exc}")

    try:
        st = db.get_json(base + "/v1/status", timeout=20)
        if isinstance(st, dict) and st.get("service") == "strata":
            n = _ints(
                (st.get("context") or {}).get("native"),
                st.get("cache_max_tokens"),
            )
            if n:
                return min(n), "strata"
        errors.append("/v1/status has no n_ctx")
    except Exception as exc:
        errors.append(f"/v1/status {type(exc).__name__}: {exc}")

    raise SystemExit(f"cannot read n_ctx from {base}: {'; '.join(errors) or 'no n_ctx field'}")


def flush_cache(base: str) -> None:
    """Empty the radix pool so every tier starts from the same state.
    Pass a timeout, else SGLang refuses (400) unless the engine is already idle."""
    url = base.rstrip("/") + "/flush_cache?timeout=30"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            first = r.read().decode("utf-8", "replace").strip().splitlines()[:1]
            log(f"    flush_cache {r.status} {first[0] if first else ''}")
    except Exception as exc:
        log(f"    flush_cache failed: {exc}")


def wait_idle(base: str, timeout: float = 180.0) -> None:
    """Strata serves one request at a time; a leftover client would inflate prefill wall time."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        try:
            st = db.get_json(base.rstrip("/") + "/v1/status", timeout=10)
        except Exception:
            return
        inf = int((st.get("activity") or {}).get("in_flight") or 0)
        if inf <= 0:
            return
        log(f"waiting for engine idle (in_flight={inf})")
        time.sleep(1.0)
    log("engine still busy; measurements may include queue time")


def pick_depths(cap: int, tg: int) -> list[tuple[str, int]]:
    """Top tier may be clipped a little to leave room for tg; more than 10% off is a skip."""
    budget = cap - tg - SLACK
    out: list[tuple[str, int]] = []
    for name, want in TIERS:
        depth = min(want, budget)
        if depth < 512 or depth < want * 0.9:
            log(f"skip {name} (needs {want}+{tg}, n_ctx {cap})")
            continue
        out.append((name, depth))
    return out


def ask_name(cli_name: str, out_dir: str = "") -> str:
    raw = cli_name.strip()
    dest = os.path.abspath(os.path.expanduser(out_dir or RESULTS))
    if not raw:
        raw = input(f"result json name (saved under {dest}): ").strip()
    if not raw:
        raw = datetime.now().strftime("decode_ctx_%Y%m%d_%H%M%S")
    raw = os.path.basename(raw)
    if raw.endswith(".json"):
        raw = raw[:-5]
    if not SAFE_NAME.match(raw):
        raise SystemExit(f"bad name {raw!r}; use letters/digits/._-")
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, raw + ".json")
    if os.path.exists(path):
        raise SystemExit(f"already exists: {path}")
    return path


def parse_tiers(raw: str, depths: list[tuple[str, int]]) -> list[tuple[str, int]]:
    if not raw or raw.strip().lower() in ("all", "*"):
        return depths
    tokens = [p.strip().lower() for p in raw.replace(",", " ").split() if p.strip()]
    alias = {n: n for n, _ in depths}
    for n, d in depths:
        alias[str(d)] = n
        alias[n.rstrip("k")] = n  # "128" -> "128k"
    want: list[str] = []
    unknown: list[str] = []
    for t in tokens:
        if t in ("all", "*"):
            return depths
        got = alias.get(t)
        if got:
            want.append(got)
        else:
            unknown.append(t)
    out = [t for t in depths if t[0] in want]
    if unknown or not out:
        have = " ".join(n for n, _ in depths)
        raise SystemExit(f"no tier matches {tokens}; have {have} (or all)")
    return out


def ask_tiers(cli: str, depths: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """CLI --tiers wins; otherwise TTY pick one/all so a full sweep is not the default surprise."""
    names = " ".join(n for n, _ in depths)
    if cli.strip():
        return parse_tiers(cli, depths)
    if not sys.stdin.isatty():
        log(f"no TTY; running all tiers that fit: {names}")
        return depths
    print(f"context tiers that fit: {names}")
    raw = input("which context?  one (8k/32k/64k/128k/...) or all  [all]: ").strip()
    return parse_tiers(raw, depths)


ABBREV = {
    "coding": "code",
    "humanities": "human",
    "math": "math",
    "multilingual": "multi",
    "qa": "qa",
    "rag": "rag",
    "reasoning": "reason",
    "roleplay": "role",
    "stem": "stem",
    "summarization": "summ",
    "writing": "write",
}
QUESTION_CHARS = 2000
# The shipped parquet is an index: rows not fetched by SPECDEC_BENCH carry this
# text instead of the question, and some carry an unfilled {placeholder}.
STUB = "FULL BENCHMARK DATA SHOULD BE FETCHED"
UNFILLED = re.compile(r"\{[A-Za-z_][A-Za-z0-9_ ]*\}")


def usable_prompt(text: str) -> bool:
    return bool(text.strip()) and STUB not in text and not UNFILLED.search(text)


def load_scenarios(cats: str) -> list[tuple[str, str]]:
    """First real first-turn prompt per category, truncated so it does not skew depth."""
    import tps_test as tps

    samples = tps.load_samples(tps.require_local_parquet(), cats or "all", None)
    picked: dict[str, str] = {}
    seen: set[str] = set()
    for s in samples:
        seen.add(s.category)
        if s.category in picked or not usable_prompt(s.turns[0]):
            continue
        picked[s.category] = s.turns[0][:QUESTION_CHARS]
    dropped = sorted(seen - set(picked))
    if dropped:
        log(f"no real prompt in parquet, skipped: {' '.join(dropped)}")
    return [(c, picked[c]) for c in sorted(picked)]


def _cell(rows: list[dict], tier_row: dict, cat: str) -> float | None:
    got = (tier_row.get("scenarios") or {}).get(cat) or {}
    # single-shot result files only have decode_tps
    return got.get("median_decode_tps") or got.get("decode_tps")


def _cv(tier_row: dict, cat: str) -> float | None:
    got = (tier_row.get("scenarios") or {}).get(cat) or {}
    return got.get("cv_pct")


def cv_table(rows: list[dict], cats: list[str]) -> str:
    """Spread of the repeats: stdev / mean per cell, in percent."""
    cols = [ABBREV.get(c, c[:6]) for c in cats]
    head = f"{'CV%':>6}  " + "  ".join(f"{c:>6}" for c in cols) + f"  {'n':>6}"
    lines = [head, "-" * len(head)]
    for r in rows:
        cells = []
        for c in cats:
            v = _cv(r, c)
            cells.append(f"{v:>6.1f}" if v is not None else f"{'-':>6}")
        got = (r.get("scenarios") or {}).get(cats[0]) if cats else None
        n = (got or {}).get("n") or 0
        lines.append(f"{r['tier']:>6}  " + "  ".join(cells) + f"  {n:>6}")
    return "\n".join(lines)


def scenario_table(rows: list[dict], cats: list[str]) -> str:
    """Rows are tiers, columns are scenarios; avg column per tier, avg row per scenario."""
    cols = [ABBREV.get(c, c[:6]) for c in cats]
    head = f"{'tier':>6}  " + "  ".join(f"{c:>6}" for c in cols)
    head += f"  {'AVG':>6}  |  {'pre_s':>8}  {'pre_min':>7}  {'pre_tps':>8}"
    lines = [head, "-" * len(head)]
    for r in rows:
        vals = [_cell(rows, r, c) for c in cats]
        cells = [f"{v:>6.1f}" if v else f"{'-':>6}" for v in vals]
        ok = [v for v in vals if v]
        cells.append(f"{sum(ok) / len(ok):>6.1f}" if ok else f"{'-':>6}")
        pre = r.get("prefill") or {}
        wall = pre.get("wall_s")
        tail = (
            f"  |  {wall:>8.1f}  {wall / 60:>7.2f}  {pre.get('prefill_tps') or 0:>8.0f}"
            if wall
            else f"  |  {'-':>8}  {'-':>7}  {'-':>8}"
        )
        lines.append(f"{r['tier']:>6}  " + "  ".join(cells) + tail)

    foot = []
    for c in cats:
        ok = [v for v in (_cell(rows, r, c) for r in rows) if v]
        foot.append(f"{sum(ok) / len(ok):>6.1f}" if ok else f"{'-':>6}")
    every = [v for r in rows for v in (_cell(rows, r, c) for c in cats) if v]
    foot.append(f"{sum(every) / len(every):>6.1f}" if every else f"{'-':>6}")
    lines.append("-" * len(head))
    lines.append(f"{'AVG':>6}  " + "  ".join(foot))
    return "\n".join(lines)


def table(rows: list[dict]) -> str:
    """decode is the steady run; ttft/decode_cold come from run 1 (cold prefill)."""
    head = (
        f"{'tier':>6}  {'in':>8}  {'out':>5}  {'ttft_cold':>10}  "
        f"{'decode':>8}  {'dec_cold':>9}  {'best':>7}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        st = r.get("steady") or r.get("best")  # older result files only have best
        if not st:
            lines.append(f"{r['tier']:>6}  {'FAIL':>8}  {r.get('error', '')[:48]}")
            continue
        cold = r.get("cold") or st
        best = (r.get("best") or st)["decode_tps"]
        lines.append(
            f"{r['tier']:>6}  {st['prompt_tokens']:>8}  {st['out_tokens']:>5}  "
            f"{cold['ttft_s']:>10.2f}  {st['decode_tps']:>8.1f}  "
            f"{cold['decode_tps'] or 0:>9.1f}  {best:>7.1f}"
        )
    return "\n".join(lines)


def run_synthetic_tier(base, model, depth, ratio, extra, args, row: dict) -> None:
    runs = db.measure_depth(
        base, model, depth, args.tg, ratio, extra, args.exact_tg, args.runs
    )
    row["runs"] = runs
    ok = [r for r in runs if r.get("decode_tps")]
    # run 1 pays the cold prefill, so its decode sits behind a hot GPU
    row["cold"] = runs[0] if runs else None
    row["steady"] = ok[-1] if ok else None
    row["best"] = max(ok, key=lambda r: r["decode_tps"]) if ok else None
    if not ok:
        row["error"] = "no tokens generated"


def run_scenario_tier(base, model, depth, ratio, extra, args, scenarios, row: dict) -> None:
    """One prefill per tier, then every scenario branches off that cached prefix.
    Rounds are interleaved (all scenarios per round) so thermal drift hits them equally."""
    prefix, pre = db.fill_prefix(base, model, depth, ratio, extra)
    # First streamed continuation after a blocking prefill often reports
    # cached_tokens=0 (radix not yet visible to the next request). Burn it.
    db.measure_on_prefix(base, model, prefix, "ok", 8, depth, extra, False)
    row["prefill"] = pre
    elapsed = pre.get("prefill_s") if pre.get("prefill_s") is not None else pre["wall_s"]
    src = "engine" if pre.get("timings") else "http"
    note = ""
    http_wall = pre.get("http_wall_s")
    if http_wall and pre.get("prefill_s") and http_wall > pre["prefill_s"] * 1.25:
        note = f"  http_wall={http_wall:.1f}s"
    log(
        f"    prefill {pre['prompt_tokens']} tok  {elapsed:.1f}s "
        f"({elapsed / 60:.2f} min)  {pre['prefill_tps'] or 0:.0f} t/s  ({src}){note}"
    )

    shots: dict[str, list[dict]] = {cat: [] for cat, _ in scenarios}
    for rnd in range(1, args.runs + 1):
        for cat, question in scenarios:
            res = db.measure_on_prefix(
                base, model, prefix, question, args.tg, depth, extra, args.exact_tg
            )
            shots[cat].append(res)
            warn = ""
            # Hybrid mamba only reports hits on checkpoint boundaries (chunked
            # prefill, often 4096). cached=4096 on an 8k prefix is a hit, not a
            # miss. True miss is cached_tokens==0 (full re-prefill).
            cached = int(res.get("cached_tokens") or 0)
            prompt_n = int(pre.get("prompt_tokens") or 0)
            if cached <= 0:
                warn = "  PREFIX LOST (re-prefilled)"
            elif prompt_n and cached < prompt_n * 0.8:
                warn = f"  prefix_ckpt={cached}/{prompt_n}"
            mtp = ""
            if res.get("draft_n"):
                mtp = f"  mtp {res.get('draft_n_accepted') or 0}/{res['draft_n']}"
            log(
                f"    r{rnd} {cat:14s} in={res['prompt_tokens']} "
                f"cached={res['cached_tokens']} out={res['out_tokens']}  "
                f"ttft={(res.get('ttft_s') or 0):.2f}s  "
                f"decode={res['decode_tps'] or 0:.1f} t/s{mtp}{warn}"
            )

    got: dict = {}
    for cat, _ in scenarios:
        vals = [r["decode_tps"] for r in shots[cat] if r.get("decode_tps")]
        entry: dict = {"runs": shots[cat], "n": len(vals)}
        if vals:
            mean = statistics.fmean(vals)
            entry.update(
                median_decode_tps=statistics.median(vals),
                mean_decode_tps=mean,
                min_decode_tps=min(vals),
                max_decode_tps=max(vals),
                cv_pct=(statistics.stdev(vals) / mean * 100) if len(vals) > 1 else 0.0,
            )
        got[cat] = entry

    row["scenarios"] = got
    ok = [e for e in got.values() if e.get("median_decode_tps")]
    row["steady"] = (ok[-1]["runs"][-1]) if ok else None
    if not ok:
        row["error"] = "no tokens generated"
        return

    if args.runs > 1:
        for cat, _ in scenarios:
            e = got[cat]
            if not e.get("median_decode_tps"):
                continue
            log(
                f"    {cat:14s} n={e['n']} med={e['median_decode_tps']:.1f} "
                f"mean={e['mean_decode_tps']:.1f} "
                f"min/max={e['min_decode_tps']:.1f}/{e['max_decode_tps']:.1f} "
                f"cv={e['cv_pct']:.1f}%"
            )
    avg = statistics.fmean(e["median_decode_tps"] for e in ok)
    row["avg_decode_tps"] = avg
    log(f"    avg of medians over {len(ok)} scenarios: {avg:.1f} t/s")


def main() -> int:
    p = argparse.ArgumentParser(description="decode t/s vs context depth")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument(
        "--n-ctx",
        type=int,
        default=0,
        help="override context cap (default: max_model_len from /v1/models)",
    )
    p.add_argument("--tg", type=int, default=TG)
    p.add_argument(
        "--runs",
        type=int,
        default=None,
        help="repeats: 2 synthetic (cold+steady), 3 per scenario (median)",
    )
    p.add_argument(
        "--tiers",
        "--ctx",
        dest="tiers",
        default="",
        help="which depths: 8k, 128k, '8k 128k', or all. "
        "If omitted and stdin is a TTY, prompt. Non-TTY default is all that fit.",
    )
    p.add_argument("--model", default="", help="served name or alias; else numbered pick")
    p.add_argument(
        "--engine",
        type=int,
        default=None,
        metavar="N",
        help="1-based engine from GET /v1/models (same menu as the TTY prompt)",
    )
    p.add_argument("--name", default="", help="skip prompt; json stem under results/")
    p.add_argument(
        "--out-dir",
        default="",
        help="directory for the result json (default ~/ubuntu_handoff/results)",
    )
    p.add_argument("--think", action="store_true", help="allow thinking (default off)")
    p.add_argument(
        "--no-exact-tg",
        dest="exact_tg",
        action="store_false",
        help="let the model stop early instead of min_tokens+ignore_eos",
    )
    p.add_argument("--extra-body", default="")
    p.add_argument(
        "--no-flush",
        dest="flush",
        action="store_false",
        help="keep whatever is in the radix pool instead of flushing per tier",
    )
    p.add_argument("--keep-going", action="store_true", help="do not stop on first failure")
    p.add_argument(
        "--scenarios",
        action="store_true",
        help="SPEED-Bench prompts (one per category) on top of each tier prefix",
    )
    p.add_argument("--cats", default="", help="scenario categories, e.g. coding,rag")
    p.add_argument("--show", default="", help="reprint the table from a saved result json")
    args = p.parse_args()

    if args.show:
        path = os.path.expanduser(args.show)
        if not os.path.isfile(path):
            path = os.path.join(RESULTS, os.path.basename(path))
            if not path.endswith(".json"):
                path += ".json"
        if not os.path.isfile(path):
            raise SystemExit(f"no such result: {path}")
        with open(path) as f:
            saved = json.load(f)
        cats = saved.get("scenarios") or []
        log(f"{path}  engine {saved.get('engine')}  n_ctx {saved.get('n_ctx')}")
        if cats:
            log(scenario_table(saved["tiers"], cats))
            if any(_cv(r, c) for r in saved["tiers"] for c in cats):
                log()
                log(cv_table(saved["tiers"], cats))
        else:
            log(table(saved["tiers"]))
        return 0

    base = args.url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]

    engines = load_engines(base)
    chosen = pick_engine(
        engines, args.model, args.engine, sys.stdin.isatty()
    )
    # vLLM lists root=weight path and id=served-model-name. POST /v1/chat
    # must use a valid alias (qwen3.8-27b), not the filesystem root.
    model = args.model.strip() or (chosen.get("aliases") or [chosen["id"]])[0]
    ninfer = chosen.get("owned_by") == "ninfer"
    cap, kind = probe_cap(base, args.n_ctx or chosen["n_ctx"])
    if ninfer:
        kind = "ninfer"
    elif db.endpoint_is_strata(base):
        kind = "strata"
        if not cap:
            cap = args.n_ctx or chosen["n_ctx"]
    extra = db.build_extra(
        not args.think, args.extra_body, ninfer=ninfer, strata=(kind == "strata")
    )
    fitted = pick_depths(cap, args.tg)
    if not fitted:
        raise SystemExit(f"no tier fits n_ctx={cap} tg={args.tg}")
    depths = ask_tiers(args.tiers, fitted)
    out_path = ask_name(args.name, args.out_dir)

    scenarios = load_scenarios(args.cats) if args.scenarios else []
    if args.runs is None:
        args.runs = 3 if scenarios else 2

    log(f"endpoint {base}")
    log(f"engine   {kind}  n_ctx={cap}")
    log(f"model    {model or '(auto)'}")
    log(
        f"tiers    {' '.join(n for n, _ in depths)}  tg={args.tg} runs={args.runs}"
        f"  exact_tg={args.exact_tg} flush={args.flush}"
    )
    log(f"json     {out_path}")
    if scenarios:
        log(f"scenarios {len(scenarios)}: {' '.join(c for c, _ in scenarios)}")
    if ninfer and args.flush:
        log("flush    skipped (ninfer has no /flush_cache)")
    if kind == "strata":
        wait_idle(base)
        if args.flush:
            log("flush    skipped (strata has no /flush_cache)")
    log()

    ratio = db.calibrate(base, model, extra)
    log(f"padding ratio {ratio:.3f} units/token")
    log()

    rows: list[dict] = []
    for name, depth in depths:
        log(f"  {name} (depth {depth})")
        row: dict = {"tier": name, "depth": depth}
        if args.flush and kind not in ("llamacpp", "ninfer", "strata"):
            flush_cache(base)
        try:
            if scenarios:
                run_scenario_tier(base, model, depth, ratio, extra, args, scenarios, row)
            else:
                run_synthetic_tier(base, model, depth, ratio, extra, args, row)
        except Exception as exc:
            row["steady"] = None
            row["error"] = str(exc)
            log(f"    FAIL {exc}")
        rows.append(row)
        if not row.get("steady") and not args.keep_going:
            log("  stopping (use --keep-going to continue)")
            break

    log()
    if scenarios:
        cats = [c for c, _ in scenarios]
        log(scenario_table(rows, cats))
        if args.runs > 1:
            log()
            log(cv_table(rows, cats))
    else:
        log(table(rows))

    payload = {
        "when": datetime.now().isoformat(timespec="seconds"),
        "url": base,
        "engine": kind,
        "n_ctx": cap,
        "model": model,
        "tg": args.tg,
        "runs": args.runs,
        "exact_tg": args.exact_tg,
        "flush_per_tier": args.flush,
        "extra_body": extra,
        "padding_ratio": ratio,
        "scenarios": [c for c, _ in scenarios],
        "tiers": rows,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    log(f"saved {out_path}")
    return 0 if any(r.get("steady") for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
