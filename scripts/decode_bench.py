#!/usr/bin/env python3
"""One decode measurement against an OpenAI /v1 endpoint (SGLang, llama.cpp, Strata).

Streams the completion and times the tokens, so prefill (TTFT) and decode
are reported separately. Strata/llama.cpp responses include `timings`
(prompt_per_second / predicted_per_second); those win over client wall
clock so a queued request is not counted as slow prefill. Only stdlib.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

CALIB_TOKENS = 2048
# SGLang's Swift template accepts low/medium/xhigh and raises on none.
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"}}
# NInfer rejects reasoning_effort together with enable_thinking=false.
NINFER_NO_THINK = {"enable_thinking": False}
# Strata maps enable_thinking=false to thinking off; extra reasoning_effort is ignored.


def log(msg: str = "") -> None:
    print(msg, flush=True)


def get_json(url: str, timeout: float = 10):
    req = urllib.request.Request(url.rstrip("/"), method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def endpoint_is_ninfer(base: str) -> bool:
    try:
        data = get_json(base.rstrip("/") + "/v1/models")
    except Exception:
        return False
    rows = data.get("data") or []
    return any(isinstance(m, dict) and m.get("owned_by") == "ninfer" for m in rows)


def endpoint_is_strata(base: str) -> bool:
    try:
        data = get_json(base.rstrip("/") + "/v1/status")
    except Exception:
        return False
    return isinstance(data, dict) and data.get("service") == "strata"


def overlay_timings(row: dict, timings: dict | None) -> dict:
    """Prefer the engine's llama.cpp-style timings over HTTP wall clock."""
    if not isinstance(timings, dict) or not timings:
        return row
    row["timings"] = timings
    pn = timings.get("prompt_n")
    pms = timings.get("prompt_ms")
    pps = timings.get("prompt_per_second")
    if isinstance(pps, (int, float)) and pps > 0:
        row["prefill_tps"] = float(pps)
    elif isinstance(pn, (int, float)) and isinstance(pms, (int, float)) and pn > 0 and pms > 0:
        row["prefill_tps"] = float(pn) / (float(pms) / 1000.0)
    if isinstance(pms, (int, float)):
        row["prefill_s"] = float(pms) / 1000.0
    if isinstance(pn, int):
        row["prompt_n_uncached"] = pn
    dps = timings.get("predicted_per_second")
    if isinstance(dps, (int, float)) and dps > 0:
        row["decode_tps"] = float(dps)
    if isinstance(timings.get("predicted_ms"), (int, float)):
        row["decode_s"] = float(timings["predicted_ms"]) / 1000.0
    if timings.get("cache_n") is not None:
        row["cached_tokens"] = int(timings["cache_n"])
    if timings.get("draft_n") is not None:
        row["draft_n"] = int(timings["draft_n"])
        row["draft_n_accepted"] = int(timings.get("draft_n_accepted") or 0)
    return row


def probe_model(base: str) -> str | None:
    """Single-id endpoints only. A multi-slot catalog must be picked explicitly."""
    try:
        data = get_json(base.rstrip("/") + "/v1/models")
        rows = data.get("data") or []
        roots: list[str] = []
        for m in rows:
            if not isinstance(m, dict):
                continue
            root = m.get("root") or m.get("id")
            if root and root not in roots:
                roots.append(str(root))
        return roots[0] if len(roots) == 1 else None
    except Exception:
        return None


def req_timeout_for(depth: int) -> float:
    # cold prefill of a deep prompt can take minutes
    return max(300.0, depth / 20.0 + 180.0)


def _post(url: str, body: dict, timeout: float, stream: bool):
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json", "Authorization": "Bearer sk-local"}
    if stream:
        headers["Accept"] = "text/event-stream"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace").strip()
        raise RuntimeError(f"HTTP {exc.code} {url}: {detail}") from None


def _body(
    model: str | None,
    messages: list[dict],
    max_tokens: int,
    stream: bool,
    extra: dict,
    exact: bool,
) -> dict:
    body: dict = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    if exact:
        body["min_tokens"] = max_tokens
        body["ignore_eos"] = True
    if model:
        body["model"] = model
    body.update(extra)
    return body


def chat_blocking(
    base: str,
    model: str | None,
    messages: list[dict],
    max_tokens: int,
    timeout: float,
    extra: dict,
) -> tuple[dict, float]:
    url = base.rstrip("/") + "/v1/chat/completions"
    body = _body(model, messages, max_tokens, False, extra, False)
    t0 = time.perf_counter()
    with _post(url, body, timeout, False) as r:
        data = json.loads(r.read().decode())
    return data, time.perf_counter() - t0


def chat_stream(
    base: str,
    model: str | None,
    messages: list[dict],
    max_tokens: int,
    timeout: float,
    extra: dict,
    exact: bool,
) -> dict:
    """Timings of a streamed completion. One SSE chunk is normally one token."""
    url = base.rstrip("/") + "/v1/chat/completions"
    body = _body(model, messages, max_tokens, True, extra, exact)

    stamps: list[float] = []
    usage: dict = {}
    timings: dict = {}
    finish: str | None = None
    chars = 0
    t0 = time.perf_counter()
    with _post(url, body, timeout, True) as r:
        # HTTPResponse.__iter__ yields 8KiB blocks, not lines. Fast DFlash
        # dumps many SSE events into one block; json.loads then fails and
        # decode_tps becomes 0 even though usage.completion_tokens is 256.
        while True:
            raw = r.readline()
            if not raw:
                break
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(obj.get("usage"), dict):
                usage = obj["usage"]
            if isinstance(obj.get("timings"), dict):
                timings = obj["timings"]
            for ch in obj.get("choices") or []:
                delta = ch.get("delta") or {}
                # thinking tokens are not decode; Strata puts them in reasoning_content
                piece = delta.get("content") or ""
                if piece:
                    stamps.append(time.perf_counter())
                    chars += len(piece)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    wall = time.perf_counter() - t0

    out_tokens = int(usage.get("completion_tokens") or 0) or len(stamps)
    span = (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
    row = {
        "ok": bool(stamps) or bool(out_tokens),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "cached_tokens": int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
        "out_tokens": out_tokens,
        "chunks": len(stamps),
        "chars": chars,
        "wall_s": wall,
        "ttft_s": (stamps[0] - t0) if stamps else None,
        "decode_span_s": span,
        "decode_tps": ((out_tokens - 1) / span) if span > 0 and out_tokens > 1 else None,
        "e2e_tps": (out_tokens / wall) if wall > 0 and out_tokens else None,
        "finish_reason": finish,
    }
    return overlay_timings(row, timings)


def filler(n_units: int, nonce: str) -> str:
    head = f"id={nonce}\nThis is padding. Ignore it. Answer the last user message only.\n"
    return head + (" x" * max(n_units, 1))


def calibrate(base: str, model: str | None, extra: dict) -> float:
    """Units of ' x' per prompt token. Cheap: one shallow request."""
    text = filler(CALIB_TOKENS, uuid.uuid4().hex)
    data, _ = chat_blocking(
        base, model, [{"role": "user", "content": text}], 1, 120.0, extra
    )
    pt = int(((data.get("usage") or {}).get("prompt_tokens")) or 0)
    if pt < 100:
        return 1.0
    ratio = CALIB_TOKENS / pt
    return min(max(ratio, 0.5), 2.0)


def measure_depth(
    base: str,
    model: str | None,
    depth: int,
    tg: int,
    ratio: float,
    extra: dict,
    exact: bool,
    runs: int = 1,
    nonce: str | None = None,
) -> list[dict]:
    """Prompt padded to ~depth tokens, then tg tokens generated. Run 1 pays prefill."""
    nonce = nonce or uuid.uuid4().hex
    prompt = filler(int(depth * ratio), nonce) + "\nReply with a long plain paragraph about rivers."
    messages = [{"role": "user", "content": prompt}]
    timeout = req_timeout_for(depth)
    out = []
    for i in range(1, runs + 1):
        res = chat_stream(base, model, messages, tg, timeout, extra, exact)
        res["run"] = i
        res["depth_target"] = depth
        out.append(res)
        log(
            f"    run {i}/{runs}  in={res['prompt_tokens']} (cached {res['cached_tokens']}) "
            f"out={res['out_tokens']}  ttft={res['ttft_s']:.2f}s  "
            f"decode={res['decode_tps'] or 0:.1f} t/s  wall={res['wall_s']:.1f}s"
        )
    return out


def fill_prefix(
    base: str,
    model: str | None,
    depth: int,
    ratio: float,
    extra: dict,
) -> tuple[list[dict], dict]:
    """Prefill a ~depth prompt once and keep it as a cached prefix for later turns."""
    pad = filler(int(depth * ratio), uuid.uuid4().hex)
    data, wall = chat_blocking(
        base, model, [{"role": "user", "content": pad}], 1, req_timeout_for(depth), extra
    )
    usage = data.get("usage") or {}
    pt = int(usage.get("prompt_tokens") or 0)
    msg = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    prefix = [
        {"role": "user", "content": pad},
        {"role": "assistant", "content": msg.strip() or "ok"},
    ]
    stats = {
        "prompt_tokens": pt,
        "http_wall_s": wall,
        "wall_s": wall,
        "prefill_tps": (pt / wall) if wall > 0 and pt else None,
    }
    overlay_timings(stats, data.get("timings") if isinstance(data, dict) else None)
    if stats.get("prefill_s") is not None:
        stats["wall_s"] = stats["prefill_s"]
    return prefix, stats


def measure_on_prefix(
    base: str,
    model: str | None,
    prefix: list[dict],
    question: str,
    tg: int,
    depth: int,
    extra: dict,
    exact: bool,
) -> dict:
    messages = list(prefix) + [{"role": "user", "content": question}]
    return chat_stream(base, model, messages, tg, req_timeout_for(depth), extra, exact)


def build_extra(no_think: bool, raw: str, *, ninfer: bool = False, strata: bool = False) -> dict:
    if no_think:
        extra = dict(NINFER_NO_THINK if (ninfer or strata) else NO_THINK)
    else:
        extra = {}
    if raw:
        extra.update(json.loads(raw))
    return extra


def main() -> int:
    p = argparse.ArgumentParser(description="single-depth decode measurement")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--depth", type=int, default=8192)
    p.add_argument("--tg", type=int, default=256)
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--model", default="")
    p.add_argument("--think", action="store_true", help="allow thinking (default off)")
    p.add_argument(
        "--no-exact-tg",
        dest="exact_tg",
        action="store_false",
        help="let the model stop early instead of min_tokens+ignore_eos",
    )
    p.add_argument("--extra-body", default="", help="extra JSON merged into the request")
    args = p.parse_args()

    base = args.url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    extra = build_extra(not args.think, args.extra_body, ninfer=endpoint_is_ninfer(base),
                        strata=endpoint_is_strata(base))
    model = args.model or probe_model(base) or ""
    log(f"endpoint {base}  model {model or '(auto)'}")
    ratio = calibrate(base, model, extra)
    log(f"padding ratio {ratio:.3f} units/token")
    log(f"depth {args.depth}  tg {args.tg}")
    rows = measure_depth(
        base, model, args.depth, args.tg, ratio, extra, args.exact_tg, args.runs
    )
    best = max((r["decode_tps"] or 0) for r in rows)
    log(f"best decode {best:.1f} t/s")
    return 0 if best > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
