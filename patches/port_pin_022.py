#!/usr/bin/env python3
"""把"钉 sampling"这份本地补丁搬到 Strata v0.2.2 的 serve/server.py。

v0.1.38 时代这份补丁在 `patches/apply_local.py` 里，锚点是 0.1.38/0.1.45 的 serve。
v0.2.2 的 serve 重构过（`svc = Service(...)` 挪到 `sampling_defaults` 之后、fit_max_tokens 那行也改了写法），
老的锚点对不上，所以单独一个脚本，锚点是 0.2.2 的实际文本。

钉的是 json 顶层两样东西：`"sampling": {...}` 给值，`"sampling_force": ["temperature", ...]` 点哪些键客户端改不掉。
官方优先级是 request 最高（DETAILS.md「Order of precedence: the request, ... the config's sampling block, the card」），
所以"无论客户端怎么传都要这个温度"上游没有开关，只能钉。

用法：`python3 patches/port_pin_022.py [STRATA_ROOT]`，默认 `~/src/Strata-0.2.2`。幂等：已经打过就说 skip。

坑（2026-10-22 本机踩过，别再踩）：v0.2.2 把进程看护从 `serve/` 拆成了树根的 `strata_control` 包，
`serve/server.py:6224` 还延迟 import `strata_cli.client`。把 serve 拷进运行目录时这两个包也要软链过去，
否则启动就 `ModuleNotFoundError: No module named 'strata_control'`。自检：
`cd <运行目录> && .venv/bin/python -c "import serve.server, strata_cli.client"`。
"""
from __future__ import annotations

import pathlib
import py_compile
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path.home() / "src" / "Strata-0.2.2")
SERVER = ROOT / "serve" / "server.py"

# 1) 请求路径：Service._run 里把被钉的键压回请求之上
P1_OLD = '''        defaults = {**model_card_sampling(self.model_arch, thinking), **self.sampling_defaults, **self.shared}
        if defaults:                   # the request's own fields win (explicit 0 stays greedy)
            req_values = {k: v for k, v in (sampling or {}).items() if v is not None}
            sampling = {**defaults, **req_values}
'''
P1_NEW = P1_OLD + '''        # 本地补丁 (2026-10-19): sampling_force 的键压过请求 (官方优先级 request 最高, 这里把这几项反过来)
        forced = {k: self.sampling_defaults[k] for k in getattr(self, "sampling_force", ())}
        if forced:
            sampling = {**(sampling or {}), **forced}
'''

# 2) /props：把被钉的键也算进"默认值"报出去，面板不说谎
P2_OLD = '''            defaults = {**model_card_sampling(svc.model_arch, True), **svc.sampling_defaults, **svc.shared}
'''
P2_NEW = P2_OLD + '''            forced = {k: svc.sampling_defaults[k] for k in getattr(svc, "sampling_force", ())}
            if forced:      # /props 也要说实话: 被钉住的键客户端改不了
                defaults = {**defaults, **forced}
'''

# 3) 读配置的那个函数（装在 sampling_defaults_from_config 之前）
P3_ANCHOR = '''def sampling_defaults_from_config(cfg: dict) -> dict:
'''
P3_BLOCK = '''# --- 本地补丁 (2026-10-19): 把 sampling 的某几项钉死, 客户端传什么都不认 ---
# 官方优先级是 request 最高 (DETAILS.md:1376 "Order of precedence: the request, the web page's shared Chat
# settings, the config's sampling block, the card"), 所以"无论客户端怎么传都要这个温度"官方没有开关。
# 用法: strata-<model>.json 里 "sampling": {"temperature": 0.9} 加 "sampling_force": ["temperature"]。
# 只能钉 config 的 sampling 块里已有值的键; 钉了没有的键在启动时点名并忽略。
SAMPLING_FORCEABLE = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty",
                      "repetition_penalty", "penalty_last_n", "seed", "experimental_speed_projection")


def sampling_force_from_config(cfg: dict, sampling_defaults: dict) -> list:
    """The run config's optional `sampling_force`: sampling keys a request cannot override."""
    raw = cfg.get("sampling_force")
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(k, str) for k in raw):
        raise SystemExit('[strata] config sampling_force must be a list of sampling keys, e.g. ["temperature"]')
    have, out = set(sampling_defaults), []
    for k in raw:
        k = SAMPLING_ALIASES.get(k, k)
        if k not in SAMPLING_FORCEABLE:
            print(f"[strata] config sampling_force: {k!r} is not a sampling key, ignored", flush=True)
            continue
        if k not in have:
            print(f"[strata] config sampling_force: {k!r} has no value in the config's sampling block, ignored",
                  flush=True)
            continue
        out.append(k)
    return out


'''

# 4) 启动时赋值 + 打一行实话（v0.2.2 里 svc 在 8130 之后才建，所以锚在 svc.* 那串赋值末尾）
P4_OLD = '''    svc.queue_timeout_s = float(queue_timeout) if queue_timeout is not None else None
'''
P4_NEW = P4_OLD + '''    svc.sampling_force = sampling_force_from_config(cfg, sampling_defaults)      # 本地补丁, 见上
    if getattr(svc, "sampling_force", []):
        print("[strata] sampling forced (the client cannot override): " +
              ", ".join(f"{k}={sampling_defaults[k]}" for k in svc.sampling_force), flush=True)
'''


def sub(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        sys.exit(f"BLOCKING: {SERVER} 里锚点「{label}」应有 1 处，实际 {n} 处（serve 版本对不上，别硬打）")
    return text.replace(old, new, 1)


def main() -> None:
    if not SERVER.is_file():
        sys.exit(f"not a Strata v0.2.2 tree: {SERVER} 不存在")
    text = SERVER.read_text(encoding="utf-8")
    if "sampling_force_from_config" in text:
        print(f"skip serve/server.py: 钉 sampling 补丁已在 {ROOT}")
        return
    for old, new, label in ((P1_OLD, P1_NEW, "Service._run 合并"), (P2_OLD, P2_NEW, "/props"),
                            (P3_ANCHOR, P3_BLOCK + P3_ANCHOR, "sampling_force_from_config 定义"),
                            (P4_OLD, P4_NEW, "svc.sampling_force 赋值")):
        text = sub(text, old, new, label)
    SERVER.write_text(text, encoding="utf-8")
    py_compile.compile(str(SERVER), doraise=True)
    print(f"patched {SERVER}: 钉 sampling (4 处, py_compile 过)")


if __name__ == "__main__":
    main()
