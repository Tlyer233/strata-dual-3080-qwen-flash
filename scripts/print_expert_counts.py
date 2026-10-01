#!/usr/bin/env python3
"""Print GPU vs CPU expert counts from a live Strata log (generic)."""
from __future__ import annotations

import json
import pathlib
import re
import sys

cfg = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path.home() / "src" / "Strata" / "strata-swift-iq3_xxs.json")
log = pathlib.Path(json.loads(cfg.read_text())["log"])
text = log.read_text(encoding="utf-8", errors="replace")
chunks = text.split("strata generate: loaded")
if len(chunks) < 2:
    sys.exit(f"no 'strata generate: loaded' in {log}")
tail = chunks[-1]
gpu = re.findall(r"gpu[_\s-]*experts?\s*[:=]\s*(\d+)", tail, re.I)
cpu = re.findall(r"cpu[_\s-]*experts?\s*[:=]\s*(\d+)", tail, re.I)
print(log)
print(tail[:2500])
if gpu or cpu:
    print(f"gpu_experts={gpu[-1] if gpu else '?'} cpu_experts={cpu[-1] if cpu else '?'}")
