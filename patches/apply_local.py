#!/usr/bin/env python3
"""Idempotent patches for Strata v0.1.38 so this bundle launches without a sm_120 rebuild
and so json sampling.force actually overrides the client.

Also applies patches/resident-on-split-v0.1.38.patch so dual-GPU --resident-experts
fills VRAM first and copies only the complement into RAM (Niko1221/Strata#604).

Usage: python3 patches/apply_local.py [$HOME/src/Strata]
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path.home() / "src" / "Strata")


def patch_file(path: pathlib.Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text or (old not in text and label in text):
        print(f"skip {path.name}: {label}")
        return
    if old not in text:
        sys.exit(f"{path}: expected block missing for {label}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"patched {path}: {label}")


def main() -> None:
    setup = ROOT / "setup.py"
    server = ROOT / "serve" / "server.py"
    if not setup.is_file() or not server.is_file():
        sys.exit(f"not a Strata tree: {ROOT}")

    text = setup.read_text(encoding="utf-8")
    if 'os.environ.get("STRATA_SKIP_BUILD")' in text:
        print("skip setup.py: STRATA_SKIP_BUILD")
    else:
        old = (
            "    new_arch = local and not set(archs) <= built\n"
            "    engine_ok = local and (eng / EXE).exists() and meta.get(\"src\") == src and not new_arch\n"
            "    vision_ok = not want_vision or ((eng / VEXE).exists() and (not local or meta.get(\"vision_src\") == vsrc))\n"
        )
        new = (
            "    new_arch = local and not set(archs) <= built\n"
            "    skip_build = os.environ.get(\"STRATA_SKIP_BUILD\") == \"1\"\n"
            "    engine_ok = local and (eng / EXE).exists() and (skip_build or meta.get(\"src\") == src) and not new_arch\n"
            "    vision_ok = not want_vision or ((eng / VEXE).exists() and (skip_build or not local or meta.get(\"vision_src\") == vsrc))\n"
        )
        patch_file(setup, old, new, "STRATA_SKIP_BUILD")

    t = server.read_text(encoding="utf-8")
    if "self.sampling_force = False" not in t:
        patch_file(
            server,
            "        self.sampling_defaults = dict(sampling_defaults or {})   # the run config's `sampling` block\n",
            "        self.sampling_defaults = dict(sampling_defaults or {})   # the run config's `sampling` block\n"
            "        self.sampling_force = False                   # config sampling.force: official params win over the client\n",
            "Service.sampling_force attr",
        )
        t = server.read_text(encoding="utf-8")

    if "if self.sampling_force and self.sampling_defaults:" not in t:
        patch_file(
            server,
            "        defaults = {**self.sampling_defaults, **self.shared}   # the config's, then the Chat settings shared with apps\n"
            "        if defaults:                   # the request's own fields win (explicit 0 stays greedy)\n"
            "            req_values = {k: v for k, v in (sampling or {}).items() if v is not None}\n"
            "            sampling = {**defaults, **req_values}\n",
            "        defaults = {**self.sampling_defaults, **self.shared}   # the config's, then the Chat settings shared with apps\n"
            "        if self.sampling_force and self.sampling_defaults:\n"
            "            sampling = {**(sampling or {}), **self.sampling_defaults}\n"
            "        elif defaults:                   # the request's own fields win (explicit 0 stays greedy)\n"
            "            req_values = {k: v for k, v in (sampling or {}).items() if v is not None}\n"
            "            sampling = {**defaults, **req_values}\n",
            "Service.run force merge",
        )
        t = server.read_text(encoding="utf-8")

    if 'sampling_force = bool((cfg.get("sampling") or {}).get("force"))' not in t:
        patch_file(
            server,
            "        sampling_defaults = sampling_defaults_from_config(cfg)\n"
            "        if sampling_defaults:\n"
            "            pretty = \", \".join(f\"{k}={v}\" for k, v in sampling_defaults.items())\n"
            "            print(f\"[strata] sampling defaults from the config: {pretty}\", flush=True)\n",
            "        sampling_defaults = sampling_defaults_from_config(cfg)\n"
            "        sampling_force = bool((cfg.get(\"sampling\") or {}).get(\"force\"))\n"
            "        if sampling_defaults:\n"
            "            pretty = \", \".join(f\"{k}={v}\" for k, v in sampling_defaults.items())\n"
            "            print(f\"[strata] sampling defaults from the config: {pretty}\", flush=True)\n"
            "            if sampling_force:\n"
            "                print(\"[strata] sampling.force: client temperature/top_p/top_k ignored\", flush=True)\n",
            "main sampling_force from cfg",
        )
        t = server.read_text(encoding="utf-8")

    if "sampling_force = False" not in t.split("else:", 1)[-1][:400] if False else True:
        pass
    if "engine, vision, sampling_defaults = MockEngine" in t and "sampling_force = False" not in t:
        patch_file(
            server,
            "        engine, vision, sampling_defaults = MockEngine",
            "        sampling_force = False\n        engine, vision, sampling_defaults = MockEngine",
            "mock sampling_force",
        )
        t = server.read_text(encoding="utf-8")

    if "svc.sampling_force = sampling_force" not in t:
        patch_file(
            server,
            "                  fit_max_tokens=a.fit_max_tokens or cfg.get(\"fit_max_tokens\") is True)\n",
            "                  fit_max_tokens=a.fit_max_tokens or cfg.get(\"fit_max_tokens\") is True)\n"
            "    svc.sampling_force = sampling_force\n",
            "svc.sampling_force assign",
        )
        t = server.read_text(encoding="utf-8")

    if 'elif key == "force":' not in t:
        patch_file(
            server,
            "            out[key] = value\n"
            "        else:\n"
            "            print(f\"[strata] config sampling.{key}={value!r}: unknown key, ignored\", flush=True)\n",
            "            out[key] = value\n"
            "        elif key == \"force\":\n"
            "            if not isinstance(value, bool):\n"
            "                raise SystemExit(f\"[strata] config sampling.force={value!r}: expected true or false\")\n"
            "        else:\n"
            "            print(f\"[strata] config sampling.{key}={value!r}: unknown key, ignored\", flush=True)\n",
            "sampling.force known key",
        )

    apply_resident_on_split(ROOT)
    print("ok")


def apply_resident_on_split(root: pathlib.Path) -> None:
    """GPU caches first, RAM holds experts no card holds. See RESIDENT_ON_SPLIT.md."""
    gen = root / "src" / "program" / "generate.cpp"
    if not gen.is_file():
        sys.exit(f"missing {gen}")
    if "resident RAM mode with a layer split" in gen.read_text(encoding="utf-8"):
        print("skip generate.cpp: resident-on-split")
        return
    patch = pathlib.Path(__file__).resolve().parent / "resident-on-split-v0.1.38.patch"
    if not patch.is_file():
        sys.exit(f"missing {patch}")
    r = subprocess.run(
        ["patch", "-p1", "--forward", "--batch", "-i", str(patch)],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        sys.stderr.write(r.stdout)
        sys.stderr.write(r.stderr)
        sys.exit(f"resident-on-split patch failed ({r.returncode})")
    print("patched generate.cpp+setup.py: resident-on-split (GPU then RAM)")


if __name__ == "__main__":
    main()
