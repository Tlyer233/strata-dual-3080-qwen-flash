#!/usr/bin/env python3
"""Codex side-request patch for a Strata 0.2.2 tree that already has same-chat rewind.

ccswitch sends strata_checkpoint=false (upstream ckpt=0, unchanged) and
strata_conversation=<thread id>. The engine copies that id onto the parked
snapshot and, on a later real turn of the same id, drops the snapshot when the
new prompt does not extend it. A subagent is a real thread and keeps one slot.
dsh does not send conv=, so this drop never runs for it.

Idempotent. Each anchor must occur once. A half-applied file is refused.
Does not compile, link, or start the engine.

  python3 patches/apply_codex_side_request.py /path/to/Strata-0.2.2
  python3 patches/apply_codex_side_request.py /path/to/Strata-0.2.2 /path/to/runtime/serve/server.py
"""
from __future__ import annotations

import pathlib
import sys


def _once(text: str, old: str, new: str, where: str) -> str:
    n = text.count(old)
    if n != 1:
        sys.exit(f"{where}: expected 1 anchor, found {n}:\n{old[:120]}")
    return text.replace(old, new, 1)


def patch_header(path: pathlib.Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "size_t drop_replaced(" in text and "std::string conv;" in text:
        print(f"skip {path.name}: conv + drop_replaced")
        return
    if "size_t drop_replaced(" in text or "std::string conv;" in text:
        sys.exit(f"{path}: half-applied (conv field and drop_replaced must appear together)")
    if "#include <string>" not in text:
        text = _once(text, "#include <deque>\n", "#include <deque>\n#include <string>\n", str(path))
    text = _once(
        text,
        "    bool cvec = true;\n"
        "    // with a layer split, the later stages' own images, one per stage, in stage order\n",
        "    bool cvec = true;\n"
        "    // Codex thread id of the conversation this snapshot was parked from. Empty: unknown.\n"
        "    // Run-time only. A later real turn of the same id whose prompt does not extend these\n"
        "    // tokens drops the snapshot (a compaction replaced the transcript).\n"
        "    std::string conv;\n"
        "    // with a layer split, the later stages' own images, one per stage, in stage order\n",
        str(path),
    )
    text = _once(
        text,
        "    size_t superseded() const { return superseded_; }\n",
        "    // Drop parked snapshots tagged `conv` whose token sequence is not a strict prefix of this\n"
        "    // prompt. A switch back onto that conversation extends the parked tokens and is kept.\n"
        "    // A rewritten transcript (Codex compaction) does not, and the old snapshot will never be\n"
        "    // asked for again. An empty id drops nothing. The guarded borrow donor is left in place.\n"
        "    template<class Token>\n"
        "    size_t drop_replaced(const std::string& conv, const std::vector<Token>& prompt,\n"
        "                         const std::vector<ConversationImageKey>& images) {\n"
        "        if (conv.empty()) return 0;\n"
        "        size_t dropped = 0;\n"
        "        for (size_t i = 0; i < entries_.size();) {\n"
        "            const auto& e = entries_[i];\n"
        "            if (e.conv == conv && !(guarded_ != 0 && serials_[i] == guarded_) &&\n"
        "                conversation_prefix(e.live, prompt, images) != (int64_t) e.live.ids.size()) {\n"
        "                bytes_ -= e.bytes();\n"
        "                erase(i);\n"
        "                ++dropped;\n"
        "                continue;\n"
        "            }\n"
        "            ++i;\n"
        "        }\n"
        "        return dropped;\n"
        "    }\n"
        "    size_t superseded() const { return superseded_; }\n",
        str(path),
    )
    path.write_text(text, encoding="utf-8")
    print(f"patched {path.name}: conv + drop_replaced")


def patch_generate(path: pathlib.Path) -> None:
    text = path.read_text(encoding="utf-8")
    markers = (
        "std::string live_conv;",
        "image.conv = live_conv;",
        'else if (key == "conv")',
        "conversations.drop_replaced(req_conv",
        "if (live_ok) live_conv = req_conv;",
    )
    have = [m in text for m in markers]
    if all(have):
        print("skip generate.cpp: conv park/drop")
        return
    if any(have):
        sys.exit("generate.cpp: half-applied, refusing to guess:\n" + "\n".join(
            f"  {'yes' if h else 'NO '} {m}" for h, m in zip(have, markers)))
    text = _once(
        text,
        "        bool live_ok = false;\n",
        "        bool live_ok = false;\n"
        "        // Codex thread id of the live conversation (the GEN key conv=). Empty when the client\n"
        "        // did not send one. Copied onto a snapshot at park time.\n"
        "        std::string live_conv;\n",
        str(path),
    )
    text = _once(
        text,
        "                const size_t snapshot_bytes = image.bytes();\n",
        "                image.conv = live_conv;\n"
        "                const size_t snapshot_bytes = image.bytes();\n",
        str(path),
    )
    text = _once(
        text,
        "            int64_t req_pin = 0;\n",
        "            int64_t req_pin = 0;\n"
        "            // conv=<thread id>: ccswitch's strata_conversation. Alphabet is [A-Za-z0-9._:-].\n"
        "            std::string req_conv;\n",
        str(path),
    )
    text = _once(
        text,
        '                    else if (key == "pin") req_pin = std::max<long long>(0, std::atoll(tok.c_str() + eq + 1));\n',
        '                    else if (key == "pin") req_pin = std::max<long long>(0, std::atoll(tok.c_str() + eq + 1));\n'
        '                    else if (key == "conv") req_conv = tok.substr(eq + 1);\n',
        str(path),
    )
    text = _once(
        text,
        "            const auto parked = conversations.best(ids, req_imgs, want_cvec);\n",
        "            // A real turn of a thread whose parked snapshot is not a prefix of this prompt replaced\n"
        "            // that transcript (Codex compaction). Drop it before matching, so it cannot be restored\n"
        "            // or keep the slot. A side request (ckpt=0) does not drop: the parent stays parked.\n"
        "            if (req_ckpt && !req_conv.empty()) {\n"
        "                if (const size_t n = conversations.drop_replaced(req_conv, ids, req_imgs))\n"
        "                    std::fprintf(stderr, \"strata serve: conversation cache: dropped %zu replaced \"\n"
        "                                 \"snapshot%s (conv %.64s); parked=%zu\\n\",\n"
        "                                 n, n == 1 ? \"\" : \"s\", req_conv.c_str(), conversations.size());\n"
        "            }\n"
        "            const auto parked = conversations.best(ids, req_imgs, want_cvec);\n",
        str(path),
    )
    text = _once(
        text,
        "                if (nf_hit) live_ok = false;   // #879 (the next request's start drops the rest)\n",
        "                if (nf_hit) live_ok = false;   // #879 (the next request's start drops the rest)\n"
        "                if (live_ok) live_conv = req_conv;\n",
        str(path),
    )
    text = _once(
        text,
        "                live_ok = true;\n"
        "                std::fprintf(stderr, \"strata serve: cancelled request keeps its conversation: %lld tokens live\\n\",\n",
        "                live_ok = true;\n"
        "                live_conv = req_conv;\n"
        "                std::fprintf(stderr, \"strata serve: cancelled request keeps its conversation: %lld tokens live\\n\",\n",
        str(path),
    )
    path.write_text(text, encoding="utf-8")
    print("patched generate.cpp: conv park/drop")


def patch_server(path: pathlib.Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "strata_conversation" in text:
        print(f"skip {path}: strata_conversation")
        return
    text = _once(
        text,
        '        if sampling.get("strata_checkpoint") is False:\n'
        '            keys += " ckpt=0"\n',
        '        if sampling.get("strata_checkpoint") is False:\n'
        '            keys += " ckpt=0"\n'
        "        # ccswitch sets this on a Codex request whose upstream is Strata. The engine tags the\n"
        "        # parked snapshot with it and drops that snapshot when a later real turn of the same\n"
        "        # thread does not extend it. Absent, or any character outside the token alphabet: ignored.\n"
        '        conv = sampling.get("strata_conversation")\n'
        "        if (isinstance(conv, str) and 1 <= len(conv) <= 128 and\n"
        '                all(c.isalnum() or c in "._:-" for c in conv)):\n'
        '            keys += f" conv={conv}"\n',
        str(path),
    )
    path.write_text(text, encoding="utf-8")
    print(f"patched {path}: strata_conversation")


def patch_test(path: pathlib.Path) -> None:
    if not path.is_file():
        print(f"skip test: {path} absent")
        return
    text = path.read_text(encoding="utf-8")
    if "drop_replaced(" in text:
        print("skip conversation_cache_test.cpp: drop_replaced")
        return
    text = _once(
        text,
        '    std::printf("conversation_cache_test: %d checks passed\\n", checks);\n',
        "    {\n"
        "        // conv=: a later prompt that does not extend this thread's parked tokens drops that snapshot.\n"
        "        // A prompt that continues it keeps the snapshot. Another thread is left alone.\n"
        "        ConversationCache cache(1 << 20, 4);\n"
        "        SavedConversation old_b = image({1, 2, 3, 4, 10, 11});\n"
        '        old_b.conv = "thread-b";\n'
        "        SavedConversation other = image({1, 2, 3, 9, 9});\n"
        '        other.conv = "thread-a";\n'
        '        check(cache.put(std::move(old_b)) && cache.put(std::move(other)), "park two threads");\n'
        '        check(cache.drop_replaced("thread-b", std::vector<int32_t>{1, 2, 3, 4, 10, 11, 12}, {}) == 0 &&\n'
        '              cache.size() == 2, "a prompt that extends the parked tokens keeps it");\n'
        '        check(cache.drop_replaced("thread-b", std::vector<int32_t>{1, 2, 3, 4, 80}, {}) == 1 &&\n'
        '              cache.size() == 1, "a rewritten transcript of the same thread drops the old snapshot");\n'
        '        check(cache.best(std::vector<int32_t>{1, 2, 3, 9, 9, 10}, {}, true).tokens == 5,\n'
        '              "the other thread is still parked");\n'
        '        check(cache.drop_replaced("", std::vector<int32_t>{9}, {}) == 0 && cache.size() == 1,\n'
        '              "an empty id drops nothing");\n'
        "    }\n"
        '    std::printf("conversation_cache_test: %d checks passed\\n", checks);\n',
        str(path),
    )
    path.write_text(text, encoding="utf-8")
    print("patched conversation_cache_test.cpp: drop_replaced")


def main() -> None:
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: apply_codex_side_request.py STRATA_ROOT [RUNTIME_SERVER_PY]")
    root = pathlib.Path(sys.argv[1])
    header = root / "include" / "strata" / "core" / "conversation_cache.hpp"
    gen = root / "src" / "program" / "generate.cpp"
    server = root / "serve" / "server.py"
    test = root / "src" / "core" / "conversation_cache_test.cpp"
    for p in (header, gen, server):
        if not p.is_file():
            sys.exit(f"missing {p}")
    patch_header(header)
    patch_generate(gen)
    patch_server(server)
    patch_test(test)
    if len(sys.argv) == 3:
        extra = pathlib.Path(sys.argv[2])
        if not extra.is_file():
            sys.exit(f"missing {extra}")
        patch_server(extra)


if __name__ == "__main__":
    main()
