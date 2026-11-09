#!/usr/bin/env python3
"""ngram 手术后质量巡检:批量长生成 + 自动循环说话检测.

每发: POST 127.0.0.1:3425 /v1/chat/completions (model=strata3080/qwen-flash-next, 走生产线路)
循环判据: 12-gram 重复率 > 0.22 或 最长重复片段 >= 60 字符 -> 该发判 LOOP.
thinking 默认开(引擎钉 temperature=1.0), 只统计 content(不含 reasoning).
用法: python3 ngram_looptest.py [tag]   tag 用于报告区分(如 before/after-delete).
"""
import json, sys, urllib.request, collections, time

TAG = sys.argv[1] if len(sys.argv) > 1 else "run"
URL = "http://127.0.0.1:3425/v1/chat/completions"
MODEL = "strata3080/qwen-flash-next"

PROMPTS = [
    ("故事-长", "写一个 600 字左右的中文微小说：一个灯塔看守人收到二十年前沉没船只的求救信号。结尾要有反转。", 1400),
    ("代码", "用 Python 写一个 LRU Cache（get/put，O(1)），附 6 行自测代码，不要解释。", 900),
    ("推理", "一个笼子里有鸡和兔共 35 个头、94 只脚。鸡兔各几只？给出两种不同解法并验算。", 1100),
    ("翻译", "把《道德经》第十二章原文逐句翻译成英文，逐句对照，共九句，不要多余话。", 1200),
    ("列表", "列出 25 种不同的水果，每种一行，只写名字（中文），不要编号以外的标点。", 700),
    ("摘要", "用 5 个要点总结光合作用的过程，每点不超过 30 字，面向初中生。", 700),
    ("闲聊-长", "详细解释为什么天空是蓝色的，以及日落时为什么变红，800 字以内，面向好奇的小学生。", 1500),
    ("古文", "写一篇 400 字左右的文言短文，主题为'论坚持'，不要白话注释。", 1100),
    ("数学-长", "证明 sqrt(2) 是无理数, 然后推广证明对任意非完全平方数 n, sqrt(n) 无理。给出完整步骤。", 1600),
    ("规划", "制定一个 7 天京都行程, 每天 3 个景点 + 餐饮建议, 表格输出。", 1400),
    ("角色", "扮演一个刻薄的资深代码审查员, 点评这段代码的问题: def f(x): return x/0 if x else 1", 800),
    ("诗歌", "写三首不同的五言绝句, 主题分别是雪、码头、旧手机, 每首附一句创作手记。", 1000),
]

def rep_score(text):
    words = list(text)
    n = 12
    grams = collections.Counter()
    for i in range(len(words) - n + 1):
        grams["".join(words[i:i+n])] += 1
    dup = sum(c - 1 for c in grams.values() if c > 1)
    ratio = dup / max(len(grams) + n - 1, 1)
    L = len(words); best = 0
    for i in range(L):
        j = i + max(best, 1)
        while j < L:
            k = 0
            while j+k < L and words[i+k] == words[j+k]: k += 1
            if k > best: best = k; j = i + max(best, 1)
            else: j += 1
    return ratio, best

def one(name, prompt, mx):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": mx, "stream": False, "temperature": 0.6}).encode()
    t0 = time.time()
    try:
        with urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=300) as r:
            d = json.loads(r.read())
        msg = d["choices"][0]["message"]
        content = msg.get("content") or ""
        reas = msg.get("reasoning_content") or ""
        usage = d.get("usage", {})
        ratio, lcs = rep_score(content)
        loop = ratio > 0.22 or lcs >= 60
        flag = "LOOP!!" if loop else "ok"
        open(f"/tmp/loopcase-{TAG}-{abs(hash(name))%999}.txt","w").write(content); print(f"  [{flag}] {name}: {len(content)} 字 in {time.time()-t0:.0f}s (reason {len(reas)}), 12g重复率 {ratio:.2f}, 最长重复片段 {lcs}, tok {usage.get('completion_tokens','?')}")
        return not loop
    except Exception as e:
        print(f"  [ERR] {name}: {e}")
        return False

def main():
    print(f"== ngram 手术质量巡检 [{TAG}] {time.strftime('%H:%M:%S')} ==")
    ok = sum(one(*p) for p in PROMPTS)
    print(f"RESULT [{TAG}]: {ok}/{len(PROMPTS)} 无循环无错误")

main()
