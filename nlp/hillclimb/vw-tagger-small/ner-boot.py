#!/usr/bin/env python
"""Sentence-level bootstrap CI for NER F1, paired against a baseline, plus
per-type F1. Usage: HC_DATA=... ner-boot.py SPLIT RUN [BASELINE_RUN]"""
import os, random, sys
from collections import Counter
import hc

split, run, base = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else None)
names = {int(i): t for t, i in (l.split("\t") for l in (hc.DATA / "train.tags").read_text().splitlines())}
gold = hc.read_vw(hc.DATA / f"{split}.conllu")

def counts(r):
    preds = hc.read_predictions(hc.SCRATCH / "runs" / r / f"{split}.pred")
    rows, by_type = [], Counter()
    for g, p in zip(gold, preds):
        gs = set(hc.spans([int(y) for y, _ in g], names)); ps = set(hc.spans([int(y) for y in p], names))
        rows.append((len(gs & ps), len(gs), len(ps)))
        for s in gs & ps: by_type[s[0], "tp"] += 1
        for s in gs: by_type[s[0], "g"] += 1
        for s in ps: by_type[s[0], "p"] += 1
    return rows, by_type

def f1(rows, idx):
    tp = sum(rows[i][0] for i in idx); g = sum(rows[i][1] for i in idx); p = sum(rows[i][2] for i in idx)
    return 200 * tp / max(g + p, 1)

a, ta = counts(run)
b, tb = counts(base) if base else (None, None)
rng = random.Random(0); n = len(a); full = list(range(n))
samp = [[rng.randrange(n) for _ in range(n)] for _ in range(1000)]
fa = sorted(f1(a, s) for s in samp)
print(f"{run} {split} F1 {f1(a, full):.2f} (95% CI {fa[25]:.2f}-{fa[974]:.2f})")
if b:
    d = sorted(f1(a, s) - f1(b, s) for s in samp)
    print(f"  vs {base}: {f1(a, full) - f1(b, full):+.2f} (95% CI {d[25]:+.2f}..{d[974]:+.2f})")
types = sorted({t for t, _ in ta}, key=lambda t: -ta[t, "g"])
print("  type gold F1" + ("  base_F1" if b else ""))
for t in types:
    ft = 200 * ta[t, "tp"] / max(ta[t, "g"] + ta[t, "p"], 1)
    fb = f"  {200 * tb[t, 'tp'] / max(tb[t, 'g'] + tb[t, 'p'], 1):6.1f}" if b else ""
    print(f"  {t:12s} {ta[t, 'g']:4d} {ft:5.1f}{fb}")
