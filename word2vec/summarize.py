#!/usr/bin/env python
"""Print markdown tables of selected rows from results.tsv.

    summarize.py NAME [NAME ...]      (exact row names, e.g. dev_w2v dev_cbow_e5_pistol.wc;
                                       the latest row of each name is shown)
"""
import csv
import sys

from paths import RESULTS

rows = {r["name"]: r for r in csv.DictReader(open(f"{RESULTS}/results.tsv"), delimiter="\t")}
print("| run | train s | SimLex | WordSim | analogy |")
print("|---|---|---|---|---|")
for n in sys.argv[1:]:
    r = rows.get(n)
    if r is None:
        print(f"| {n} | (missing) | | | |")
        continue
    print(f"| `{n}` | {r['train_s']} | {r['simlex']} | {r['ws353']} | {r['analogy']} |")
