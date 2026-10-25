#!/usr/bin/env python
"""Bag-of-words documents for VW --lda: one line per document.

    gen_docs.py [--max-paras N] [--chunk 0] [--passes 1 --seed 0] | vw --lda K ...

A document is a paragraph of WORK/corpus.ids, or with --chunk L consecutive
L-token windows (smaller documents make topics closer to local co-occurrence).
Features are integer word ids with counts: `| 17:3 942:1 ...` (default
namespace, so feature i has weight index i).
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import WORK  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-paras", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop-top", type=int, default=0,
                    help="drop the N most frequent words (stop words; ids < N) from documents")
    ap.add_argument("--count-docs", action="store_true", help="print the number of documents and exit")
    args = ap.parse_args()
    paras = [l.split() for l in open(f"{WORK}/corpus.ids")]
    paras = [p for p in paras if len(p) > 1]
    if args.drop_top:  # ids are frequency ranks, so the top N words are ids < N
        paras = [[w for w in p if int(w) >= args.drop_top] for p in paras]
    if args.max_paras:
        paras = paras[:args.max_paras]
    docs = paras
    if args.chunk:
        docs = [p[i:i + args.chunk] for p in paras for i in range(0, len(p), args.chunk)]
        docs = [d for d in docs if len(d) > 1]
    if args.count_docs:
        print(len(docs))
        return
    rng = np.random.default_rng(args.seed)
    for _ in range(args.passes):
        for i in rng.permutation(len(docs)):
            c = Counter(docs[i])
            sys.stdout.write("| " + " ".join(f"{w}:{n}" for w, n in c.items()) + "\n")


if __name__ == "__main__":
    main()
