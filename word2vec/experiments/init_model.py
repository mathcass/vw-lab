#!/usr/bin/env python
"""Write a VW --lrq model whose word factors start small and centred.

    init_model.py OUT.model K "VW_ARGS" [--scale 0.5 --dist uniform --seed 0]

VW's lrq starts each factor at a non-negative pseudo-random value the first
time it is touched (about half come out ~0, the rest spread up to 1-2), so
every word vector starts with a large shared positive component. This writes
the vocabulary's factors as U(-scale/K, +scale/K) (word2vec's input init) or
N(0, scale/sqrt(K)) instead. lrq only initializes factors that are exactly 0,
so training with `-i OUT.model` keeps these values. VW_ARGS must be the same
model flags used for training (the -b and the optimizer flags change the layout).
"""
import argparse
import os
import sys

import numpy as np
from vowpalwabbit import Workspace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import WORK  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("k", type=int)
    ap.add_argument("vwargs")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--dist", choices=["uniform", "normal"], default="uniform")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    V = sum(1 for _ in open(f"{WORK}/vocab.tsv"))
    K = args.k
    vw = Workspace(f"--lrq wc{K} {args.vwargs} --quiet")
    N = vw.num_weights()
    step = 1 << int(np.ceil(np.log2(K + 1)))
    rng = np.random.default_rng(args.seed)
    for ns in "wc":
        h = vw.hash_space(ns)
        if args.dist == "uniform":
            F = rng.uniform(-args.scale / K, args.scale / K, (V, K))
        else:
            F = rng.normal(0, args.scale / np.sqrt(K), (V, K))
        F[F == 0] = 1e-6  # exactly-zero factors would be re-initialized by lrq
        for i in range(V):
            b = ((h + i) * step) % N
            for k in range(K):
                vw.set_weight(b + 1 + k, 0, float(F[i, k]))
    vw.save(args.out)
    vw.finish()
    print(f"wrote {args.out}: {V} words x {K} per namespace, {args.dist} scale {args.scale}")


if __name__ == "__main__":
    main()
