#!/usr/bin/env python
"""Reference GloVe (numpy, minibatch AdaGrad) on the same counts gen_counts.py uses.

    glove_ref.py OUT.npz [--max-paras N --dim 50 --epochs 25 --lr 0.05 --xmax 100 --batch 4096]

Tells us whether a weak VW GloVe result is VW's fault or the objective's at
this corpus size. Same cells, labels (log X) and weights (min(1,(X/xmax)^.75))
as `gen_counts.py glove --dist-weight`. Output vectors are W + C (as in GloVe).
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gen_counts import counts  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import WORK  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--max-paras", type=int, default=0)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--dim", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--xmax", type=float, default=100.0)
    ap.add_argument("--batch", type=int, default=4096)
    args = ap.parse_args()
    args.dist_weight = True
    r, c, x, V = counts(args)
    y = np.log(x)
    f = np.minimum(1.0, (x / args.xmax) ** 0.75)
    rng = np.random.default_rng(0)
    d = args.dim
    W = (rng.random((V, d)) - 0.5) / d
    C = (rng.random((V, d)) - 0.5) / d
    bw, bc = np.zeros(V), np.zeros(V)
    gW, gC = np.ones((V, d)), np.ones((V, d))
    gbw, gbc = np.ones(V), np.ones(V)
    t = time.time()
    for ep in range(args.epochs):
        perm = rng.permutation(len(r))
        loss = 0.0
        for s in range(0, len(perm), args.batch):
            p = perm[s:s + args.batch]
            i, j = r[p], c[p]
            diff = np.einsum("ij,ij->i", W[i], C[j]) + bw[i] + bc[j] - y[p]
            g = f[p] * diff
            loss += float((f[p] * diff * diff).sum())
            dW = g[:, None] * C[j]
            dC = g[:, None] * W[i]
            np.add.at(gW, i, dW * dW)
            np.add.at(gC, j, dC * dC)
            np.add.at(W, i, -args.lr * dW / np.sqrt(gW[i]))
            np.add.at(C, j, -args.lr * dC / np.sqrt(gC[j]))
            np.add.at(gbw, i, g * g)
            np.add.at(gbc, j, g * g)
            np.add.at(bw, i, -args.lr * g / np.sqrt(gbw[i]))
            np.add.at(bc, j, -args.lr * g / np.sqrt(gbc[j]))
        print(f"epoch {ep + 1} weighted loss {loss / f.sum():.4f}", file=sys.stderr)
    dt = time.time() - t
    vocab = [l.split("\t")[1] for l in open(f"{WORK}/vocab.tsv")]
    np.savez(args.out, words=np.array(vocab), vectors=(W + C).astype(np.float32))
    print(f"glove_ref: {V} x {d} train_s={dt:.0f}")


if __name__ == "__main__":
    main()
