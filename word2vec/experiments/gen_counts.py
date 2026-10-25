#!/usr/bin/env python
"""Stream co-occurrence-matrix examples (GloVe / PPMI objectives) to VW.

    gen_counts.py MODE [--passes 20 --window 5 --max-paras N --seed 0] | vw ...

The (word, context) count matrix is built once from WORK/corpus.ids (symmetric
window; with --dist-weight each co-occurrence at distance d counts 1/d, as in
GloVe) and cached in WORK. Every pass is reshuffled: VW's own --passes replays
the cache in a fixed order, and the earlier attempt wrote cells in
first-seen order, which is poor for SGD.

Modes (one VW line per matrix cell, `LABEL WEIGHT |w i |c j`)
  glove      nonzero cells, label log(X), weight min(1, (X/xmax)^0.75);
             the per-word linear terms play the role of GloVe's biases
  ppmi       positive-PMI cells only (what the earlier attempt did)
  ppmi_full  positive-PMI cells plus `--zeros` uniformly sampled cells (label
             0), importance-weighted so the expected loss is the loss over the
             WHOLE matrix, i.e. the objective truncated SVD minimizes
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import WORK  # noqa: E402


def counts(args):
    path = f"{WORK}/cooc_w{args.window}_d{int(args.dist_weight)}_p{args.max_paras}.npz"
    if os.path.exists(path):
        z = np.load(path)
        return z["r"], z["c"], z["x"], int(z["V"])
    paras = [np.array(l.split(), dtype=np.int64) for l in open(f"{WORK}/corpus.ids")]
    paras = [p for p in paras if len(p) > 1]
    if args.max_paras:
        paras = paras[:args.max_paras]
    V = sum(1 for _ in open(f"{WORK}/vocab.tsv"))
    ids = np.concatenate(paras)
    pid = np.concatenate([np.full(len(p), i) for i, p in enumerate(paras)])
    acc = {}
    for d in range(1, args.window + 1):
        ok = pid[:-d] == pid[d:]
        key = np.concatenate([ids[:-d][ok] * V + ids[d:][ok], ids[d:][ok] * V + ids[:-d][ok]])
        u, n = np.unique(key, return_counts=True)
        w = n / d if args.dist_weight else n.astype(np.float64)
        for k, v in zip(u.tolist(), w.tolist()):
            acc[k] = acc.get(k, 0.0) + v
    keys = np.fromiter(acc.keys(), dtype=np.int64, count=len(acc))
    x = np.fromiter(acc.values(), dtype=np.float64, count=len(acc))
    r, c = keys // V, keys % V
    np.savez(path, r=r, c=c, x=x, V=V)
    return r, c, x, V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["glove", "ppmi", "ppmi_full"])
    ap.add_argument("--passes", type=int, default=20)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--dist-weight", action="store_true")
    ap.add_argument("--xmax", type=float, default=100.0)
    ap.add_argument("--cds", type=float, default=0.75)
    ap.add_argument("--zeros", type=float, default=1.0, help="ppmi_full: sampled zero cells per positive cell")
    ap.add_argument("--max-paras", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    r, c, x, V = counts(args)
    rng = np.random.default_rng(args.seed)

    if args.mode == "glove":
        label = np.log(x)
        weight = np.minimum(1.0, (x / args.xmax) ** 0.75)
    else:
        row = np.bincount(r, weights=x, minlength=V)
        col = np.bincount(c, weights=x, minlength=V) ** args.cds
        pmi = np.log(x * col.sum() / (row[r] * col[c]))
        keep = pmi > 0
        r, c, label = r[keep], c[keep], pmi[keep]
        weight = np.ones(len(r))
    n = len(r)
    print(f"{args.mode}: {n} cells, label mean {label.mean():.3f} var {label.var():.3f}", file=sys.stderr)

    out = sys.stdout
    for _ in range(args.passes):
        rr, cc, ll, ww = r, c, label, weight
        if args.mode == "ppmi_full":
            # Sampled cells stand in for all V*V - n cells: weight each by
            # (V*V - n) / z relative to weight 1 per positive cell; then scale
            # everything so the largest weight is 1 (only the ratio matters).
            z = int(args.zeros * n)
            zr, zc = rng.integers(0, V, z), rng.integers(0, V, z)
            zw = (V * V - n) / z
            rr = np.concatenate([r, zr]); cc = np.concatenate([c, zc])
            ll = np.concatenate([label, np.zeros(z)])
            ww = np.concatenate([np.full(n, 1.0 / zw), np.ones(z)])
            # a sampled cell can be a positive one; with V^2 >> n this is rare
        perm = rng.permutation(len(rr))
        for s in range(0, len(perm), 200_000):
            p = perm[s:s + 200_000]
            out.write("".join(f"{l:.5g} {w:.4g} |w {i} |c {j}\n"
                              for l, w, i, j in zip(ll[p].tolist(), ww[p].tolist(),
                                                    rr[p].tolist(), cc[p].tolist())))


if __name__ == "__main__":
    main()
