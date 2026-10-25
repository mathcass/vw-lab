#!/usr/bin/env python
"""Score word vectors on a fixed vocabulary so every method is comparable.

    evalvec.py VECS.npz [--name NAME --train-s SECONDS --notes TEXT]

VECS.npz holds `words` and `vectors`. Words outside WORK/vocab.tsv (the 10M corpus's 20k words) are ignored;
vocabulary words a method failed to produce count as zero vectors (and are
reported as `missing`), so coverage is identical across methods.

Metrics
  simlex   SimLex-999 Spearman (similarity), with a bootstrap 95% CI
  ws353    WordSim-353 Spearman (relatedness), with a bootstrap 95% CI
  analogy  Google analogies (3CosAdd, top-1) over questions whose four words
           are all in the vocabulary; the search space is the whole vocabulary
With --name, a row is appended to results.tsv (train time included).
"""
import argparse
import csv
import os
import time

import numpy as np

from paths import BENCH, RESULTS, WORK


def rank(a):
    a = np.asarray(a, dtype=np.float64)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    return ((csum - counts + csum + 1) / 2.0)[inv]


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    return float(rx @ ry / (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-12))


def load_pairs(path, delim, header=True, score_col=2):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter=delim))[1 if header else 0:]
    out = []
    for r in rows:
        try:
            out.append((r[0].lower(), r[1].lower(), float(r[score_col])))
        except (IndexError, ValueError):
            pass
    return out


def load_vocab():
    with open(f"{WORK}/vocab.tsv") as f:
        return [line.split("\t")[1] for line in f]


def normalized(vocab, words, vectors):
    idx = {w: i for i, w in enumerate(words)}
    M = np.zeros((len(vocab), vectors.shape[1]), dtype=np.float32)
    missing = 0
    for i, w in enumerate(vocab):
        j = idx.get(w)
        if j is None:
            missing += 1
        else:
            M[i] = vectors[j]
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    return M, missing


def sim_score(M, wid, pairs, rng, boots=1000):
    got = [(wid[a], wid[b], s) for a, b, s in pairs if a in wid and b in wid]
    a = np.array([g[0] for g in got])
    b = np.array([g[1] for g in got])
    human = np.array([g[2] for g in got])
    cos = np.einsum("ij,ij->i", M[a], M[b])
    rho = spearman(cos, human)
    bs = [spearman(cos[s], human[s]) for s in
          (rng.integers(0, len(got), len(got)) for _ in range(boots))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return rho, lo, hi, len(got)


def analogy(M, wid):
    qs = []
    with open(f"{BENCH}/questions-words.txt") as f:
        for line in f:
            if line.startswith(":"):
                continue
            ws = line.lower().split()
            if len(ws) == 4 and all(w in wid for w in ws):
                qs.append([wid[w] for w in ws])
    q = np.array(qs)
    correct = 0
    for s in range(0, len(q), 512):
        b = q[s:s + 512]
        t = M[b[:, 1]] - M[b[:, 0]] + M[b[:, 2]]
        sc = t @ M.T
        r = np.arange(len(b))
        for k in range(3):
            sc[r, b[:, k]] = -np.inf
        correct += int((sc.argmax(1) == b[:, 3]).sum())
    return correct / len(q), len(q)


def evaluate(path):
    z = np.load(path, allow_pickle=True)
    vocab = load_vocab()
    wid = {w: i for i, w in enumerate(vocab)}
    raw = z["vectors"].astype(np.float32)
    nan_rows = int(np.isnan(raw).any(1).sum())
    if nan_rows:
        print(f"WARNING: {nan_rows} rows contain NaN; zeroed (they score as missing)")
        raw = np.nan_to_num(raw, nan=0.0)
    M, missing = normalized(vocab, [str(w) for w in z["words"]], raw)
    rng = np.random.default_rng(0)
    sl = sim_score(M, wid, load_pairs(f"{BENCH}/SimLex-999.txt", "\t", score_col=3), rng)
    ws = sim_score(M, wid, load_pairs(f"{BENCH}/ws353.csv", ","), rng)
    an, nq = analogy(M, wid)
    return dict(dim=z["vectors"].shape[1], missing=missing,
                simlex=sl[0], simlex_ci=f"[{sl[1]:+.2f},{sl[2]:+.2f}]", n_simlex=sl[3],
                ws353=ws[0], ws353_ci=f"[{ws[1]:+.2f},{ws[2]:+.2f}]", n_ws353=ws[3],
                analogy=an, n_analogy=nq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vecs")
    ap.add_argument("--name")
    ap.add_argument("--train-s", default="-")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    r = evaluate(args.vecs)
    print(f"{args.name or args.vecs}: dim {r['dim']} missing {r['missing']} | "
          f"SimLex {r['simlex']:+.3f} {r['simlex_ci']} (n={r['n_simlex']}) | "
          f"WS353 {r['ws353']:+.3f} {r['ws353_ci']} (n={r['n_ws353']}) | "
          f"analogy {r['analogy']:.3f} (n={r['n_analogy']}) | train {args.train_s}s")
    if args.name:
        path = f"{RESULTS}/results.tsv"
        new = not os.path.exists(path)
        with open(path, "a") as f:
            if new:
                f.write("name\ttrain_s\tdim\tmissing\tsimlex\tsimlex_ci\tws353\tws353_ci\tanalogy\tnotes\tdate\n")
            f.write(f"{args.name}\t{args.train_s}\t{r['dim']}\t{r['missing']}\t{r['simlex']:+.3f}\t"
                    f"{r['simlex_ci']}\t{r['ws353']:+.3f}\t{r['ws353_ci']}\t{r['analogy']:.3f}\t"
                    f"{args.notes}\t{time.strftime('%Y-%m-%d %H:%M')}\n")


if __name__ == "__main__":
    main()
