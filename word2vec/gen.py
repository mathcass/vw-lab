#!/usr/bin/env python
"""Stream VW training examples for word embeddings to stdout.

    gen.py MODE [--epochs 5 --window 5 --negatives 5 --sample 1e-3 --seed 0] | vw ...

Reads WORK/corpus.ids (see paths.py; or --corpus), streaming it once per epoch, so memory
is O(vocabulary) whatever the corpus size. Each epoch resamples word2vec subsampling, the dynamic
window and the negatives, the way word2vec does, so there is no need to bake
epochs into a file or use VW --passes (which the named weight dumps can't do).
Features are integer word ids, which VW uses as-is (no string hashing), so
the extractor can compute every weight's index directly.

Modes
  sgns   one line per (target, context) pair plus K negative lines:
             1 |w T |c C          -1 |w T |c N1   ...
  cbow   one line per target with its whole window as the context bag, plus K
         lines that swap in negative targets:
             1 |w T |c C1 C2 ...   -1 |w N1 |c C1 C2 ...
  sgbag  like cbow but each context is weighted 1/len(window), so the score is
         the mean context vector dotted with the target
  ldf    one multi-line example per (target, context) pair for --csoaa_ldf /
         --wap_ldf: the target is shared, the true context costs 0 and K
         negatives cost 1, so VW ranks the true context against the negatives:
             shared |w T
             1:0 |c C
             2:1 |c N1  ...
             (blank line)
Lines come out in corpus order (target by target), like word2vec.
"""
import argparse
import sys

import numpy as np

from paths import WORK


def paragraphs(path, max_paras=0):
    """Stream paragraphs (len > 1) from the ids file, so memory stays O(vocab)."""
    n = 0
    with open(path) as f:
        for line in f:
            p = np.array(line.split(), dtype=np.int64)
            if len(p) < 2:
                continue
            yield p
            n += 1
            if max_paras and n >= max_paras:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sgns", "cbow", "sgbag", "ldf"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--negatives", type=int, default=5)
    ap.add_argument("--sample", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-paras", type=int, default=0)
    ap.add_argument("--corpus", default=f"{WORK}/corpus.ids")
    ap.add_argument("--words", help="vocab.tsv: write words instead of ids (VW hashes them)")
    ap.add_argument("--ctx-ns", default="c", help="context namespace name (its hash decides where context ids land in the weights)")
    args = ap.parse_args()

    counts = np.zeros(0, dtype=np.int64)
    for p in paragraphs(args.corpus, args.max_paras):
        c = np.bincount(p)
        if len(c) > len(counts):
            counts = np.pad(counts, (0, len(c) - len(counts)))
        counts[:len(c)] += c
    total = counts.sum()
    # word2vec/gensim keep probability
    thr = args.sample * total
    keep_p = np.minimum(1.0, (np.sqrt(counts / thr) + 1) * thr / np.maximum(counts, 1))
    neg_cdf = np.cumsum(counts ** 0.75)
    neg_cdf /= neg_cdf[-1]
    rng = np.random.default_rng(args.seed)
    out = sys.stdout
    W, K, C = args.window, args.negatives, args.ctx_ns
    # feature names: integer ids (VW uses them as-is) or, with --words, the words
    # themselves (VW hashes them, so -b can be smaller than the vocabulary)
    L = ([l.split("\t")[1] for l in open(args.words)] if args.words
         else [str(i) for i in range(len(counts))])

    for _ in range(args.epochs):
        for p in paragraphs(args.corpus, args.max_paras):
            p = p[rng.random(len(p)) < keep_p[p]]
            n = len(p)
            if n < 2:
                continue
            r = rng.integers(1, W + 1, n)
            lines = []
            if args.mode in ("sgns", "ldf"):
                for i in range(n):
                    ctx = np.concatenate([p[max(0, i - r[i]):i], p[i + 1:i + 1 + r[i]]])
                    if not len(ctx):
                        continue
                    negs = np.searchsorted(neg_cdf, rng.random(len(ctx) * K))
                    t = p[i]
                    for j, c in enumerate(ctx):
                        if args.mode == "ldf":
                            lines.append(f"shared |w {L[t]}\n1:0 |{C} {L[c]}\n" + "".join(
                                f"{k + 2}:1 |{C} {L[nn]}\n" for k, nn in enumerate(negs[j * K:(j + 1) * K])))
                            continue
                        lines.append(f"1 |w {L[t]} |{C} {L[c]}")
                        for nn in negs[j * K:(j + 1) * K]:
                            lines.append(f"-1 |w {L[t]} |{C} {L[nn]}")
            else:
                negs = np.searchsorted(neg_cdf, rng.random(n * K))
                for i in range(n):
                    ctx = np.concatenate([p[max(0, i - r[i]):i], p[i + 1:i + 1 + r[i]]])
                    if not len(ctx):
                        continue
                    if args.mode == "cbow":
                        bag = " ".join([L[c] for c in ctx])
                    else:
                        v = 1.0 / len(ctx)
                        bag = " ".join(f"{L[c]}:{v:.4g}" for c in ctx)
                    lines.append(f"1 |w {L[p[i]]} |{C} {bag}")
                    for nn in negs[i * K:(i + 1) * K]:
                        lines.append(f"-1 |w {L[nn]} |{C} {bag}")
            out.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
