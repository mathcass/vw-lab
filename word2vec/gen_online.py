#!/usr/bin/env python
"""One-pass, fixed-memory CBOW examples for VW: no vocabulary, no counting pass.

    gen_online.py CORPUS.txt [--bits 20 --buffer 1000000 --window 5 --negatives 5
                              --sample 1e-3 --seed 0] | vw -q wc --rank K -b B ...

Reads raw tokens (one paragraph per line, every word, however rare) once, in
order, and writes the words themselves for VW to hash. What gen.py gets from a
full counting pass is estimated on the fly in fixed memory:

  counts       running counts in a 2^bits array indexed by a hash of the word
               (collisions only inflate a word's count a little)
  subsampling  word2vec's keep probability from running count / running total
  negatives    drawn from a ring buffer of the last --buffer tokens (a sample of
               the unigram distribution), each kept with probability
               count^-0.25, which leaves word2vec's unigram^0.75 distribution

Memory is O(2^bits + buffer) whatever the vocabulary or corpus size, so with
VW's hashed weights the whole pipeline can run on an endless text stream.
"""
import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--bits", type=int, default=20)
    ap.add_argument("--buffer", type=int, default=1_000_000)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--negatives", type=int, default=5)
    ap.add_argument("--sample", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mask = (1 << args.bits) - 1
    counts = np.zeros(mask + 1, dtype=np.int64)
    total = 0
    R = args.buffer
    buf_w = np.empty(R, dtype=object)   # recent tokens (negative-sampling pool)
    buf_h = np.zeros(R, dtype=np.int64)  # their count buckets
    pos = filled = 0
    rng = np.random.default_rng(args.seed)
    W, K = args.window, args.negatives
    out = sys.stdout

    with open(args.corpus) as f:
        for line in f:
            words = line.split()
            n = len(words)
            if n < 2:
                continue
            h = np.fromiter((hash(w) & mask for w in words), dtype=np.int64, count=n)
            np.add.at(counts, h, 1)
            total += n
            idx = (pos + np.arange(n)) % R
            buf_w[idx] = np.array(words, dtype=object)
            buf_h[idx] = h
            pos, filled = (pos + n) % R, min(R, filled + n)

            c = counts[h].astype(np.float64)
            thr = args.sample * total
            keep = np.minimum(1.0, (np.sqrt(c / thr) + 1) * thr / c)
            ws = [w for w, k in zip(words, rng.random(n) < keep) if k]
            n = len(ws)
            if n < 2:
                continue
            r = rng.integers(1, W + 1, n)

            need, got, negs = n * K, 0, []
            while got < need:
                j = rng.integers(0, filled, need * 20)
                sel = j[rng.random(len(j)) < counts[buf_h[j]].astype(np.float64) ** -0.25]
                negs.append(buf_w[sel])
                got += len(sel)
            negs = np.concatenate(negs)[:need]

            lines = []
            for i in range(n):
                ctx = ws[max(0, i - r[i]):i] + ws[i + 1:i + 1 + r[i]]
                if not ctx:
                    continue
                bag = " ".join(ctx)
                lines.append(f"1 |w {ws[i]} |c {bag}")
                for nn in negs[i * K:(i + 1) * K]:
                    lines.append(f"-1 |w {nn} |c {bag}")
            out.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
