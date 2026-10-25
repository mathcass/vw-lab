#!/usr/bin/env python
"""Distributional word clusters for the tagger's `c` feature group.

Usage: clusters.py OUT.tsv TEXT [TEXT ...]

Each TEXT is either CoNLL-U (tokens read from column 2) or a file of
`# text = ...` lines (tokenized with spaCy's rule-based English tokenizer,
which matches spacy-label.py's tokens). Words are lowercased.

For the V most frequent words, counts how often each of the C most frequent
words appears immediately left and immediately right of it, turns the counts
into positive PMI, reduces them to D dimensions with an SVD, and k-means
clusters the unit-length vectors three times (K = 64, 256, 1024). Writes
`word<TAB>c64<TAB>c256<TAB>c1024`, one line per word.
"""

import sys
from collections import Counter

import numpy as np

V, C, D, KS, SEED = int(__import__("os").environ.get("CL_V", 50000)), 1000, 100, (64, 256, 1024), 0


def tokens(paths):
    tok = None
    for path in paths:
        with open(path, encoding="utf-8") as f:
            first = f.readline()
            f.seek(0)
            raw = first.startswith("# text = ") and "\t" not in f.read(10000)
            f.seek(0)
            if raw:
                if tok is None:
                    import spacy
                    tok = spacy.blank("en").tokenizer
                for line in f:
                    if line.startswith("# text = "):
                        yield ["<s>"] + [t.text.lower() for t in tok(line[9:].strip())
                                         if not t.is_space] + ["</s>"]
            else:
                sent = []
                for line in f:
                    row = line.rstrip("\n").split("\t")
                    if len(row) == 10 and row[0].isdigit():
                        sent.append(row[1].lower())
                    elif not line.strip() and sent:
                        yield ["<s>"] + sent + ["</s>"]
                        sent = []
                if sent:
                    yield ["<s>"] + sent + ["</s>"]


def kmeans(x, k, rng, iters=25):
    cent = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        assign = np.argmax(x @ cent.T, axis=1)  # unit vectors: max dot = nearest
        for j in range(k):
            members = x[assign == j]
            if len(members):
                c = members.sum(0)
                cent[j] = c / (np.linalg.norm(c) + 1e-9)
            else:
                cent[j] = x[rng.integers(len(x))]
    return assign


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    # Two streaming passes (count, then map to ids) so the text never sits in memory.
    counts = Counter()
    for s in tokens(paths):
        counts.update(s)
    vocab = [w for w, _ in counts.most_common(V)]
    wid = {w: i for i, w in enumerate(vocab)}
    n_tok = sum(counts.values())
    print(f"{n_tok} tokens, {len(counts)} types, keeping {len(vocab)}", file=sys.stderr)
    ids = np.empty(n_tok, dtype=np.int32)
    k = 0
    for s in tokens(paths):
        ids[k:k + len(s)] = [wid.get(w, -1) for w in s]
        k += len(s)
    ids = ids.astype(np.int64)
    # Left and right neighbors; context features are the C most frequent words.
    m = np.zeros(len(vocab) * 2 * C, dtype=np.float32)
    for side, (a, b) in enumerate(((ids[1:], ids[:-1]), (ids[:-1], ids[1:]))):
        ok = (a >= 0) & (b >= 0) & (b < C)
        flat = a[ok] * (2 * C) + side * C + b[ok]
        m += np.bincount(flat, minlength=m.size).astype(np.float32)
        del flat, ok
    m = m.reshape(len(vocab), 2 * C)
    total = float(m.sum(dtype=np.float64))
    rows, cols = m.sum(1, keepdims=True), m.sum(0, keepdims=True)
    # Positive PMI, in place: log(count * total / (row * col)), clipped at 0.
    nz = m > 0
    m[nz] = np.log((m * total / (rows * cols + 1e-9))[nz])
    np.maximum(m, 0, out=m)
    ppmi = m

    # SVD through the small Gram matrix: ppmi = U S Vt, so U S = ppmi @ Vt.T.
    evals, evecs = np.linalg.eigh(ppmi.T @ ppmi)
    top = evecs[:, np.argsort(evals)[::-1][:D]]
    x = ppmi @ top
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9

    rng = np.random.default_rng(SEED)
    assigns = [kmeans(x, k, rng) for k in KS]
    with open(out, "w", encoding="utf-8") as f:
        for i, w in enumerate(vocab):
            if w in ("<s>", "</s>"):
                continue
            f.write(w + "\t" + "\t".join(str(a[i]) for a in assigns) + "\n")
    print(f"{out}: {len(vocab)} words", file=sys.stderr)


if __name__ == "__main__":
    main()
