#!/usr/bin/env python
"""Build a ~100M-token corpus for the memory/scale tests, streaming throughout.

    bigcorpus.py OUT_DIR [--tokens 100000000 --min-count 5]

Sources, in order: the WikiText-103 parquet shard in data/wikitext, then the C4
shard in data/c4 (one paragraph per line in both). Same tokenizer as corpus.py.
Unlike corpus.py the vocabulary is "every word seen >= min-count times" (what
gensim does by default), so it is large, which is the point of the test.

Writes OUT_DIR/corpus.txt (all tokens), vocab.tsv, corpus.ids (in-vocab ids)
and corpus.vocab.txt (in-vocab words only, so gensim and VW see identical
tokens).
"""
import argparse
import gzip
import json
from collections import Counter

import pyarrow.parquet as pq

from corpus import TOKEN

from paths import DATA

ROOT = DATA


def source_lines():
    f = pq.ParquetFile(f"{ROOT}/wikitext/train-00000.parquet")
    for batch in f.iter_batches(columns=["text"], batch_size=10000):
        for t in batch.column(0).to_pylist():
            t = t.strip()
            if t and not t.startswith("="):  # skip blank lines and " = Heading = "
                yield t
    with gzip.open(f"{ROOT}/c4/c4-train-00000.json.gz", "rt", encoding="utf-8") as f:
        for line in f:
            for para in json.loads(line)["text"].split("\n"):
                yield para


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--tokens", type=int, default=100_000_000)
    ap.add_argument("--min-count", type=int, default=5)
    args = ap.parse_args()
    counts, n = Counter(), 0
    with open(f"{args.out_dir}/corpus.txt", "w") as out:
        for line in source_lines():
            toks = TOKEN.findall(line.lower())
            if len(toks) < 2:
                continue
            counts.update(toks)
            out.write(" ".join(toks) + "\n")
            n += len(toks)
            if n >= args.tokens:
                break
    vocab = [w for w, c in counts.most_common() if c >= args.min_count]
    wid = {w: i for i, w in enumerate(vocab)}
    with open(f"{args.out_dir}/vocab.tsv", "w") as f:
        for i, w in enumerate(vocab):
            f.write(f"{i}\t{w}\t{counts[w]}\n")
    kept = 0
    with open(f"{args.out_dir}/corpus.txt") as src, open(f"{args.out_dir}/corpus.ids", "w") as fi, \
            open(f"{args.out_dir}/corpus.vocab.txt", "w") as fv:
        for line in src:
            toks = [t for t in line.split() if t in wid]
            kept += len(toks)
            fi.write(" ".join(str(wid[t]) for t in toks) + "\n")
            fv.write(" ".join(toks) + "\n")
    print(f"tokens {n}  unique {len(counts)}  vocab(>= {args.min_count}) {len(vocab)}  in-vocab tokens {kept}")


if __name__ == "__main__":
    main()
