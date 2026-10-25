#!/usr/bin/env python
"""Tokenize a `# text = ` corpus once, so every method sees identical tokens.

Writes:
    OUT_DIR/corpus.txt   one paragraph per line, tokens space-separated
    OUT_DIR/corpus.ids   same, but as vocabulary ids (out-of-vocab dropped)
    OUT_DIR/corpus.vocab.txt  in-vocabulary words only (gensim's streaming input)
    OUT_DIR/vocab.tsv    id<TAB>word<TAB>count, most frequent first

Tokens are lowercase runs of letters/digits with inner apostrophes
("beecher's", "don't"); all other punctuation is a separator. This is the
single tokenizer for the PPMI, word2vec and VW runs. The earlier attempt
(results/earlier-attempt) used two different ones, which made its comparison
unfair.

Usage: corpus.py TEXT OUT_DIR [--vocab 20000] [--max-tokens N]
"""
import argparse
import re
from collections import Counter

TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")


def paragraphs(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# text = "):
                line = line[len("# text = "):]
            elif not line or line.startswith("#"):
                continue
            toks = TOKEN.findall(line.lower())
            if toks:
                yield toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("out_dir")
    ap.add_argument("--vocab", type=int, default=20000)
    ap.add_argument("--max-tokens", type=int, default=0)
    args = ap.parse_args()

    paras, n = [], 0
    for toks in paragraphs(args.text):
        paras.append(toks)
        n += len(toks)
        if args.max_tokens and n >= args.max_tokens:
            break
    counts = Counter(t for p in paras for t in p)
    vocab = [w for w, _ in counts.most_common(args.vocab)]
    wid = {w: i for i, w in enumerate(vocab)}

    with open(f"{args.out_dir}/corpus.txt", "w") as ft, open(f"{args.out_dir}/corpus.ids", "w") as fi, \
            open(f"{args.out_dir}/corpus.vocab.txt", "w") as fv:
        for p in paras:
            ft.write(" ".join(p) + "\n")
            fi.write(" ".join(str(wid[t]) for t in p if t in wid) + "\n")
            fv.write(" ".join(t for t in p if t in wid) + "\n")
    with open(f"{args.out_dir}/vocab.tsv", "w") as f:
        for i, w in enumerate(vocab):
            f.write(f"{i}\t{w}\t{counts[w]}\n")
    kept = sum(counts[w] for w in vocab)
    print(f"{n} tokens, {len(counts)} types, vocab {len(vocab)} "
          f"(min count {counts[vocab[-1]]}, covers {kept / n:.1%} of tokens)")


if __name__ == "__main__":
    main()
