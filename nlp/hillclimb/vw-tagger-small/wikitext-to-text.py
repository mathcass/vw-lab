#!/usr/bin/env python
"""Turn a WikiText-103 raw parquet shard into `# text = ...` lines for spacy-label.py.

Usage: wikitext-to-text.py SHARD.parquet OUT.conllu MAX_WORDS [SEED]

Skips headings, undoes WikiText's tokenization spacing (" @-@ " -> "-",
" , " -> ", " and so on) so spaCy sees ordinary text, and samples paragraphs
at random (seeded) until MAX_WORDS whitespace-separated words are written.
WikiText is CC BY-SA 4.0 (Wikipedia).
"""

import random
import re
import sys

import pyarrow.parquet as pq

RULES = [
    (r" @([-.,])@ ", r"\1"),                   # 1 @,@ 000 -> 1,000; role @-@ playing
    (r" ([,.;:!?%)\]}])", r"\1"),              # no space before closing punctuation
    (r"([(\[{]) ", r"\1"),                     # no space after opening brackets
    (r" (n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1"),  # clitics
]


def detok(text):
    text = " " + text.strip() + " "
    for pat, rep in RULES:
        text = re.sub(pat, rep, text)
    return " ".join(text.split())


def main():
    shard, out, max_words = sys.argv[1], sys.argv[2], int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    paras = [p for p in pq.read_table(shard).column("text").to_pylist()
             if p.strip() and not p.strip().startswith("=")]
    random.Random(seed).shuffle(paras)
    words = n = 0
    with open(out, "w", encoding="utf-8") as f:
        for p in paras:
            if words >= max_words:
                break
            t = detok(p)
            f.write(f"# text = {t}\n\n")
            words += len(t.split()); n += 1
    print(f"{out}: {n} paragraphs, {words} words", file=sys.stderr)


if __name__ == "__main__":
    main()
