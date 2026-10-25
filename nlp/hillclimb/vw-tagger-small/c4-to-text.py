#!/usr/bin/env python
"""Turn a C4 (en) json.gz shard into `# text = ...` lines for spacy-label.py.

Usage: c4-to-text.py SHARD.json.gz OUT.txt MAX_WORDS

Takes documents in file order (the shard is already in random order), one
`# text` line per line of a document, until MAX_WORDS whitespace-separated
words are written, so a larger MAX_WORDS extends a smaller sample. C4 is
ODC-BY (allenai/c4).
"""
import gzip, json, sys

shard, out, max_words = sys.argv[1], sys.argv[2], int(sys.argv[3])
words = docs = lines = 0
with gzip.open(shard, "rt", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as o:
    for raw in f:
        if words >= max_words:
            break
        docs += 1
        for line in json.loads(raw)["text"].split("\n"):
            t = " ".join(line.split())
            if t:
                o.write(f"# text = {t}\n\n"); lines += 1; words += len(t.split())
print(f"{out}: {docs} documents, {lines} lines, {words} words", file=sys.stderr)
