#!/usr/bin/env python
"""Compare two analyses of the same text: spaCy's (REFERENCE) and another's.

Usage:
    uv run python compare.py REFERENCE.conllu SYSTEM.conllu

Both files are in spacy-label.py's format with `# newpar` on each
paragraph's first sentence (spacy-label.py on a .txt file, and pipeline.py,
write these). Paragraphs are matched in order and must have the same tokens
(both use spaCy's tokenizer); sentences may be split differently, so
everything is scored per paragraph:

  sentence starts  F1 over tokens other than a paragraph's first, and the
                   share of REFERENCE sentences found exactly
  tags             XPOS agreement over all tokens
  entities         exact-span F1 (type and boundaries)
  parses           UAS / LAS, where a head is right if it is the same token
                   of the paragraph (or both are roots); over all tokens and
                   without punctuation. A sentence-split error makes some
                   heads impossible to get right.

Scores come with 95% paragraph-bootstrap intervals.
"""

import random
import sys


def read(path):
    """Paragraphs as lists of tokens: (word, starts_sentence, xpos, head, rel, ner)."""
    paragraphs, sent = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("# newpar"):
                paragraphs.append([])
                continue
            if line.startswith("#"):
                continue
            if not line.strip():
                if sent:
                    paragraphs[-1].extend(sent)
                sent = []
                continue
            row = line.rstrip("\n").split("\t")
            if not row[0].isdigit():
                continue
            if not paragraphs:
                sys.exit(f"{path}: no `# newpar` before the first sentence")
            offset = len(paragraphs[-1])
            i = int(row[0]) - 1
            head = None if row[6] in ("0", "_") else offset + int(row[6]) - 1
            ner = next((m[4:] for m in row[9].split("|") if m.startswith("NER=")), "O")
            sent.append((row[1], i == 0, row[4], head, row[7], ner))
    if sent:
        paragraphs[-1].extend(sent)
    return paragraphs


def spans(tags):
    out, cur = set(), None
    for i, tag in enumerate(tags + ["O"]):
        if cur and (tag == "O" or tag.startswith("B-") or tag[2:] != cur[0]):
            out.add((cur[0], cur[1], i))
            cur = None
        if tag != "O" and cur is None:
            cur = (tag[2:], i)
    return out


def sentences(par):
    starts = [i for i, t in enumerate(par) if t[1]] + [len(par)]
    return set(zip(starts, starts[1:]))


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    ref, sys_ = read(sys.argv[1]), read(sys.argv[2])
    if len(ref) != len(sys_):
        sys.exit(f"{len(ref)} reference paragraphs but {len(sys_)} system paragraphs")

    # Per paragraph: (hits, total) for ratios, (tp, gold, pred) for F1s.
    rows = []
    skipped = 0
    for r, s in zip(ref, sys_):
        if [t[0] for t in r] != [t[0] for t in s]:
            skipped += 1
            continue
        g = {i for i, t in enumerate(r) if t[1] and i}
        p = {i for i, t in enumerate(s) if t[1] and i}
        gs, ps = sentences(r), sentences(s)
        ge, pe = spans([t[5] for t in r]), spans([t[5] for t in s])
        content = [i for i, t in enumerate(r) if t[4] != "punct"]
        rows.append({
            "start": (len(g & p), len(g), len(p)),
            "exact": (len(gs & ps), len(gs)),
            "tags": (sum(a[2] == b[2] for a, b in zip(r, s)), len(r)),
            "ent": (len(ge & pe), len(ge), len(pe)),
            "uas": (sum(a[3] == b[3] for a, b in zip(r, s)), len(r)),
            "las": (sum(a[3] == b[3] and a[4] == b[4] for a, b in zip(r, s)), len(r)),
            "uas_np": (sum(r[i][3] == s[i][3] for i in content), len(content)),
            "las_np": (sum(r[i][3] == s[i][3] and r[i][4] == s[i][4] for i in content), len(content)),
        })

    def score(key, idx):
        vals = [rows[i][key] for i in idx]
        if len(vals[0]) == 3:
            tp, g, p = (sum(v[k] for v in vals) for k in range(3))
            return 200 * tp / max(g + p, 1)
        return 100 * sum(v[0] for v in vals) / max(sum(v[1] for v in vals), 1)

    n = len(rows)
    rng = random.Random(0)
    samples = [[rng.randrange(n) for _ in range(n)] for _ in range(1000)]
    print(f"paragraphs           {n:>8}" + (f"   ({skipped} skipped: tokens differ)" if skipped else ""))
    print(f"tokens               {sum(r['tags'][1] for r in rows):>8}")
    print(f"reference sentences  {sum(r['exact'][1] for r in rows):>8}")
    print(f"reference entities   {sum(r['ent'][1] for r in rows):>8}")
    print()
    for name, key in (("sentence-start F1", "start"), ("sentences exact", "exact"),
                      ("tag agreement", "tags"), ("entity F1", "ent"),
                      ("UAS", "uas"), ("LAS", "las"),
                      ("UAS without punct", "uas_np"), ("LAS without punct", "las_np")):
        boot = sorted(score(key, idx) for idx in samples)
        print(f"{name:20} {score(key, range(n)):>7.2f}   (95% CI {boot[25]:.2f}-{boot[974]:.2f})")


if __name__ == "__main__":
    main()
