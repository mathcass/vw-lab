#!/usr/bin/env python
"""Add feature namespaces to VW sequence files written by conllu-to-vw.py.

Usage: featurize.py SPEC IN OUT

SPEC is a comma-separated list of feature groups. Each group becomes its own
namespace, named by the first letter (which is what VW's -q, --affix and
--search_neighbor_features refer to):

  w      the word as written                         |w Running
  l      lowercased word                             |l running
  sN     suffixes of length 1..N                     |s 1=g 2=ng 3=ing
  pN     prefixes of length 1..N                     |p 1=R 2=Ru
  h      word shape, long and collapsed              |h L=Xxxxxxx S=Xx
  f      orthographic flags and length bucket        |f cap len=7
"""

import re
import sys


def shape(word, collapse):
    out = []
    for ch in word:
        c = "X" if ch.isupper() else "x" if ch.islower() else "d" if ch.isdigit() else ch
        if not (collapse and out and out[-1] == c):
            out.append(c)
    return "".join(out)


def flags(word):
    f = []
    if word[:1].isupper():
        f.append("cap")
    if word.isupper() and any(c.isalpha() for c in word):
        f.append("allcap")
    if any(c.isdigit() for c in word):
        f.append("digit")
    if word.replace(".", "").replace(",", "").isdigit():
        f.append("number")
    if "-" in word:
        f.append("hyphen")
    if "." in word:
        f.append("period")
    if "@" in word:
        f.append("at")
    if re.match(r"(https?|www\.)", word, re.I) or re.search(r"\.(com|org|net|edu)\b", word, re.I):
        f.append("url")
    if not any(c.isalnum() for c in word):
        f.append("punct")
    f.append(f"len={min(len(word), 10)}")
    return f


def features(word, spec):
    # conllu-to-vw.py escapes VW syntax characters; undo that for the derived features.
    raw = word.replace("-COLON-", ":").replace("-PIPE-", "|")
    groups = []
    for g in spec:
        kind, n = g[0], int(g[1:] or 0)
        if kind == "w":
            feats = [word]
        elif kind == "l":
            feats = [raw.lower()]
        elif kind == "s":
            lw = raw.lower()
            feats = [f"{k}={lw[-k:]}" for k in range(1, n + 1) if len(lw) >= k]
        elif kind == "p":
            lw = raw.lower()
            feats = [f"{k}={lw[:k]}" for k in range(1, n + 1) if len(lw) >= k]
        elif kind == "h":
            feats = [f"L={shape(raw, False)}", f"S={shape(raw, True)}"]
        elif kind == "f":
            feats = flags(raw)
        else:
            raise SystemExit(f"unknown feature group {g!r}")
        # ':' and '|' are VW syntax; words already have them escaped, affixes may not.
        feats = [x.replace(":", "-COLON-").replace("|", "-PIPE-").replace(" ", "_") for x in feats]
        groups.append(f"|{kind} " + " ".join(feats))
    return " ".join(groups)


def main():
    spec = sys.argv[1].split(",")
    with open(sys.argv[2]) as src, open(sys.argv[3], "w") as out:
        for line in src:
            line = line.strip()
            if not line:
                out.write("\n")
                continue
            label, rest = line.split("|", 1)
            word = rest.split(" ", 1)[1]
            out.write(f"{label.strip()} {features(word, spec)}\n")


if __name__ == "__main__":
    main()
