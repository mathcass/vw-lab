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
  nK     lowercased neighbor words within K positions |n -1=the +1=dog
         (an ordinary namespace, so --lrq and -q can use it)
  tK     POS tags of the word and its neighbors within K positions, from
         IN with the suffix .postags (one line of tags per sentence)
                                                     |t 0=NN -1=DT +1=VBZ
  c      distributional cluster ids of the lowercased word at three sizes,
         from the file in $HC_CLUSTERS (clusters.py)  |c 64=12 256=200 1k=901
"""

import os

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


CLUSTERS = None


def clusters():
    global CLUSTERS
    if CLUSTERS is None:
        CLUSTERS = {}
        with open(os.environ["HC_CLUSTERS"], encoding="utf-8") as f:
            for line in f:
                w, *ids = line.rstrip("\n").split("\t")
                CLUSTERS[w] = ids
    return CLUSTERS


def features(word, spec, sentence=None, i=None, tags=None):
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
        elif kind == "n":
            feats = []
            for d in range(-n, n + 1):
                if d == 0:
                    continue
                j = i + d
                nb = sentence[j] if 0 <= j < len(sentence) else ("<s>" if j < 0 else "</s>")
                nb = nb.replace("-COLON-", ":").replace("-PIPE-", "|").lower()
                feats.append(f"{d:+d}={nb}")
        elif kind == "t":
            feats = []
            for d in range(-n, n + 1):
                j = i + d
                t = tags[j] if 0 <= j < len(tags) else ("<s>" if j < 0 else "</s>")
                feats.append(f"{d:+d}={t}" if d else f"0={t}")
        elif kind == "c":
            ids = clusters().get(raw.lower())
            feats = ([f"64={ids[0]}", f"256={ids[1]}", f"1k={ids[2]}"] if ids
                     else ["unk"])
        else:
            raise SystemExit(f"unknown feature group {g!r}")
        # ':' and '|' are VW syntax; words already have them escaped, affixes may not.
        feats = [x.replace(":", "-COLON-").replace("|", "-PIPE-").replace(" ", "_") for x in feats]
        groups.append(f"|{kind} " + " ".join(feats))
    return " ".join(groups)


def main():
    spec = sys.argv[1].split(",")
    postags = None
    if any(g[0] == "t" for g in spec):
        from pathlib import Path
        postags = iter(Path(sys.argv[2]).with_suffix(".postags").read_text().splitlines())
    with open(sys.argv[2]) as src, open(sys.argv[3], "w") as out:
        sentence = []  # (label, word) pairs

        def flush():
            words = [w for _, w in sentence]
            tags = next(postags).split() if postags else None
            assert tags is None or len(tags) == len(words), (words, tags)
            for i, (label, word) in enumerate(sentence):
                out.write(f"{label} {features(word, spec, words, i, tags)}\n")
            out.write("\n")
            sentence.clear()

        for line in src:
            line = line.strip()
            if not line:
                if sentence:
                    flush()
                continue
            label, rest = line.split("|", 1)
            # "LABEL | word" or "LABEL |w word |l ..."; the word comes first.
            sentence.append((label.strip(), rest.split(" ")[1]))
        if sentence:
            flush()


if __name__ == "__main__":
    main()
