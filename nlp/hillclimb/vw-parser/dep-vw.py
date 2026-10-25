#!/usr/bin/env python
"""CoNLL-U -> VW dep_parser format (hill-climb prototype).

Usage: dep-vw.py IN.conllu OUT.vw --rels RELS.tsv [--build] [--feats w,p]
                 [--pos-pred PRED --pos-tags TAGS]

Each token line is `HEAD,REL |w word |p tag ...` (HEAD 1-based, 0 = root;
REL a 1-based id from RELS.tsv, built from the input with --build). VW's
parser copies every namespace of a token into one slot and crosses slots
itself, so token features are kept few. Feature groups:
  w word   l lowercase   p POS tag   s3 suffix (3)   h collapsed shape
The tag is the XPOS column, or with --pos-pred a VW tagger's predictions
(one line per sentence of label ids) mapped through --pos-tags.
"""
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("inp", type=Path); ap.add_argument("out", type=Path)
ap.add_argument("--rels", type=Path, required=True)
ap.add_argument("--build", action="store_true")
ap.add_argument("--feats", default="w,p")
ap.add_argument("--pos-pred", type=Path); ap.add_argument("--pos-tags", type=Path)
a = ap.parse_args()


def esc(x):
    return x.replace(":", "-COLON-").replace("|", "-PIPE-").replace(" ", "_")


def shape(w):
    out = []
    for ch in w:
        c = "X" if ch.isupper() else "x" if ch.islower() else "d" if ch.isdigit() else ch
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


sents, cur = [], []
for line in open(a.inp, encoding="utf-8"):
    if line.startswith("#"):
        continue
    if not line.strip():
        if cur: sents.append(cur); cur = []
        continue
    r = line.rstrip("\n").split("\t")
    if r[0].isdigit():
        cur.append((r[1], r[4], int(r[6]), r[7]))
if cur: sents.append(cur)

if a.build:
    rels = {}
    for s in sents:
        for t in s:
            rels.setdefault(t[3], len(rels) + 1)
    a.rels.write_text("".join(f"{k}\t{v}\n" for k, v in rels.items()))
rels = dict((k, int(v)) for k, v in (l.split("\t") for l in a.rels.read_text().splitlines()))

if a.pos_pred:
    id2tag = {int(v): k for k, v in (l.split("\t") for l in a.pos_tags.read_text().splitlines())}
    preds = [[id2tag[int(float(x))] for x in l.split()] for l in open(a.pos_pred) if l.strip()]
    assert len(preds) == len(sents)
else:
    preds = [[t[1] for t in s] for s in sents]

groups = a.feats.split(",")
with open(a.out, "w", encoding="utf-8") as f:
    for s, tags in zip(sents, preds):
        for (word, _, head, rel), tag in zip(s, tags):
            lw = word.lower(); parts = []
            for g in groups:
                if g == "w": parts.append(f"|w {esc(word)}")
                elif g == "l": parts.append(f"|l {esc(lw)}")
                elif g == "p": parts.append(f"|p {esc(tag)}")
                elif g.startswith("s"): parts.append(f"|s {esc(lw[-int(g[1:]):])}")
                elif g == "h": parts.append(f"|h {esc(shape(word))}")
                else: raise SystemExit(f"unknown group {g}")
            f.write(f"{head},{rels.get(rel, rels['dep'])} " + " ".join(parts) + "\n")
        f.write("\n")
print(f"{a.out}: {len(sents)} sentences, {len(rels)} relations", flush=True)
