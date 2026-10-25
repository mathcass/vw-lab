#!/usr/bin/env python
"""Transform CoNLL-U into VW's dependency-parser format (`--search_task dep_parser`).

Usage:
    conllu-to-dep.py INPUT.conllu OUTPUT [--tags TAGS]

One line per token, a blank line between sentences:

    7,12 |w killed |p VBD

The label is `HEAD,REL`: the head's position in the sentence (1-based, 0 for
the root) and the relation's id from the relation map. The features are the
word as written (`w`) and its XPOS tag (`p`). VW's parser copies every
namespace of a token into one slot and builds its own pair and triple crosses
over the stack and buffer, so extra token features multiply quickly; in the
parser hill-climb, lowercase, suffix and shape features all made it worse.

Without --tags, a relation map is built from the input (in order of first
appearance) and written next to OUTPUT with a .tags suffix. Dev, test and new
text must reuse train's map. A relation missing from the map is written as
`dep`. For new text, HEAD and DEPREL can be `_`: the label is then `0,1`,
which `vw -t` ignores.
"""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("input_file", type=Path)
parser.add_argument("output_file", type=Path)
parser.add_argument("--tags", type=Path, help="existing relation map to reuse (train's)")
args = parser.parse_args()


def escape(feature):
    """':' and '|' are VW syntax and spaces separate features."""
    return feature.replace(":", "-COLON-").replace("|", "-PIPE-").replace(" ", "_")


sentences, sentence = [], []
with open(args.input_file, encoding="utf-8") as f:
    for line in f:
        if line.startswith("#"):
            continue
        if not line.strip():
            if sentence:
                sentences.append(sentence)
            sentence = []
            continue
        row = line.rstrip("\n").split("\t")
        if row[0].isdigit():  # skip multiword tokens and empty nodes
            sentence.append((row[1], row[4], row[6], row[7]))
if sentence:
    sentences.append(sentence)

if args.tags:
    rels = {k: int(v) for k, v in (l.split("\t") for l in args.tags.read_text().splitlines())}
else:
    rels = {}
    for s in sentences:
        for _, _, _, rel in s:
            rels.setdefault(rel, len(rels) + 1)
    args.output_file.with_suffix(".tags").write_text("".join(f"{k}\t{v}\n" for k, v in rels.items()))

with open(args.output_file, "w", encoding="utf-8") as f:
    for s in sentences:
        for word, tag, head, rel in s:
            label = "0,1" if head == "_" else f"{head},{rels.get(rel, rels['dep'])}"
            f.write(f"{label} |w {escape(word)} |p {escape(tag)}\n")
        f.write("\n")
