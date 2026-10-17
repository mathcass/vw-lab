#!/usr/bin/env python
"""Transform CONLLU into Vowpal Wabbit sequence format.

CONLLU looks like,
```
# newdoc id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000
# sent_id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000-0001
# newpar id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000-p0001
# text = Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the mosque in the town of Qaim, near the Syrian border.
1       Al      Al      PROPN   NNP     Number=Sing     0       root    0:root  SpaceAfter=No
2       -       -       PUNCT   HYPH    _       1       punct   1:punct SpaceAfter=No
3       Zaman   Zaman   PROPN   NNP     Number=Sing     1       flat    1:flat  _
4       :       :       PUNCT   :       _       1       punct   1:punct _
5       American        american        ADJ     JJ      Degree=Pos      6       amod    6:amod  _
6       forces  force   NOUN    NNS     Number=Plur     7       nsubj   7:nsubj _
7       killed  kill    VERB    VBD     Mood=Ind|Tense=Past|VerbForm=Fin        1       parataxis       1:parataxis     _
8       Shaikh  Shaikh  PROPN   NNP     Number=Sing     7       obj     7:obj   _
9       Abdullah        Abdullah        PROPN   NNP     Number=Sing     8       flat    8:flat  _
10      al      al      PROPN   NNP     Number=Sing     8       flat    8:flat  SpaceAfter=No
11      -       -       PUNCT   HYPH    _       8       punct   8:punct SpaceAfter=No
12      Ani     Ani     PROPN   NNP     Number=Sing     8       flat    8:flat  SpaceAfter=No
13      ,       ,       PUNCT   ,       _       8       punct   8:punct _
14      the     the     DET     DT      Definite=Def|PronType=Art       15      det     15:det  _
15      preacher        preacher        NOUN    NN      Number=Sing     8       appos   8:appos _
16      at      at      ADP     IN      _       18      case    18:case _
17      the     the     DET     DT      Definite=Def|PronType=Art       18      det     18:det  _
18      mosque  mosque  NOUN    NN      Number=Sing     7       obl     7:obl:at        _
19      in      in      ADP     IN      _       21      case    21:case _
20      the     the     DET     DT      Definite=Def|PronType=Art       21      det     21:det  _
21      town    town    NOUN    NN      Number=Sing     18      nmod    18:nmod:in      _
22      of      of      ADP     IN      _       23      case    23:case _
23      Qaim    Qaim    PROPN   NNP     Number=Sing     21      nmod    21:nmod:of      SpaceAfter=No
24      ,       ,       PUNCT   ,       _       21      punct   21:punct        _
25      near    near    ADP     IN      _       28      case    28:case _
26      the     the     DET     DT      Definite=Def|PronType=Art       28      det     28:det  _
27      Syrian  syrian  ADJ     JJ      Degree=Pos      28      amod    28:amod _
28      border  border  NOUN    NN      Number=Sing     21      nmod    21:nmod:near    SpaceAfter=No
29      .       .       PUNCT   .       _       1       punct   1:punct _

```

(Note the newline at the end, and newlines will separate records.)

The format for Vowpal Wabbit looks like this (one line per word, shortened):
```
1 |w Al |l al |s 1=l 2=al |p 1=a 2=al |h L=Xx S=Xx |f cap len=2
2 |w - |l - |s 1=- |p 1=- |h L=- S=- |f hyphen punct len=1
1 |w Zaman |l zaman |s 1=n 2=an 3=man 4=aman |p 1=z 2=za 3=zam |h L=Xxxxx S=Xx |f cap len=5
...
29 |w . |l . |s 1=. |p 1=. |h L=. S=. |f period punct len=1

```

The label (the number before the first "|") is the part-of-speech id from the
tag map. A blank line separates sentences, so VW learns each sentence as one
sequence. Each feature group is its own namespace, named by its first letter,
which is how run.sh refers to them (`--search_neighbor_features`, `-q`):

  w  the word as written          s  suffixes of length 1-4 (lowercased)
  l  the word lowercased          p  prefixes of length 1-3 (lowercased)
  h  word shape: long (Xxxxx) and collapsed (Xx)
  f  flags: capitalized, all caps, digit, number, hyphen, period, @, URL,
     punctuation-only, and length (capped at 10)

"""

import argparse
from pathlib import Path
import re
import sys

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("input_file", type=Path)
parser.add_argument("output_file", type=Path, nargs="?")
parser.add_argument(
    "--tags",
    type=Path,
    help="Existing tag map to reuse (e.g. the one written for train). Without "
    "it, a new map is built from the input. Dev and test must reuse train's "
    "map, or their label numbers will not match the model's.",
)
args = parser.parse_args()

input_file = args.input_file
output_file = args.output_file or sys.stdout
tags_file = args.output_file.with_suffix(".tags") if args.output_file else Path("conllu.tags")

POS_MAP = {}
if args.tags:
    with open(args.tags) as f:
        for line in f:
            tag, id_ = line.rstrip("\n").split("\t")
            POS_MAP[tag] = int(id_)
    frozen = True
else:
    frozen = False

sentences = []

# From https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu
with open(input_file) as f:
    sentence = []
    for line in f:
        if line.startswith("#"):
            # Skip comments
            continue

        if not line.strip():
            # At a blank line, complete the sentence and start a new one
            if sentence:
                sentences.append(sentence)
            sentence = []
        else:
            row = line.rstrip("\n").split("\t")
            id_, word, pos = row[0], row[1], row[4]
            if not id_.isdigit():
                # Skip multiword tokens ("19-20 don't") and empty nodes
                # ("8.1"). Their parts are already listed as regular tokens.
                continue
            if frozen:
                if pos not in POS_MAP:
                    sys.exit(f"{input_file}: tag {pos!r} is not in {args.tags}")
                pos_id = POS_MAP[pos]
            else:
                pos_id = POS_MAP.setdefault(pos, len(POS_MAP) + 1)
            sentence.append((pos_id, word))
    if sentence:
        sentences.append(sentence)


if not frozen:
    with open(tags_file, "w") as f:
        for k, v in POS_MAP.items():
            print(f"{k}\t{v}", file=f)


def escape(feature):
    """':' and '|' are VW syntax and spaces separate features."""
    return feature.replace(":", "-COLON-").replace("|", "-PIPE-").replace(" ", "_")


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


def features(token):
    word = token.replace(" ", "_")
    lw = word.lower()
    groups = {
        "w": [word],
        "l": [lw],
        "s": [f"{k}={lw[-k:]}" for k in range(1, 5) if len(lw) >= k],
        "p": [f"{k}={lw[:k]}" for k in range(1, 4) if len(lw) >= k],
        "h": [f"L={shape(word, False)}", f"S={shape(word, True)}"],
        "f": flags(word),
    }
    return " ".join(f"|{ns} " + " ".join(escape(x) for x in feats) for ns, feats in groups.items())


def write_output(f):
    for sentence in sentences:
        for label, token in sentence:
            print(f"{label} {features(token)}", file=f)

        print("", file=f)


if isinstance(output_file, Path):
    with open(output_file, "w") as f:
        write_output(f)
else:
    write_output(output_file)
