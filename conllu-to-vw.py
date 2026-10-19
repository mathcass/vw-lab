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
1 |w Al |l al |s 1=l 2=al |p 1=a 2=al |h L=Xx S=Xx |f cap len=2 |n -1=<s> +1=-
2 |w - |l - |s 1=- |p 1=- |h L=- S=- |f hyphen punct len=1 |n -1=al +1=zaman
1 |w Zaman |l zaman |s 1=n 2=an 3=man 4=aman |p 1=z 2=za 3=zam |h L=Xxxxx S=Xx |f cap len=5 |n -1=- +1=-COLON-
...
29 |w . |l . |s 1=. |p 1=. |h L=. S=. |f period punct len=1 |n -1=border +1=</s>

```

The label (the number before the first "|") is the part-of-speech id from the
tag map, or with `--label ner`, the entity tag read from `NER=` in the MISC
column (as written by spacy-label.py). Entity tags use the numbering VW's
`sequencespan` search task expects: 1 = O, 2n = B-type, 2n+1 = I-type, over
spaCy's 18 OntoNotes entity types in a fixed order. A blank line separates sentences, so VW learns each sentence as one
sequence. Each feature group is its own namespace, named by its first letter,
which is how run.sh refers to them (`--search_neighbor_features`, `-q`):

  w  the word as written          s  suffixes of length 1-4 (lowercased)
  l  the word lowercased          p  prefixes of length 1-3 (lowercased)
  h  word shape: long (Xxxxx) and collapsed (Xx)
  f  flags: capitalized, all caps, digit, number, hyphen, period, @, URL,
     punctuation-only, and length (capped at 10)
  n  the lowercased words one either side (`-1=the +1=dog`, with <s> and
     </s> at sentence edges; `--context 2` adds -2 and +2). An ordinary
     namespace, so `-q ln` can cross it with the current word.

With `--label sent` (sentence splitting), each VW sequence is a whole
paragraph instead (paragraphs start at `# newpar` comments, as written by
spacy-label.py for .txt input, or at `# newdoc`; a file with neither has one
sentence per paragraph). The label is 2 (B) for a token that starts a
sentence and 1 (I) otherwise, and one more namespace says whether there is
whitespace around the token (from SpaceAfter=No in MISC):

  g  `b=0`/`b=1`: no space / space before;  `a=0`/`a=1`: the same after

More CoNLL-U files can be appended with `--add` (the NER training set is the
UD text plus WikiText), and `--shuffle SEED` shuffles the sentences of all of
them together before writing.

"""

import argparse
from array import array
from pathlib import Path
import random
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
parser.add_argument(
    "--label",
    choices=["xpos", "ner", "sent"],
    default="xpos",
    help="What to predict: the XPOS column (default), the NER= entity tag in "
    "MISC, in VW sequencespan numbering, or sentence starts (one sequence per "
    "paragraph).",
)
parser.add_argument(
    "--context",
    type=int,
    default=1,
    help="How many words either side go in the `n` namespace (default 1).",
)
parser.add_argument(
    "--add",
    type=Path,
    action="append",
    default=[],
    help="Another CoNLL-U file to read after INPUT_FILE (repeatable).",
)
parser.add_argument(
    "--shuffle",
    type=int,
    metavar="SEED",
    help="Shuffle the sentences (from all inputs) with this random seed.",
)
args = parser.parse_args()

# spaCy's English entity types (OntoNotes 5).
NER_TYPES = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
             "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY",
             "QUANTITY", "ORDINAL", "CARDINAL"]

input_file = args.input_file
output_file = args.output_file or sys.stdout
tags_file = args.output_file.with_suffix(".tags") if args.output_file else Path("conllu.tags")

POS_MAP = {}
if args.label == "ner":
    # A fixed map, so every split numbers labels the same way.
    POS_MAP["O"] = 1
    for n, t in enumerate(NER_TYPES, start=1):
        POS_MAP[f"B-{t}"], POS_MAP[f"I-{t}"] = 2 * n, 2 * n + 1
    frozen = True
elif args.label == "sent":
    POS_MAP.update(I=1, B=2)
    frozen = True
elif args.tags:
    with open(args.tags) as f:
        for line in f:
            tag, id_ = line.rstrip("\n").split("\t")
            POS_MAP[tag] = int(id_)
    frozen = True
else:
    frozen = False

# Each sentence is kept compactly as (label ids, words joined by newlines):
# the NER training set has 12M tokens, too many for a tuple per token.
sentences = []


def keep(sentence):
    if sentence:
        sentences.append((array("H", (t[0] for t in sentence)),
                          "\n".join(t[1] for t in sentence),
                          array("B", (t[2] for t in sentence)) if args.label == "sent" else None))


def has_paragraphs(path):
    with open(path) as f:
        return any(line.startswith(("# newpar", "# newdoc")) for line in f)


# From https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu
for path in [input_file, *args.add]:
    # For sentence splitting, `paragraph` collects sentences until the next
    # paragraph marker; otherwise each sentence is kept on its own.
    by_paragraph = args.label == "sent" and has_paragraphs(path)
    with open(path) as f:
      sentence, paragraph = [], []
      for line in f:
          if line.startswith("#"):
              if by_paragraph and line.startswith(("# newpar", "# newdoc")):
                  keep(paragraph)
                  paragraph = []
              # Skip comments
              continue

          if not line.strip():
              # At a blank line, complete the sentence and start a new one
              if args.label == "sent" and sentence:
                  sentence[0] = (POS_MAP["B"], *sentence[0][1:])
                  paragraph += sentence
                  if not by_paragraph:
                      keep(paragraph)
                      paragraph = []
              else:
                  keep(sentence)
              sentence = []
          else:
              row = line.rstrip("\n").split("\t")
              id_, word, pos = row[0], row[1], row[4]
              if args.label == "ner":
                  pos = next((m[4:] for m in row[9].split("|") if m.startswith("NER=")), None)
                  if pos is None:
                      sys.exit(f"{path}: token {id_} {word!r} has no NER= in MISC")
              if args.label == "sent":
                  pos = "I"
              if not id_.isdigit():
                  # Skip multiword tokens ("19-20 don't") and empty nodes
                  # ("8.1"). Their parts are already listed as regular tokens.
                  continue
              if frozen:
                  if pos not in POS_MAP:
                      sys.exit(f"{path}: tag {pos!r} is not in {args.tags}")
                  pos_id = POS_MAP[pos]
              else:
                  pos_id = POS_MAP.setdefault(pos, len(POS_MAP) + 1)
              space = 0 if "SpaceAfter=No" in row[9].split("|") else 1
              sentence.append((pos_id, word, space))
      if args.label == "sent":
          if sentence:
              sentence[0] = (POS_MAP["B"], *sentence[0][1:])
          keep(paragraph + sentence)
      else:
          keep(sentence)

if args.shuffle is not None:
    random.Random(args.shuffle).shuffle(sentences)


if not frozen or args.label in ("ner", "sent"):
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


def features(sentence, i, spaces=None):
    word = sentence[i].replace(" ", "_")
    lw = word.lower()

    def neighbor(j):
        if j < 0:
            return "<s>"
        if j >= len(sentence):
            return "</s>"
        return sentence[j].replace(" ", "_").lower()

    groups = {
        "w": [word],
        "l": [lw],
        "s": [f"{k}={lw[-k:]}" for k in range(1, 5) if len(lw) >= k],
        "p": [f"{k}={lw[:k]}" for k in range(1, 4) if len(lw) >= k],
        "h": [f"L={shape(word, False)}", f"S={shape(word, True)}"],
        "f": flags(word),
        "n": [f"{d:+d}={neighbor(i + d)}" for d in range(-args.context, args.context + 1) if d],
    }
    if spaces is not None:
        groups["g"] = [f"b={spaces[i - 1] if i else 1}", f"a={spaces[i]}"]
    return " ".join(f"|{ns} " + " ".join(escape(x) for x in feats) for ns, feats in groups.items())


def write_output(f):
    for labels, words, spaces in sentences:
        words = words.split("\n")
        for i, label in enumerate(labels):
            print(f"{label} {features(words, i, spaces)}", file=f)

        print("", file=f)


if isinstance(output_file, Path):
    with open(output_file, "w") as f:
        write_output(f)
else:
    write_output(output_file)
