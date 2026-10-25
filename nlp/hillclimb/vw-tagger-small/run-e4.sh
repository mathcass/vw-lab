#!/usr/bin/env bash
# Label the extra 6M WikiText words, build the ~11M-token NER set, run round 4. One job at a time.
set -eu
S=/tmp/claude-1000/-home-cass-src-vw-universal-dependencies/c8522105-f977-49c4-8ec3-ce42492803f5/scratchpad
R=/home/cass/src/vw-universal-dependencies
H=$R/.claude/hillclimb/vw-tagger-small
cd $R
[ -s $S/wiki/wiki-10m-extra.conllu ] || { time uv run --group spacy python spacy-label.py $S/wiki/wiki-10m-extra.conllu $S/wiki/wiki-10m-extra.txt; }
uv run python conllu-to-vw.py --label ner $S/wiki/wiki-10m-extra.conllu $S/wiki/wiki-10m-extra.vw
mkdir -p data/spacy-ner-wiki10-work
python3 - <<PY
import random
def sents(p): return [b for b in open(p).read().split("\n\n") if b.strip()]
a = sents("data/spacy-ner-all-work/train.conllu") + sents("$S/wiki/wiki-4m.vw") + sents("$S/wiki/wiki-10m-extra.vw")
random.Random(0).shuffle(a)
open("data/spacy-ner-wiki10-work/train.conllu", "w").write("\n\n".join(a) + "\n\n")
print(len(a), "sentences,", sum(b.count("\n") + 1 for b in a), "tokens")
PY
cp data/spacy-ner-all-work/train.tags data/spacy-ner-wiki10-work/
for s in dev test; do ln -sf ../spacy-ner-work/$s.conllu data/spacy-ner-wiki10-work/$s.conllu; done
cd $H
HC_DATA=data/spacy-ner-wiki10-work ./round.sh e4.cfg 2>&1 | grep -v '^==' | awk -F'\t' '{print $1, "F1="$5, "P="$6, "R="$7, "tok="$8, "train_s="$9, "MB="$11}'
