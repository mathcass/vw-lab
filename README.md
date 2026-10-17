using vowpal wabbit to train a part of speech tagger.

## Usage

Dependencies are managed with [uv](https://docs.astral.sh/uv/). The
`vowpalwabbit` wheel doesn't ship a standalone `vw` binary, so the scripts call
its CLI as `uv run python -m vowpalwabbit` (same flags, but it can't read stdin).

```
make download   # fetch UD v2.7 into data/ (then untar ud-treebanks-v2.7.tgz)
make prepare    # convert UD_English-EWT to VW format in data/work/
make train      # train data/work/model.vw (PASSES=n ./run.sh to change passes)
make eval       # score dev and test, with a most-frequent-tag baseline
```

## Benchmark

Task: predict the Penn Treebank tag (XPOS column, 50 tags) for each word of
UD English-EWT v2.7, given gold tokenization. Train is 12,543 sentences /
204,585 words; dev and test are about 25k words each. Scored with `make eval`
(token accuracy, multiword-token lines like `don't` excluded). Measured with
vowpalwabbit 9.11.9 on 2026-10-16.

| Model | Dev | Test | Test, unseen words | Train time |
|---|---|---|---|---|
| Most-frequent-tag baseline | 83.92% | 83.72% | 22.3% | — |
| Original `run.sh`, 10 passes | 73.29% | 72.41% | 17.7% | ~7 min |
| Original `run.sh`, 100 passes | 73.39% | 72.80% | 19.6% | ~87 min |
| `model-2021.vw` (100 passes, scored with VW 8.9.0) | 74.16% | 73.93% | 17.2% | — |
| Original + `-b 24`, 5 passes | 86.48% | 86.75% | 33.3% | ~4 min |
| Original + `-b 24` + context/affix/shape features, 5 passes | 90.11% | 90.40% | 67.2% | ~6 min |
| **Current `run.sh` (hill-climbed), 3 passes** | **92.95%** | **93.01%** | **74.5%** | **~4.5 min** |

Test accuracy of the current config: 93.01% (95% CI 92.58-93.40, sentence
bootstrap); +2.61 points over the previous row (paired CI +2.27..+2.93) and
+20.2 over the original. It needs ~1.6 GB of RAM to train and the model file is
~770 MB. Unseen words are 8.3% of dev tokens. Train times are rough: other jobs
were often running on the same 4-core machine.

The current config was found by a hill-climb on dev, with test scored only
once at the end (records in `.claude/hillclimb/vw-tagger/`: `narrative.md` for
the story, `results.tsv` for every run). What it uses:

- Features from `conllu-to-vw.py`, one namespace each: the word (`w`),
  lowercased word (`l`), suffixes 1-4 (`s`), prefixes 1-3 (`p`), word shape
  (`h`) and orthographic flags (`f`).
- Neighbor features: words two either side, suffixes and shapes one either
  side.
- `-q 'w\x83'`: the current word crossed with the neighbor features (`\x83` is
  the hidden namespace VW stores neighbor features in). This lets a linear
  model learn pairs like "to" + verb -> TO vs "to" + noun -> IN.
- `-b 26`. Each bit from 24 to 26 added ~0.4-0.6 points on dev; `-b 25`
  (92.46% dev, ~420 MB model) is the option if size matters.
- 3 passes: as accurate as 5 (92.84%), and 10 passes overfit (92.55%).

Tried and rejected (dev, vs the best at the time): a 2-tag history
(-0.14 to -0.39), crossing suffixes with neighbors (-0.74), dropping the
`l` and `p` namespaces (-0.34), 10 passes (-0.29).

Notes:

- The original config underperforms the baseline because of hash collisions:
  VW's default 18-bit table is shared across all 50 tags. More passes don't
  help (training loss is flat after ~10 passes); a bigger table does.
- `model-2021.vw` was saved by VW 8.9.0. VW 9.x loads it without error but
  predicts differently (56% on dev), so score it with VW 8.x.

### Published results for comparison

Same treebank and tag set, but different UD releases (EWT is revised between
releases), so treat these as approximate.

| System | Kind of model | UD | XPOS, gold tokens | XPOS, raw text |
|---|---|---|---|---|
| This repo (`run.sh`) | VW linear tagger, hand-built features | 2.7 | 93.0 | — |
| UDPipe 1 | linear tagger, hand-built features | 2.5 | 93.9 | 92.8 |
| Stanza | BiLSTM | 2.5 | — | 95.1 |
| Trankit | XLM-R transformer | 2.5 | — | 95.7 |
| UDPipe 2 | BiLSTM + multilingual BERT embeddings | 2.6 | 97.0 | 96.1 |

"Raw text" scores include the system's own tokenization errors; this repo uses
gold tokens, so compare against the gold-token column where it exists.

UDPipe 2 is shown at UD 2.6, its release closest to the 2.7 data used here.
Its English-EWT XPOS score with gold tokens barely moves across later
releases: 97.03 (2.6), 97.06 (2.10), 97.17 (2.12, 2.15), 97.13 (2.17).

Sources: [UDPipe 1 models](https://ufal.mff.cuni.cz/udpipe/1/models),
[UDPipe 2 models](https://ufal.mff.cuni.cz/udpipe/2/models),
[Stanza paper](https://arxiv.org/abs/2003.07082),
[Trankit paper](https://arxiv.org/abs/2101.03289).

## Conclusion

The question behind this repo was whether VW could learn a POS tagger
efficiently, both in training cost and in model size. Half yes:

- Training is cheap: about 4.5 minutes on a 4-core CPU, no GPU.
- The model is not small. Accuracy kept rising only as the hash table grew
  (`-b 24` ~210 MB at 91.9% dev, `-b 26` ~770 MB at 93.0%), so the model buys
  accuracy by storing more word/context combinations, not by generalizing.

At 93% it is close to UDPipe 1's linear tagger (93.9%) but about 4 points
behind neural taggers. That means roughly twice as many errors. Most of the
remaining errors are unseen words, or words with a tag they never had in
training, and those need lexical knowledge from outside the treebank. Off-the-shelf
taggers such as spaCy, Stanza or UDPipe 2 are both more accurate and a fraction of
the size, so this stays an experiment rather than something to use.

Not tried, if anyone picks it up again: word clusters or embeddings from
unlabeled text (the only idea likely to gain more than a point), VW
learning-rate/regularization and search rollin/rollout settings, and word-pair
features computed in the converter to replace the large `-q 'w\x83'` cross.
