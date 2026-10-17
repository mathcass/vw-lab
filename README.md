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
| `run.sh` as committed, 10 passes | 73.29% | 72.41% | 17.7% | ~7 min |
| `run.sh` as committed, 100 passes | 73.39% | 72.80% | 19.6% | ~87 min |
| `model-2021.vw` (100 passes, scored with VW 8.9.0) | 74.16% | 73.93% | 17.2% | — |
| `run.sh` + `-b 24`, 5 passes | 86.48% | 86.75% | 33.3% | ~4 min |
| `run.sh` + `-b 24` + context/affix/shape features, 5 passes | 90.11% | 90.40% | 67.2% | ~6 min |

Unseen words are 8.3% of dev tokens. Train times are rough: other jobs were
running on the same 4-core machine.

The last row puts words in a `w` namespace (`1 |w Al`) and adds
`--search_neighbor_features -1:w,1:w,-2:w,2:w --affix -3w,-2w,+2w --spelling w`.

Notes:

- The committed config underperforms the baseline because of hash collisions:
  VW's default 18-bit table is shared across all 50 tags. More passes don't
  help (training loss is flat after ~10 passes); `-b 24` does.
- `model-2021.vw` was saved by VW 8.9.0. VW 9.x loads it without error but
  predicts differently (56% on dev), so score it with VW 8.x.

### Published results for comparison

Same treebank and tag set, but different UD releases (EWT is revised between
releases), so treat these as approximate.

| System | Kind of model | UD | XPOS, gold tokens | XPOS, raw text |
|---|---|---|---|---|
| UDPipe 1 | linear tagger, hand-built features | 2.5 | 93.9 | 92.8 |
| Stanza | BiLSTM | 2.5 | — | 95.1 |
| Trankit | XLM-R transformer | 2.5 | — | 95.7 |
| UDPipe 2 | BiLSTM + BERT-style embeddings | 2.17 | 97.1 | 96.3 |

"Raw text" scores include the system's own tokenization errors; this repo uses
gold tokens, so compare against the gold-token column where it exists.

Sources: [UDPipe 1 models](https://ufal.mff.cuni.cz/udpipe/1/models),
[UDPipe 2 models](https://ufal.mff.cuni.cz/udpipe/2/models),
[Stanza paper](https://arxiv.org/abs/2003.07082),
[Trankit paper](https://arxiv.org/abs/2101.03289).
