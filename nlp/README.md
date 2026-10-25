# nlp: VW taggers, entities and parses

Vowpal Wabbit models for English text: a part-of-speech tagger trained on UD
English-EWT, and VW copies of spaCy's tagger, sentence splitter, entity
tagger and dependency parser, chained into a pipeline that runs on raw text.

## Usage

Run everything from this folder (or `make -C nlp TARGET` from the repo root).
Dependencies are managed with [uv](https://docs.astral.sh/uv/); this folder is
its own uv project. The `vowpalwabbit` wheel doesn't ship a standalone `vw`
binary, so the scripts call its CLI as `uv run python -m vowpalwabbit` (same
flags, but it can't read stdin).

Paths like `data/work/` below are in the repo's shared `data/` folder
(`../data` from here).

```
make download   # fetch UD v2.7 into data/ (checksummed; the other targets do this when needed)
make prepare    # convert UD_English-EWT to VW format in data/work/
make train      # train data/work/model.vw (~8.4 MB; SIZE=small for ~0.4 MB)
make eval       # score dev and test, with a most-frequent-tag baseline
```

The named-entity model (spaCy's entities, distilled into VW; see
[Matching spaCy](#matching-spacy)) has its own targets. They need the UD
download above; `ner-prepare` also fetches a C4 web-text shard and labels
part of it with spaCy:

```
make ner-prepare  # label UD text + 4M words of C4 with spaCy, build data/ner-work/ (~9 min + 319 MB download)
make ner-train    # train data/ner-work/model.vw (~9.3 MB, ~14 min)
make ner-eval     # entity F1 vs spaCy on dev and test, with a 95% CI and per-type F1
```

The other spaCy-copying models, and the pipeline that chains them on raw text
(see [End to end](#end-to-end)):

```
make spacy-pos-prepare spacy-pos-train spacy-pos-eval  # tagger on spaCy's tags (~8.3 MB, ~1 min)
make sent-prepare sent-train sent-eval                 # sentence splitter (~0.6 MB, ~30 s; prepare ~2 min)
make dep-prepare dep-train dep-eval                    # dependency parser (~4.9 MB, ~12 min)
make pipeline-eval  # raw EWT dev/test paragraphs through all four models, scored against spaCy
make c4-eval        # the same on 1M words of held-out C4, timed against spaCy (~8 min)
uv run --group spacy python pipeline.py IN.txt OUT.conllu  # any text, one paragraph per line
```

Word embeddings trained with VW are a separate project in
[`../word2vec/`](../word2vec/README.md).

## Data

Nothing under `data/` is committed; `make download` fetches the UD release and
the commands below rebuild the rest. The C4 and WikiText downloads are shared
with `word2vec/`, so they live in the [root Makefile](../Makefile), which this
folder's Makefile calls when it needs them.

### Universal Dependencies v2.7, English

Human-annotated treebanks from the [UD v2.7 release](https://universaldependencies.org/).
Word counts exclude multiword-token lines.

| Treebank | Text | Train | Dev | Test | License | Used for |
|---|---|---|---|---|---|---|
| EWT | web text: blogs, reviews, email, Q&A | 204,585 | 25,148 | 25,096 | CC BY-SA 4.0 | gold POS training and scoring; its text is also spaCy-labeled |
| GUM | Wikipedia, interviews, news, how-tos, fiction | 81,861 | 15,598 | 15,926 | CC BY-NC-SA 4.0 | spaCy-labeled extra text (all splits) |
| LinES | translated fiction and manuals | 57,372 | 19,170 | 17,675 | CC BY-NC-SA 4.0 | spaCy-labeled extra text (all splits) |
| ParTUT | legal, Wikipedia, web text | 43,504 | 2,722 | 3,408 | CC BY-NC-SA 4.0 | spaCy-labeled extra text (all splits) |
| PUD | news and Wikipedia (test only) | — | — | 21,176 | CC BY-SA 3.0 | spaCy-labeled extra text |
| ESL, GUMReddit | learner essays, Reddit | 78,541 / 10,831 | | | CC BY-SA 4.0 / CC BY 4.0 | not used: the release ships the text masked as `_` |
| Pronouns | constructed pronoun test sentences | — | — | 1,695 | CC BY-SA 4.0 | not used |

`make prepare` converts EWT to VW format in `data/work/` (`train`, `dev`,
`test` and the tag map `train.tags`). Every POS number in the benchmark
tables uses this data: the treebank's own tokens and tags.

### spaCy-labeled data

To measure how closely VW can copy spaCy, `spacy-label.py` runs
`en_core_web_sm` 3.8.0 over raw sentence text and writes spaCy's output as
CoNLL-U (spaCy's tokens, tags, lemmas, morphology, dependency parse, and
entities as `NER=B-ORG` in MISC). Only the `# text` lines of the input are
read; the treebanks' own annotation is ignored. These labels are spaCy's
predictions, not gold, and the scores trained on them are agreement with spaCy.

| File | Text | Tokens | Used for |
|---|---|---|---|
| `data/spacy/train-ewt.conllu` | EWT train | 206,546 | POS and NER training |
| `data/spacy/train.conllu` | EWT train + GUM + LinES + ParTUT + PUD | 487,819 | POS and NER training ("all text") |
| `data/spacy/dev.conllu` | EWT dev | 25,686 | dev agreement for every spaCy-labeled run |
| `data/spacy/test.conllu` | EWT test | 25,530 | test agreement, scored once per final model |
| `data/spacy/train-par.conllu` | all text, as 5,103 paragraphs (25,766 spaCy sentences) | 487,819 | sentence-splitter training |
| `data/spacy/{dev,test}-par.conllu` | EWT dev / test, as 750 / 854 paragraphs | 25,686 / 25,530 | sentence-splitter and end-to-end scoring |

spaCy splits sentences and tokens its own way, so these files have slightly
different sentence and token counts from the treebank. The `-par` files are
labeled from whole paragraphs instead of single sentences, so spaCy's own
sentence splits can be learned: `ud-paragraphs.py` joins each paragraph's
sentences (EWT marks paragraphs; GUM, LinES and PUD only documents, which
are used whole; ParTUT has neither, so each sentence stands alone) into a
.txt file with one paragraph per line, and `spacy-label.py` marks each
paragraph's first sentence with `# newpar`.

The VW-format versions, built with `conllu-to-vw.py` (`--label ner` for
entities, which numbers spaCy's 18 entity types the way VW's sequence tasks
expect; `--label sent` for sentence starts) and `conllu-to-dep.py` (parses):

| Folder | Labels | Training text |
|---|---|---|
| `data/spacy-pos-work/` | spaCy XPOS tags (`make spacy-pos-prepare`) | EWT train |
| `data/spacy-work/` | spaCy XPOS tags | all text |
| `data/spacy-ner-work/` | spaCy entities (BIO, 37 labels) | EWT train |
| `data/spacy-ner-all-work/` | spaCy entities (BIO, 37 labels) | all text |
| `data/sent-work/` | sentence starts, one sequence per paragraph (`make sent-prepare`) | all text, as paragraphs |
| `data/dep-work/` | spaCy heads and relations (45), spaCy's tags as input (`make dep-prepare`) | all text |

Because GUM, LinES and ParTUT are non-commercial (CC BY-NC-SA), anything
trained on the "all text" sets inherits that restriction; the EWT-only sets
are CC BY-SA.

### C4

The final NER model's extra training text is web text from
[C4](https://huggingface.co/datasets/allenai/c4) (`en` config; Common Crawl
pages, ODC-BY), labeled by spaCy the same way. Web text is closer to EWT's
genre than Wikipedia, and it beat the same amount of WikiText by 2 F1.
`c4-to-text.py` takes documents in file order from the first training shard
(already in random order) and writes each line of each document as a `# text`
line, stopping at a word count, so a larger count extends a smaller sample.

`make ner-prepare` builds the NER data the final model uses (8.5 min on a
4-core CPU after the download, almost all of it spaCy labeling; the data
comes out byte-identical to the hill-climb's):

| File or folder | Contents | Sentences | Tokens |
|---|---|---|---|
| `data/c4/c4-train-00000.json.gz` | C4 en, first of 1,024 training shards (319 MB, 356,317 documents, pinned revision and checksum) | — | — |
| `data/c4/c4-4m.txt` | first 11,043 documents, 93,785 lines | — | 4.0M words |
| `data/c4/c4-4m.conllu` | the sample labeled by spaCy | 230,540 | — |
| `data/ner-work/train.conllu` | all UD text + the sample, VW format, shuffled (seed 0) | 258,389 | 5,121,897 |
| `data/ner-work/{dev,test}.conllu` | EWT dev/test, VW format | 2,063 / 2,124 | 25,686 / 25,530 |

The NER training set includes the "all text" UD sentences, so it inherits the
non-commercial restriction.

### WikiText-103

The earlier NER models used paragraphs from
[WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) (raw
version; Wikipedia articles, CC BY-SA 4.0, ~100M words) instead.
`wikitext-to-text.py` (at the repo root, since `word2vec/` uses the same
sample) skips headings, undoes WikiText's tokenization spacing
(`role @-@ playing` -> `role-playing`, `word ,` -> `word,`) and writes a
seeded random sample of paragraphs as `# text` lines, all from the first of
the two training shards (~50M words). The root Makefile builds the sample and
this folder's Makefile labels it:

| File | Contents | Sentences | Tokens |
|---|---|---|---|
| `data/wikitext/train-00000.parquet` | WikiText-103 raw, first training shard (157 MB, pinned revision and checksum) | — | ~50M words |
| `data/wikitext/wiki-10m.txt` | seed-0 sample, 100,179 paragraphs (17 min to label) | — | 10.0M words |
| `data/wikitext/wiki-10m.conllu` | the sample labeled by spaCy | 444,145 | — |

With all UD text, the 10M-word sample makes 471,994 training sentences
(11,973,329 tokens); the 4M-word runs used its first 4M words (5.08M tokens).

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
| First hill-climb (`-b 26 -q 'w\x83'`), 3 passes, ~770 MB | 92.95% | 93.01% | 74.5% | ~4.5 min |
| **Current `run.sh`** (teacher-forced, 8.4 MB) | **93.75%** | **93.50%** | **76.2%** | **40 s** |
| Current `run.sh`, `SIZE=small` (0.40 MB) | 91.18% | 90.92% | 73.7% | 33 s |

Test accuracy of the current config: 93.50% (95% CI 93.09-93.90, sentence
bootstrap), +20.7 points over the original. Unseen words are 8.3% of dev
tokens. It came out of two hill-climbs, described below and in "Small model".
Train times before the small-model climb are rough: other jobs were often
running on the same 4-core machine.

The first hill-climb (records in [`hillclimb/vw-tagger/`](hillclimb/README.md):
`narrative.md` for the story, `results.tsv` for every run) found the
features; the second (`hillclimb/vw-tagger-small/`) found the training settings and made
the model small. The first climb's config, now replaced, used:

- Features from `conllu-to-vw.py`, one namespace each: the word (`w`),
  lowercased word (`l`), suffixes 1-4 (`s`), prefixes 1-3 (`p`), word shape
  (`h`) and orthographic flags (`f`). The converter now also writes `n`
  (the lowercased words one either side), which the current config uses.
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

### Small model

A second hill-climb looked for the smallest model that stays at or above 90%
dev (records in `hillclimb/vw-tagger-small/`). spaCy's small English
pipeline, about 12 MB, was the size to beat. Train times are for one job at a
time on the same 4-core machine.

| Model | Dev | Test (95% CI) | Test, unseen words | Model size | Train time |
|---|---|---|---|---|---|
| First hill-climb | 92.95% | 93.01% (92.58-93.40) | 74.5% | ~770 MB | ~4.5 min |
| Small, default search settings (`--l1 4e-8`) | 90.71% | 90.72% (90.27-91.16) | 73.6% | 1.7 MB | 233 s |
| **Teacher-forced, `--l1 3e-7`** (`run.sh`) | **93.75%** | **93.50% (93.09-93.90)** | **76.2%** | **8.4 MB** | **37 s** |
| **Teacher-forced, `--l1 7e-7`** (`SIZE=small`) | **91.18%** | **90.92% (90.47-91.39)** | **73.7%** | **0.40 MB** | **33 s** |

The two teacher-forced rows are the results. The 8.4 MB model is more
accurate than the 770 MB first-climb model and trains 7x faster; the 0.40 MB
model is the smallest that stays above 90% (`--l1 8.5e-7` gives 88.7% at
0.17 MB). Both are now `run.sh` (`make train`, and `SIZE=small make train`),
which reproduces these numbers exactly. Their flags, on top of the usual
search flags and neighbor features:

```
--search_rollin ref --search_rollout none -b 26 -q ln --l1 3e-7 --predict_only_model
```

with 3 passes and all the namespaces `conllu-to-vw.py` writes (the hill-climb's
`featurize.py` spec `w,l,s4,p3,h,f,n1` gives byte-identical files).

What made it small and fast:


- **Teacher forcing.** VW tags left to right, and each decision can look at the
  tags it already chose. By default VW trains on its own earlier guesses and
  runs rollouts, trying alternative tags and playing out the rest of the
  sentence to measure their cost. `--search_rollin ref --search_rollout none`
  instead trains each step on the *correct* earlier tags, with no rollouts. For
  example, in "I want to go" (gold `PRP VBP TO VB`), the default setting might
  learn the tag for "to" after its own mistaken `NN` for "want". Teacher forcing
  always learns it after the gold `VBP`. That gives one clean training example
  per word instead of many noisy ones. In theory the model gets less practice
  recovering from its own mistakes, but here it was +2.3 points on POS at the
  same size (1.7 MB: 92.99% vs 90.71% dev), +7.4 F1 on NER, and 7–10× faster to
  train.
- `--predict_only_model`. By default VW 9 saves the optimizer's training
  state with the model, so the file is about 12.5 bytes per hash slot whatever
  the weights are. With this flag the file halves at identical accuracy
  (52.4 MB -> 26.2 MB at `-b 22`) and only nonzero weights cost anything.
- L1 regularization on a big table beats a small table. Shrinking `-b` makes
  unrelated features share weights; `-b 26` with `--l1` keeps collisions rare
  and drops the weights that don't matter. VW applies `--l1` on every update,
  so its scale depends on how many updates training makes: teacher forcing
  makes far fewer than rollouts, so it needs ~10x the L1 for the same size.
  Teacher-forced L1 curve (dev / MB): 2e-7 93.72/13.9, 3e-7 93.75/8.4,
  5e-7 92.99/1.7, 7e-7 91.18/0.40, 8.5e-7 88.68/0.17, 1e-6 85.41/0.09.
- A narrower cross. `-q ln` (lowercased word x the words one either side)
  replaces `-q 'w\x83'`: slightly more accurate and a third smaller at the
  same L1. Dropping the cross entirely loses 1.3 points.

Tried and rejected for size: `--lrq ln4` (low-rank cross; 80.3% dev at 21 MB),
`--nn 8` (with L1 the network is zeroed out; without L1 at `-b 22` it is 1.2
points worse than the linear model at the same size and trains 7x slower),
and smaller tables without L1 (`-b 20`: 87.2%, `-b 21`: 89.3%).

### Matching spaCy

To see how close VW can get to replacing spaCy, `spacy-label.py` runs
`en_core_web_sm` 3.8.0 over the raw sentence text of the UD English treebanks
and writes its predictions as CoNLL-U (tokens, tags, lemmas, morphology,
dependency parse, and entities in MISC). VW is then trained on spaCy's output
instead of the treebank's labels, and scored by agreement with spaCy on the
EWT dev/test text.

**POS tags.** Trained on spaCy's tags for EWT train text (207k tokens):

| Model | Dev agreement | Test agreement (95% CI) | Model size | Train time |
|---|---|---|---|---|
| Small, default search settings | 90.87% | 91.33% (90.58-91.97) | 1.9 MB | 237 s |
| Teacher-forced, `--l1 3e-7` | 94.33% | **94.53% (94.18-94.88)** | 8.3 MB | 34 s |
| Teacher-forced, `--l1 7e-7` | 91.07% | **91.83% (91.08-92.47)** | 0.40 MB | 33 s |

Adding the other English treebanks' text (488k tokens in all) didn't help the
small model, which is limited by size, not data.

**Named entities.** Trained on spaCy's entities (18 OntoNotes types, BIO
tags), scored by entity F1 against spaCy (exact span and type; test has 1,876
spaCy entities):

| Model | Training text | Dev F1 | Test F1 (95% CI) | Model size | Train time |
|---|---|---|---|---|---|
| Default search settings, `-b 24` | EWT (207k tokens) | 55.19 | — | 48.9 MB | 115 s |
| Teacher-forced, `-b 24 -q ln` | all UD (488k) | 66.82 | 63.86 (61.55-66.20) | 27.3 MB | 34 s |
| Teacher-forced, `-b 26 -q ln --l1 2e-7` | all UD (488k) | 66.49 | 66.20 (63.91-68.40) | 10.2 MB | 54 s |
| + words ±2 in `n`, `--l1 5e-8` | + 4M words WikiText (5.1M) | 72.92 | — | 10.0 MB | 12.5 min |
| + words ±2 in `n`, `--l1 2.7e-8` | + 10M words WikiText (12.0M) | 73.79 | 69.27 (66.96-71.39) | 9.7 MB | 28 min |
| + words ±2 in `n`, `--l1 5e-8` | + 4M words C4 (5.1M) | 74.92 | — | 15.9 MB | 13 min |
| + words ±2 in `n`, `--l1 2.5e-8` | + 4M C4 + 4M WikiText (9.9M) | 75.53 | — | 21.0 MB | 25 min |
| **+ words ±2 in `n`, `--l1 6e-8`** | **+ 4M words C4 (5.1M)** | **74.41** | **71.84 (69.70-73.79)** | **9.3 MB** | **14 min** |

The last row is the result of an NER hill-climb (target: F1 vs spaCy at
≤12 MB, about 5.8 h of compute in all, one training job at a time). Full
flags:

```
--search_task sequence --search 37 --search_rollin ref --search_rollout none
--search_neighbor_features -1:w,1:w,-2:w,2:w,-1:s,1:s,-1:h,1:h
-q ln -b 26 --l1 6e-8 --predict_only_model --passes 3
```

with the converter's features plus lowercase words two to the left and right
in the `n` namespace (`conllu-to-vw.py --context 2`).

- **More spaCy-labeled text was the biggest lever.** 4M words of WikiText
  gave +6.4 dev F1 at the same size; 6M more gave +1.5 more. Training time
  grows with it: 34 s for 488k tokens to 28 min for 12M. L1 has to shrink as
  the data grows (VW applies it on every update) to keep the size steady.
- **The genre of that text matters too.** 4M words of C4 web text beat 4M of
  WikiText by +2.0 dev F1 at the same flags (paired bootstrap 95% CI +0.2 to
  +3.9) and beat 10M of WikiText on test by +2.6 (+0.9 to +4.1), at a smaller
  size and half the training time. C4 helps the common types (ORG, DATE,
  CARDINAL, MONEY, TIME); WikiText was better on rare ones (LOC, FAC,
  WORK_OF_ART). The dev-test gap shrank from 4.5 to 2.6 F1.
- **Mixing C4 and WikiText** is the most accurate (75.5 dev) but needs 21 MB;
  pushed under 12 MB with L1, it's no better than C4 alone.
- **Didn't help:** the VW tagger's POS tags as NER features (+0.1, within
  noise; the suffix, shape and lowercase features already carry it),
  distributional word clusters from 24M words of text (64/256/1024 clusters
  from PPMI + SVD + k-means; within noise, slightly worse), `-q sn`, a 2-tag
  history, longer quad n-grams.

NER is much harder to copy than tags. VW fits its training data well (F1 86
on train) but generalizes poorly. Errors are mostly entity boundaries and
spaCy's own inconsistencies (e.g. "Holocaust" as WORK_OF_ART). The same
features, with teacher forcing and `--search_task sequence --search 37`, give
the best results. VW's dedicated `sequencespan` task did worse (F1 38.6: it
predicts O for most entity starts), and its `--search_span_bilou` output is
broken in VW 9.x (labels come back in the internal BILOU numbering).

**Dependency parses.** VW's `dep_parser` search task (arc-hybrid,
transition-based) trained on spaCy's parses of all UD text (488k tokens),
with each word's form and POS tag as features. Scored by agreement with
spaCy's heads (UAS) and heads plus labels (LAS, 45 spaCy labels), over all
tokens.

VW 9.x's parser leaves the last token of most sentences without a head (it
outputs `9999999:9999999`; 1,774 of 2,124 test sentences). That token is
nearly always the final punctuation, which spaCy attaches to the root, so
`evaluate.py` and `pipeline.py` attach it to the sentence's root (as `punct`
if it has no letters or digits, else `dep`). That alone is worth +5 points:

| Model (`make dep-train`) | Dev UAS / LAS | Test UAS (95% CI) | Test LAS (95% CI) | Model size | Train time |
|---|---|---|---|---|---|
| **Teacher-forced, `-b 26 --l1 5e-8`, spaCy's tags** | **85.41 / 81.47** | **85.73 (84.94-86.46)** | **82.55 (81.71-83.33)** | **4.9 MB** | **12 min** |
| same, the VW tagger's tags as input | 82.91 / 77.89 | 83.16 (82.33-83.98) | 78.95 (77.97-79.86) | 4.9 MB (+ 8.3 MB tagger) | — |
| same, spaCy's tags, VW's output as is | 79.99 / 76.46 | 80.22 (79.48-80.88) | 77.64 (76.89-78.33) | 4.9 MB | 12 min |

The spaCy-tags rows are an upper bound; the VW-tags row is what a VW-only
pipeline gets (the tagger agrees with spaCy's tags 94.5% of the time, and the
parser loses ~3.6 LAS to the difference). Without punctuation the model
scores 86.7 UAS / 83.3 LAS on test with spaCy's tags, 84.4 / 79.8 with VW's.

The model came from a parser hill-climb (~4.6 h of compute, one job at a
time), which scored VW's output as is, before the fix was found. The fix
moves every configuration by the same structural amount, so the ranking
holds:

| Model, VW's output as is | Dev UAS / LAS | Test UAS | Test LAS | Model size | Train time |
|---|---|---|---|---|---|
| VW's default search settings, `-b 24`, 1 pass | 72.77 / 69.08 | — | — | 97.5 MB | 8 min |
| Teacher-forced, `-b 24` | 77.47 / 74.74 | — | — | 94.2 MB | 6 min |
| Teacher-forced, `-b 26 --l1 3e-8` | 80.09 / 77.32 | 80.12 | 77.77 | 15.8 MB | 13 min |
| **Teacher-forced, `-b 26 --l1 5e-8`** | **79.99 / 76.46** | **80.22** | **77.64** | **4.9 MB** | **13 min** |
| Teacher-forced, `-b 26 --l1 1e-7`, VW tags | — | 74.99 | 70.13 | 0.58 MB | 9 min |

- **Teacher forcing wins again** (as for tags and entities), and is the
  fastest. The parser's dynamic oracle (`--cost_to_go`) scored 68.5-72.1
  LAS and took 20-56 min; arc-eager was worse (62.9).
- **A bigger hash table** (`-b 26`) was +2.4 LAS. The task builds its own
  pair and triple feature crosses over the stack and buffer, which collide a
  lot at `-b 24`.
- **More word features hurt.** Lowercase, suffix and shape features each
  made it slightly worse and 2-6x slower, since every one is multiplied
  through the triples.
- **L1 sets the size,** as for the tagger: 3e-8 keeps full accuracy at
  15.8 MB, 5e-8 costs ~0.9 dev LAS at 4.9 MB, 1e-7 costs 4.5 LAS at 0.58 MB.

For scale, non-neural transition parsers trained on treebanks typically get
UAS in the high 80s; this is a copy of another parser, from 488k tokens.
More spaCy-labeled text (the big lever for NER) and training on predicted
tags are the obvious next steps.

**Sentence splitting.** spaCy's small pipeline splits sentences with its
parser. The VW splitter labels each token of a paragraph as starting a
sentence or not (`--search_task sequence --search 2`, the tagger's settings
and features plus whether there is whitespace on each side of the token),
trained on spaCy's splits of the UD text as paragraphs. Scored over every
token but a paragraph's first, against spaCy's splits of EWT paragraphs:

| Model (`make sent-train`) | Training text | Dev start F1 | Test start F1 (95% CI) | Test sentences exact | Model size | Train time |
|---|---|---|---|---|---|---|
| Rule: start after `.` `!` `?` | — | 83.29 | 85.68 | — | — | — |
| Teacher-forced, `--l1 3e-7` | all UD (488k tokens) | 94.45 | — | — | 2.5 MB | 28 s |
| **Teacher-forced, `--l1 3e-6`** | **all UD** | **94.72** | **95.51 (94.03-96.65)** | **92.60%** | **0.62 MB** | **29 s** |
| Teacher-forced, `--l1 1e-5` | all UD | 93.11 | — | — | 0.01 MB | 28 s |
| Teacher-forced, `--l1 3e-7` | + 4M words C4 (5.1M) | 95.03 | — | — | 5.0 MB | 200 s |

Precision is high (97%) and recall lower (94%): 58 of the 65 starts it
misses on test follow a token that doesn't end in `.`, `!` or `?`, i.e.
sentences without final punctuation, which spaCy's parser splits and the
splitter joins. More text (C4) is within noise, so the default is UD-only.
spaCy's tokenizer is rule-based (`spacy.blank("en")`, the library without a
trained model, gives exactly the same tokens), so tokenization isn't learned.

### End to end

`pipeline.py` runs raw text through spaCy's rule-based tokenizer and then the
four VW models in turn: sentence splitter, tagger (spaCy's tags), entity
tagger, and parser (given the VW tags). `compare.py` scores its output
against `en_core_web_sm` run on the same paragraphs, matching tokens within
each paragraph, so a sentence-split error counts against the parse (those
heads are hard to get right) but not against tags or entities. Tokens agreed
on every paragraph.

| Text | Paragraphs | Tokens | Sentence starts F1 | Tags | Entity F1 | UAS | LAS |
|---|---|---|---|---|---|---|---|
| EWT dev | 750 | 25,686 | 94.72 | 94.21% | 72.72 | 81.55 | 76.75 |
| EWT test | 854 | 25,530 | 95.51 | 94.40% | 70.57 | 81.66 | 77.69 |
| C4, 1M held-out words | 23,371 | 1,162,354 | 96.57 | 93.90% | 68.28 | 80.54 | 76.72 |

(95% intervals are about ±1-3 points on EWT and ±0.2-0.4 on C4; `make
pipeline-eval` and `make c4-eval` print them.) The C4 text is the documents
after the NER training sample in the same shard (`make c4-eval` builds it,
runs both, and prints the times and peak memory below).

Speed and size on the C4 text, one process each on a 4-core CPU (2026-10-18,
with other jobs running on the machine):

| | Models | Wall time | Tokens/s | Peak memory |
|---|---|---|---|---|
| spaCy `en_core_web_sm` (`spacy-label.py`) | ~12 MB | 161 s | 7,200 | 0.51 GB |
| VW pipeline (`pipeline.py`) | 23.2 MB (0.6 + 8.3 + 9.3 + 4.9) | 265 s | 4,400 | 0.51 GB |

A rerun with `make c4-eval` on a quiet machine (2026-10-24) took 118 s for
spaCy and 198 s for VW (9,800 and 5,900 tokens/s), the same ratio, with
identical output and scores.

spaCy's time includes its lemmatizer and morphology, which the VW pipeline
doesn't produce. The VW models themselves (`vw -t`) take ~120 s of the 265
s: the parser 67 s, sentences, tags and entities 13-25 s each. Most of the
rest is the Python converters writing VW's text format (~60 s) and the
tokenizer (8 s). VW's own share is below spaCy's total, so a pipeline that
built features in memory and called VW's library directly could beat spaCy,
though not by a wide margin (not tried).

### Published results for comparison

Same treebank and tag set, but different UD releases (EWT is revised between
releases), so treat these as approximate.

| System | Kind of model | UD | XPOS, gold tokens | XPOS, raw text |
|---|---|---|---|---|
| This repo (`run.sh`, 8.4 MB) | VW linear tagger, hand-built features, L1 | 2.7 | 93.5 | — |
| This repo (`run.sh SIZE=small`, 0.40 MB) | same, stronger L1 | 2.7 | 90.9 | — |
| This repo (first hill-climb, ~770 MB) | same, no L1 | 2.7 | 93.0 | — |
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
efficiently, both in training cost and in model size. For tags, yes:

- Training is cheap: about 35 seconds on a 4-core CPU, no GPU, with teacher
  forcing.
- The model is small. 8.4 MB gets 93.5% test, more than the 770 MB model, and
  0.40 MB still gets 90.9%; spaCy's small English pipeline is ~12 MB (though
  that includes a parser, NER and more). Trained on spaCy's own output, the
  8.3 MB model matches spaCy's tags 94.5% of the time.

At 93.5% it is close to UDPipe 1's linear tagger (93.9%) but about 3.5 points
behind neural taggers, roughly twice as many errors. Most of the remaining
errors are unseen words, or words with a tag they never had in training, and
those need lexical knowledge from outside the treebank.

Named entities are the weak spot: the best VW model matches spaCy's entities
at F1 72 on test (9.3 MB, ~14 min training on 5.1M spaCy-labeled tokens, 4M
of them C4 web text). Labeled text in EWT's genre helped more than more text
of another genre; word clusters and POS-tag features didn't help. Dependency
parses are in between: a 4.9 MB VW parser agrees with spaCy on 86% of heads
and 83% of heads plus labels when given spaCy's tags, and 83% / 79% with the
VW tagger's (after attaching the last token VW's parser leaves unattached).
Sentence splitting is the easy part: 0.62 MB matches spaCy's sentence starts
at F1 95.5.

End to end, from raw text, the four VW models (23.2 MB, about twice spaCy's
small pipeline) agree with spaCy on 94% of tags, entities at F1 68-71, and
77-78% of heads plus labels. They're slower than spaCy as run here (4,400
vs 7,200 tokens/s), mostly because of the Python glue around VW, and each
is clearly less accurate than the spaCy component it copies. VW learns
each piece cheaply (the slowest, NER, trains in 14 min on a CPU), but it
doesn't beat spaCy's small pipeline on accuracy, size or speed.

Not tried, if anyone picks it up again: word clusters or embeddings for tags
(the only idea likely to gain more than a point there), more C4 text for NER
(with stronger L1 to hold the size), gazetteer features (`--dictionary`),
VW learning-rate settings, more spaCy-labeled text and predicted-tag training
for the parser, building features in memory and calling VW's library from
the pipeline, a VW tokenizer (characters tagged as token starts) to drop the
spaCy library entirely.

