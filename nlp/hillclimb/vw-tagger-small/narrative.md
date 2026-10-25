# VW tagger hill-climb: smallest model at >= 90% dev

Goal: minimize the saved model file (`-f`, MB) while dev token accuracy stays
>= 90.0% (UD English-EWT, XPOS). Train time is tracked, not constrained.
Training runs one job at a time (the machine has ~3 GB free RAM).
Test is scored once, at the end, for the winner.

Reference size: spaCy's small English pipeline is about 12.2 MB, the number
to beat. At ~12 bytes per stored weight that is about 1M nonzero weights.

Starting point (from `../vw-tagger/`): the accuracy winner (r8a) is 92.95% dev
at 772 MB. Its feature set at `-b 24` was 91.86% / 210 MB.

How size works: this VW build writes only nonzero weights, about 12 bytes
each (the `-b 18` model from 2021 is 3.3 MB, about 2^18 x 12.5 bytes; the
`-b 26` model is about 2^26 x 11.5). So size = number of nonzero weights,
and two levers move it: a smaller hash table (`-b`) and sparsity (`--l1`).

Keep rule: a change is kept if it makes the model smaller while dev stays
>= 90.0. Noise: single-run dev CI is about +-0.35 points (paired difference
about +-0.2), so results within ~0.3 of the 90.0 line get a margin note.
Stop: 3 rounds in a row without a >= 20% size cut, or ~3 hours of compute.

Tools: `featurize.py` (now also `nK`: neighbor words as an ordinary
namespace, for `--lrq`), `hc.py` (train + score dev), `round.sh` (runs a
round's configs one at a time), `rN.cfg` (each round's configs),
`results.tsv` (every run).

| round | change | dev acc | MB | kept? |
|---|---|---|---|---|
| 1a | r8a features, `-b 22` | 90.44 | 52.4 | yes (just above the line) |
| 1b | r8a features, `-b 20` | 87.21 | 13.1 | no: below 90 |
| 1c | `-b 24 --l1 1e-6` | 48.38 | 209.7 | no: L1 far too strong, and the file did not shrink |
| 1d | `-b 24 --l1 1e-7` | 84.42 | 209.7 | no: same |

Round 1 finding: VW 9 saves training state (the optimizer's per-weight
statistics) with the model by default, so the file is about 12.5 bytes per
hash slot and L1-zeroed weights are still written. `--predict_only_model`
drops that state: on a 300-sentence smoke test the file halved (820 KB ->
410 KB), and with `--l1 1e-5` it fell to 10 KB. The models in `../vw-tagger/`
could all have been half the size. Every run from round 2 on uses it.

| 2a | 1a + `--predict_only_model` | 90.44 | 26.2 | yes: same accuracy, half the size (as predicted) |
| 2b | `-b 21`, predict-only | 89.27 | 13.1 | no: below 90 |
| 2c | `-b 24 --l1 1e-8`, predict-only | 92.15 | 28.0 | yes: +1.7 over 2a at about the same size |
| 2d | `-b 24 --l1 1e-9`, predict-only | 92.32 | 94.4 | no: barely sparser |

Round 2 finding: L1 beats shrinking the table. Shrinking `-b` forces
unrelated features to share weights; L1 keeps a big table (few collisions)
and drops the weights that don't matter. Next: push L1 harder, and try it on
`-b 26`. The `--nn` / interaction / `--lrq` ideas move to round 4, run on top
of the best L1 setting (draft in `r4-draft.cfg`, written for a fixed `-b 20`).

| 3a | `-b 24 --l1 2e-8` | 90.84 | 5.58 | yes |
| 3b | `-b 24 --l1 4e-8` | 88.73 | 0.66 | no: below 90 |
| 3c | `-b 26 --l1 2e-8` | 92.33 | 22.70 | dominated by 3d for this goal |
| 3d | `-b 26 --l1 4e-8` | 90.60 | 2.54 | yes: best so far, a fifth of spaCy's 12.2 MB |

Round 3 finding: at the same L1 strength, `-b 26` is ~1.5-1.9 points better
than `-b 24` (fewer collisions) and L1 still keeps the file small. `-b 27`
would need ~3 GB RAM to train, more than the machine has free, so `-b 26` is
the ceiling here.

| 4a | `-b 26 --l1 5e-8` | 89.88 | 1.30 | no: below 90 |
| 4b | `-b 26 --l1 6e-8` | 89.32 | 0.79 | no: below 90 |
| 4c | `-b 26 --l1 3e-8` | 91.57 | 6.35 | larger than 3d |

L1 curve at `-b 26` (dev / MB): 2e-8 92.33/22.7, 3e-8 91.57/6.35,
4e-8 90.60/2.54, 5e-8 89.88/1.30, 6e-8 89.32/0.79. Every 1e-8 of L1 roughly
halves the file and costs ~0.6-0.9 points. The 90% line falls between 4e-8
and 5e-8, so 3d stays the best setting for this feature set.

| 5a | 3d without `-q 'w\x83'` | 89.30 | 0.82 | no: below 90 |
| 5b | 3d with `-q ln` (lowercase x neighbor words +-1) instead of `-q 'w\x83'` | 90.71 | 1.71 | yes: smaller and slightly more accurate than 3d. **Best so far** |
| 5c | 3d with `--lrq ln4` instead of `-q` | 80.33 | 21.31 | no: much worse and bigger |
| 5d | 3d + `--nn 8` | - | - | not finished: Claude Code stopped the round for low system memory (~1 GB free, swap full) |

Round 5 finding: the word x neighbor cross is worth ~1.3 points, but only the
lowercase-word x neighbor-word part of it is needed; crossing with neighbor
suffixes/shapes as well costs size for nothing at this L1 strength. Low-rank
crosses (`--lrq`) don't work well here, possibly because L1 also zeroes the
latent factors.
| 6a | 5b + `--nn 8` (rerun with more free memory) | 13.30 | 0.00 | no: L1 4e-8 zeroed the whole network (constant output); 2317 s to train, 9x slower. Not a fair test of `--nn`; 7a retries it without L1 |

## Distillation (spaCy labels)

`spacy-label.py` runs `en_core_web_sm` 3.8.0 (spaCy 3.8.16) on the raw
sentence text of the UD 2.7 English treebanks and writes its predictions as
CoNLL-U. Train: EWT train + GUM + LinES + ParTUT + PUD, 488k spaCy tokens
(ESL and GUMReddit ship masked text). Dev/test: EWT dev/test text, labeled the
same way. "dev" for `d*` runs = agreement with spaCy's tags on dev, not gold
accuracy. Data in `data/spacy-work/` (all) and `data/spacy-ewt-work/` (EWT
train text only, 207k tokens). Runs set `HC_DATA` to pick the folder.

| run | data | config | spaCy agreement (dev) | MB | train s |
|---|---|---|---|---|---|
| d1a | all (488k tok) | 5b (`-b 26 --l1 4e-8 -q ln`) | 86.93 | 0.33 | 772 |
| d1b | all (488k tok) | r8a-style, no L1, `-q 'w\x83'` | 94.13 | 416.92 | 828 |
| d1c | EWT train text (207k tok) | 5b | 90.87 | 1.93 | 234 |

d1 finding: on EWT text alone, the 1.9 MB config matches spaCy's tags 90.9% of
the time, the same as its gold accuracy (90.7). On 2.4x more text it got
*smaller and worse* (0.33 MB, 86.9): VW applies L1 on every update, so the
same `--l1` is ~2.4x stronger on 2.4x the examples. L1 needs scaling by data
size (4e-8 x 207/488 ~ 1.7e-8), tried in d2. The no-L1 ceiling on all data
is 94.1% agreement (417 MB). (d1a and d1b scoring the same 1122/1581 on
unseen words is a coincidence: checked.)

| 7a | r8a features `-b 22 --nn 8`, no L1 (fair `--nn` test vs 2a) | 89.25 | 26.22 | no: 1.2 points *worse* than 2a at the same size, 1782 s (7x slower). `--nn` is dropped |

| run | data | config | spaCy agreement (dev) | MB | train s |
|---|---|---|---|---|---|
| d2a | all (488k tok) | 5b with `--l1 1.7e-8` (scaled) | 90.67 | 2.10 | 834 |
| d2b | all (488k tok) | 5b with `--l1 2.5e-8` | 89.54 | 0.86 | 793 |

d2 finding: scaling L1 by data size works as predicted (d2a lands at about
d1c's size), but the extra 280k tokens of non-EWT text don't help the small
model: d1c (EWT text only) is both more accurate and smaller, and trains 3.5x
faster. At ~2 MB the model is capacity-bound, not data-bound. Distillation
winner: d1c.

## Final (test scored once)

`final.sh` retrains each winner with `HC_TEST=1` (hc.py keeps the model and
scores test; row in `test-results.tsv`). Dev reproduced exactly.

| run | dev | test (95% CI) | test unseen | MB | train s |
|---|---|---|---|---|---|
| 5b, gold EWT tags | 90.71 | 90.72 (90.27-91.16) | 73.56 | 1.71 | 233 |
| d1c, spaCy tags (agreement) | 90.87 | 91.33 (90.58-91.97) | 77.11 | 1.93 | 237 |

## NER (distilled from spaCy)

Data: `data/spacy-ner-work/`, from `conllu-to-vw.py --label ner` on spaCy's
output for EWT train/dev/test text (207k / 26k / 26k tokens). Labels are
spaCy's 18 entity types in VW `sequencespan` numbering (1 = O, 2n = B, 2n+1 = I;
37 labels). Metric: entity F1 against spaCy (exact span and type), plus
precision, recall and token accuracy, in `results-ner.tsv`. Dev has 1,711
spaCy entities.

| run | change | F1 | P | R | tok acc | MB | train s |
|---|---|---|---|---|---|---|---|
| n1a | `sequencespan` BIO, r8a features, `-b 24` | 38.59 | 58.41 | 28.81 | 91.21 | 1.83 | 79 |
| n1b | n1a + `--search_span_bilou` | 0.36 | 0.20 | 2.10 | 12.16 | 1.02 | 230 |
| n1c | n1a + `-q ln` (n1 features) | 34.38 | 56.72 | 24.66 | 90.91 | 1.93 | 91 |
| n1d | n1a, small config (`-b 26 --l1 4e-8 -q ln`) | 30.23 | 69.92 | 19.29 | 90.50 | 0.75 | 112 |
| n2a | plain `--search_task sequence --search 37`, `-b 24` | **55.19** | 56.89 | 53.59 | 93.41 | 48.85 | 115 |

n1/n2a finding: VW's `sequencespan` task hurts here. In BIO mode it predicts O
for 1,020 of 1,705 B- tokens (plain `sequence`: 461), so recall halves. Its
BILOU mode is broken in VW 9.x for `-p` output: `takedown` converts a *copy*
of the label back to BIO (`VW::multiclass_label ylab = ...`, source
`search_sequencetask.cc`), and decoding the file with `y/2+1` still gives
garbage. Plain `sequence` with conlleval-style span decoding (a stray I-
starts a new entity) is the base from here.
| n2b | n2a + `-q 'w\x83'` | 55.21 | 60.63 | 50.67 | 93.45 | 77.54 | 156 |
| n2c | n2a, `-b 26` | 55.80 | 57.59 | 54.12 | 93.46 | 68.22 | 115 |
| n2d | n2a, 5 passes | 53.83 | 53.49 | 54.18 | 93.29 | 49.39 | 223 |
| n2e | n2a on all 488k tokens | 54.33 | 52.81 | 55.93 | 93.71 | 64.01 | 366 |

n2 finding: flat. Everything is within ~1 point of 55 (entity F1 on 1,711
entities has a standard error of roughly 1.2 points). Error breakdown for n2c
on dev: 926 exact, 168 right span / wrong type, 295 overlapping span, 322
missed. Typical errors are spans that run on ("Tuesday nominated two" as one
DATE) and spaCy's own noise ("Holocaust" WORK_OF_ART, "Shojai" ORG). Training
fit for n2a is F1 86.5 (P 85.6 / R 87.5): the model memorizes rather than
generalizes, so round 3 tries generalization levers.
| n3a | n2a, 2 passes | 56.02 | 59.17 | 53.19 | 93.38 | 48.62 | 68 |
| n3b | n2a, 1 pass | 57.14 | 61.12 | 53.65 | 93.51 | 47.98 | 31 |
| n3c | n2a without `w`/`p` (neighbors use `l`) | 52.38 | 55.53 | 49.56 | 93.16 | 39.94 | 104 |
| n3d | n2a + shape/flags +-2 as neighbor features | 55.51 | 57.56 | 53.59 | 93.46 | 49.03 | 126 |
| n3e | n2a + `--search_rollin ref --search_rollout none` | **62.64** | 63.41 | 61.89 | 94.31 | 13.66 | 12 |

n3 finding: teacher forcing (train on the reference previous tags, no
rollouts) is +7.4 F1, 10x faster to train and a quarter of the size. The
default search setup (learned roll-in, mixed roll-out) hurts here.
| n4a | n3e, 1 pass | 61.32 | 63.59 | 59.21 | 94.02 | 12.91 | 5 |
| n4b | n3e, 5 passes | 62.26 | 62.87 | 61.66 | 94.29 | 13.81 | 19 |
| n4c | n3e, `-b 26` | 62.87 | 63.56 | 62.19 | 94.36 | 15.16 | 13 |
| n4d | n3e + `-q 'w\x83'` | 61.47 | 65.87 | 57.63 | 94.14 | 40.58 | 21 |
| n4e | n3e on all 488k tokens | **63.47** | 63.52 | 63.41 | 94.30 | 17.95 | 27 |

## POS with teacher forcing (back from NER)

| run | change | dev | MB | train s |
|---|---|---|---|---|
| 8a | 5b + `--search_rollin ref --search_rollout none` | **93.63** | 39.14 | 32 |
| 8b | r8a features `-b 24` + teacher forcing | 93.19 | 69.42 | 30 |

Finding: teacher forcing also helps POS a lot. 8a beats the 772 MB
accuracy winner (92.95) at 39 MB and 32 s of training (vs ~4.5 min). It
sparsifies less with the same `--l1` (one update per token instead of many
rollout updates), so round 9 scales L1 up. (8a/8b were first scored as NER by
a bug: "-LRB-" contains "B-"; fixed and rescored from their dev.pred.)
| 9a | 8a, `--l1 2e-7` | **93.72** | 13.92 | 33 |
| 9b | 8a, `--l1 5e-7` | 92.99 | **1.72** | 33 |
| 9c | 8a, `--l1 1e-6` | 85.41 | 0.09 | 33 |
| 9d | r8a (`-b 26 -q 'w\x83'`, no L1) + teacher forcing | 93.56 | 122.35 | 36 |

Round 9 finding: with teacher forcing the 1.7 MB model is 92.99% (was 90.71)
and trains in 33 s (was 233). The best dev ever is now 9a, 93.72% at 13.9 MB.

NER on all data, teacher forcing:

| run | change | F1 | P | R | tok | MB | train s |
|---|---|---|---|---|---|---|---|
| n5a | n4e, `-b 26` | 64.06 | 64.53 | 63.59 | 94.52 | 20.80 | 30 |
| n5b | n4e + `-q ln` | **66.82** | 67.62 | 66.04 | 94.84 | 27.32 | 34 |
| n5c | n4e + `--l1 1e-7` | 63.92 | 63.85 | 64.00 | 94.39 | 8.82 | 42 |
| n5d | n4e + `--l1 4e-7` | 64.26 | 65.13 | 63.41 | 94.39 | 2.41 | 45 |
| 10a | 8a, `--l1 7e-7` | 91.18 | **0.40** | 33 |
| 10b | 8a, `--l1 3e-7` | **93.75** | 8.39 | 34 |
| 11a | 8a, `--l1 8.5e-7` | 88.68 | 0.17 | 33 |

POS winners: 10a (smallest >= 90%) and 10b (most accurate, under spaCy's 12.2 MB).

NER, continued:

| run | change | F1 | P | R | tok | MB | train s |
|---|---|---|---|---|---|---|---|
| n6a | n5b + `--l1 4e-7` | 65.15 | 66.98 | 63.41 | 94.72 | 3.01 | 54 |
| n6b | n5b, `-b 26` | 65.95 | 67.01 | 64.93 | 94.72 | 35.29 | 35 |
| n6c | n5b, `-b 26 --l1 2e-7` | 66.49 | 67.74 | 65.28 | 94.83 | 10.19 | 53 |

## Final 2 (teacher forcing; test scored once, `final2.sh`)

| run | data | dev | test (95% CI) | MB | train s |
|---|---|---|---|---|---|
| s10a | gold EWT | 91.18 | 90.92 (90.47-91.39) | 0.40 | 33 |
| s10b | gold EWT | 93.75 | 93.50 (93.09-93.90) | 8.39 | 37 |
| s10a | spaCy tags (agreement) | 91.07 | 91.83 (91.08-92.47) | 0.40 | 33 |
| s10b | spaCy tags (agreement) | 94.33 | 94.53 (94.18-94.88) | 8.34 | 34 |
| n5b | spaCy NER, all text (F1) | 66.82 | 63.86 (61.55-66.20) | 27.32 | 34 |
| n6c | spaCy NER, all text (F1) | 66.49 | 66.20 (63.91-68.40) | 10.19 | 54 |

## NER climb (target: dev entity F1 vs spaCy, model <= ~12 MB)

Plan approved 2026-10-17: VW flags, features and more spaCy-labeled text in
scope; scoring and dev/test data off-limits. Stop after 3 rounds without +2
F1 on dev, or ~3 h. Baseline: n6c (66.49 dev, 10.19 MB, 53 s). Paired
bootstrap noise on dev F1: n6c - n5b = -0.34 (95% CI -1.61..+0.89), so the
half-width is ~1.3 and +2 is a real gain.

| round | change (vs base) | F1 | P | R | tok | MB | train s | kept? |
|---|---|---|---|---|---|---|---|---|
| e1a | words +-2 in `n` (`n2`) | **68.22** | 69.56 | 66.92 | 95.07 | 13.31 | 77 | best so far (+1.7, under the +2 bar; over the 12 MB cap by 1.3 MB, needs a bit more L1) |
| e1b | + `-q sn` | 65.59 | 67.20 | 64.06 | 94.71 | 14.24 | 80 | no |
| e1c | tag history 2 (+ bias 3-grams) | 66.27 | 67.85 | 64.76 | 94.81 | 11.81 | 84 | no |
| e1d | `--search_max_quad_ngram_length 2` | 66.49 | 67.74 | 65.28 | 94.83 | 10.19 | 59 | no (identical to base: no effect) |

Round 1: no +2 gain (1 of 3 flat rounds). Carry `n2` forward.
| e2a | e1a on UD + 4M words WikiText (5.08M tokens), `--l1 2e-8` | **73.73** | 74.66 | 72.82 | 95.85 | 48.34 | 884 | over the 12 MB cap |
| e2b | same, `--l1 5e-8` | **72.92** | 74.80 | 71.13 | 95.75 | 10.01 | 752 | **yes: +4.7 over e1a, +6.4 over base, under cap** |

Round 2: 10x more spaCy-labeled text is the first real gain (e2b, +6.4 F1 vs
the n6c base at about the same size). Training time grows with the data
(752 s). Flat-round count resets to 0.

Clusters for round 3: `clusters.py` on 20M words of WikiText (seed 1) + the
UD "all text" training sentences (23.9M tokens, 50k-word vocabulary): PPMI of
left/right neighbors (1,000 context words), 100-d SVD, k-means at 64/256/1024.
4 min, 2 GB. Spot check: one 1k-cluster holds nintendo/universal/disney/sony,
another john/james/william, another london/bristol/portland. Built from
training text only (no dev/test text).
| e3a | e2b + clusters (`c`; neighbors +-1) | 71.88 | 73.96 | 69.90 | 95.68 | 7.97 | 972 | no |
| e3b | e2b + clusters (neighbors +-2) | 71.72 | 73.90 | 69.67 | 95.64 | 6.80 | 1007 | no |

Round 3: clusters don't help. The extra features also make the same `--l1`
prune harder (8 MB vs 10 MB), but even allowing for that (e2's L1 curve puts
~8 MB at ~72.7) they're ~-0.8, within noise. With 5M spaCy-labeled tokens the
model already sees most of the vocabulary clusters would generalize over.
Flat rounds: 1.

Round 4 data: 6M more WikiText words (same parquet shard, paragraphs not in
the 4M sample), spaCy-labeled in 10m36s. Training set
`data/spacy-ner-wiki10-work`: UD all text + 10M words WikiText, 471,994
sentences, 11,973,329 tokens (2.4x round 2), shuffled with seed 0.
| e4a | e2b settings on 12M tokens, `--l1 2.5e-8` | **74.39** | 76.40 | 72.47 | 95.94 | 12.30 | 1869 | best dev, but 0.3 MB over cap |
| e4b | same, `--l1 3.5e-8` | 71.14 | 73.25 | 69.14 | 95.48 | 3.59 | 1739 | no |

Round 4: 2.4x more text gives +1.5 at about the same size (under the +2 bar).
Data still helps but with diminishing returns: 10x gave +6.4, 2.4x more gives
+1.5. The L1 curve is steep here (12.3 MB -> 3.6 MB for 2.5e-8 -> 3.5e-8).
Flat rounds: 2. Compute so far ~2.3 h.

Round 5 (last; would reach both 3 flat rounds and ~3 h): e4a with
`--l1 2.7e-8` to fit the cap, run in final mode (dev + test in one go).
Selection is still on dev: if it falls below e2b's 72.92 on dev, its test score
is ignored and e2b is scored instead.
| e5a | e4a with `--l1 2.7e-8` (final mode) | **73.79** | 75.68 | 72.00 | 95.88 | 9.70 | 1654 | **yes: winner (best dev under cap)** |

Round 5: +0.9 over e2b on dev, under the +2 bar. Flat rounds: 3, and compute
is ~2.9 h, so the climb stops here (plateau).

## NER climb result

Winner e5a (selected on dev over e2b, 73.79 vs 72.92). Test, scored once:
**F1 69.27** (P 71.57, R 67.11; sentence bootstrap 95% CI 66.96-71.39),
9.70 MB, 1654 s training. Paired vs the climb's baseline n6c on test: +3.07
(95% CI +1.29..+4.86); on dev +7.31 (+5.59..+8.99). `ner-boot.py` computes
these and per-type F1.

The dev->test gap (4.5 F1) is wider than the baseline's (0.3). Per type,
PERSON/ORG/GPE gain on both splits (test +6.6/+6.9/+3.4); DATE and CARDINAL
gain on dev but lose a little on test (-2.2/-0.7), and the rare types are
noisy either way. The likely cause is genre: the WikiText text is
encyclopedic, EWT test is web/email/reviews. Some of the dev gain is also
selection on dev across ~11 runs.

Next levers (not tried): genre-matched unlabeled text (web/reviews) for
spaCy to label, predicted POS tags as NER features, gazetteers.

## NER round 6 (step 3: after the parser climb)

Two new ideas, each screened cheaply against an existing control:
- POS features (`tK` group: tags of the word and neighbors within K). Train
  uses spaCy's XPOS (from data/spacy/train.conllu), dev/test the VW tagger's
  predictions (final-s10b-spacy, 94.5% agreement), as a pipeline would see.
  Data: data/spacy-ner-all-pos-work (UD text only). Control e1a 68.22.
- Genre: 4M words of C4 (allenai/c4 en, shard 00000, pinned
  1588ec45, ODC-BY; web text, closer to EWT) instead of 4M of WikiText.
  Data: data/spacy-ner-c4-work. Control e2b 72.92 (same flags).

First attempt (Oct 17) was killed by Claude Code for low system memory (a
restic backup held 2.7 GB, swap full); rerun Oct 18 after it finished.

| run | data | dev F1 | vs control | MB | train s |
|---|---|---|---|---|---|
| e6a-pos1 | UD, +t1 | 68.36 | +0.14 vs e1a | 13.10 | 67 |
| e6b-pos2 | UD, +t2 | 68.30 | +0.08 vs e1a | 12.91 | 72 |
| e6c-c4-4m | UD + C4 4M | **74.92** (72.66-77.14) | **+2.01 (+0.21..+3.87)** vs e2b | 15.86 | 794 |

- POS features: noise. The `s`/`h`/`l` features already carry what a tag
  would; dropped.
- C4 beats the same amount of WikiText (paired CI excludes 0) and is +1.13
  (-0.36..+2.66) over e5a, which used 2.5x more text. Per type, C4 wins the
  frequent web-ish types (ORG +4.2, DATE +5.8, MONEY +10.4, TIME +6.0) and
  loses the rare encyclopedic ones (LOC, FAC, WORK_OF_ART, PRODUCT), which
  suggests mixing the two. Over the 12 MB cap, so L1 needs raising.

## NER round 7: fit C4 under 12 MB; C4 + WikiText mix

- e7a: UD + C4 4M, l1 7e-8.
- e7b/e7c: UD + C4 4M + WikiText 4M (data/spacy-ner-c4wiki-work), l1 5e-8 / 8e-8.

| run | data | l1 | dev F1 | MB | train s |
|---|---|---|---|---|---|
| e7a | UD + C4 4M | 7e-8 | 73.03 | 5.47 | 961 |
| e7b | UD + C4 4M + wiki 4M | 5e-8 | 69.77 | 1.71 | 1723 |
| e7c | UD + C4 4M + wiki 4M | 8e-8 | 54.16 | 0.16 | 1713 |

L1 is applied per update, so the 2x-larger mix needs a much smaller L1; both
mix runs were over-regularized. Size vs L1 is steep (C4: 5e-8 -> 15.9 MB,
7e-8 -> 5.5 MB). Round 8 brackets: C4 at 6e-8, mix at 3e-8 and 2.5e-8.

## NER round 8

| run | data | l1 | dev F1 | MB | train s |
|---|---|---|---|---|---|
| e8a | UD + C4 4M | 6e-8 | **74.41** (72.14-76.62) | 9.35 | 923 |
| e8b | UD + C4 4M + wiki 4M | 3e-8 | 74.49 | 13.15 | 1797 |
| e8c | UD + C4 4M + wiki 4M | 2.5e-8 | 75.53 | 21.01 | 1482 |

Under the 12 MB cap, the mix doesn't beat C4 alone (e8b is over the cap and
only level with e8a). e8a vs e5a: +0.62 (-0.83..+2.19), not significant, but
smaller (9.35 vs 9.70 MB), half the training time and less than half the
extra text. Rounds 6-8 gave no +2 F1 under the cap: plateau, stop. Final
candidate e8a, scored on test once (run-e8.sh).

## NER final (C4)

e8a on test: **F1 71.84** (69.70-73.79), P 76.06, R 68.07; paired vs e5a
+2.57 (+0.94..+4.13). Dev-test gap 2.6 (e5a: 4.5), consistent with the genre
explanation. Applied to the repo: `make ner-prepare` now labels the first 4M
words of the C4 shard (c4-to-text.py), run.sh TASK=ner uses --l1 6e-8.
Reproduced end to end: c4-4m.txt, c4-4m.conllu, train data and model.vw all
byte-identical; prepare 8.5 min (shard already downloaded), train 14m10s,
1.1 GB RSS, 9,347,400 bytes.
