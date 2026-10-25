# VW word embeddings: proof of concept

Goal: find out whether Vowpal Wabbit can be used to train word embeddings, as a
possible source of the "lexical knowledge from outside the treebank" that the
tagger and NER hill-climbs kept needing. Work lives entirely in `.pi/` with its
own venv (`.pi/.venv`, VW 9.11.9) so it never touches the main repo's
environment. All runs are in `results.tsv`; scripts are in this folder.

## What VW actually offers for embeddings

VW has no `word2vec`; the candidate mechanisms are:

- **`--lrq a b K`** — a rank-K factorization of the a x b interaction. Each
  feature in `a` and `b` gets a K-vector and the score is their dot product,
  which is exactly skip-gram with negative sampling. This is the only path that
  is cleanly *extractable* (see below).
- **`-q a b --rank K`** — VW's gradient-descent matrix factorization. It trains
  fine, but `--invert_hash` dumps raw pointers for it, not names, so the latent
  factors cannot be recovered with the tools in this wheel (there is a C++
  `gd_mf_weights` example in the VW source, not shipped here). Dropped.
- **`--nn K`** / **`--lda K`** — a hidden layer / topic distributions are
  embeddings of a sort but are not what we want and are not name-extractable.

Extraction route: `--dump_json_weights_experimental FILE
--dump_json_weights_include_feature_names_experimental`. It writes every
weight with a name; lrq factors come out as `w^WORD^K` / `c^WORD^K`. It is the
structured route (one JSON object), checked against `--invert_hash` on the same
model: identical to 5e-7. Both dumps are **single-pass only** — `--invert_hash`
refuses a cache file, and the JSON dump hangs under `--passes` (no output after
15 min) — so `pairs.py` bakes the epochs into the file instead.

One bug to route around: `--ignore_linear` on *both* namespaces (pure lrq, no
linear terms) makes VW 9.11.9 emit `update is NAN` and learn nothing. Ignore one
side only, or keep a linear term.

## Pipeline

`pairs.py` reads CoNLL-U (column 2) or `# text = ` lines, lowercases, drops
punctuation, subsamples frequent words (word2vec `t`), and emits
`+1 |w TARGET |c CONTEXT` / `-1 |w TARGET |c NEGATIVE` with unigram^0.75
negatives. `train.sh` / the raw VW call trains `--lrq wcK` and dumps JSON.
`extract_json.py` turns the dump into `word<TAB>v1..vK` and `.npz`.
`evaluate.py` reports nearest neighbours and kNN UPOS purity against a random
baseline (the fraction of random word pairs that share a tag). `ppmi_svd.py` is
the count-based control. `wiki_text.py` samples a WikiText-103 parquet shard
(read-only).

## Does the mechanism work at all? Yes.

On clean synthetic data — words belong to topics and co-occur only within topic
— `--lrq` recovers the structure:

| run | data | vocab | rank | kNN topic purity | random |
|---|---|---|---|---|---|
| 1a | 2 topics x 4 words | 8 | 2 | within-topic cos 1.00, cross -1.00 | — |
| 1b | 20 topics x 10 words | 200 | 10 | 0.882 | 0.050 |
| 1c | 20 topics x 10 words | 200 | 50 | 0.900 | 0.050 |
| 1d | 50 topics x 20 words | 1000 | 20 | 0.985 | 0.020 |

And on real data with a clean, coarse context signal — target word paired with
its gold POS tag — the word vectors cluster by POS:

| run | data | config | kNN UPOS purity | random |
|---|---|---|---|---|
| 2a | EWT word -> UPOS pairs, 488k examples | `--lrq wc10` | **0.659** | 0.164 |

So the optimizer, the factorization and the extraction all work. The question
is whether the *skip-gram* signal is learnable at our scale.

## Skip-gram on real text: weak

| run | data | tokens | vocab | rank | config | kNN UPOS purity | random |
|---|---|---|---|---|---|---|---|
| 3a | UD EWT+GUM+LinES+ParTUT+PUD | 424k | 7654 | 50 | `--ignore_linear c` | 0.219 | 0.158 |
| 4a | WikiText-103 sample | 1.0M | 15000 | 50 | `--ignore_linear c` | 0.223 | 0.192 |
| 5a | WikiText-103 sample | 200k | 8000 | 50 | `--ignore_linear c` | 0.198 | 0.170 |
| 5b | WikiText-103 sample | 200k | 8000 | 10 | lower rank | 0.205 | 0.164 |
| 5c | WikiText-103 sample | 200k | 8000 | 20 | lower rank | 0.194 | 0.168 |
| 5d | WikiText-103 sample | 200k | 1500 | 50 | smaller vocab | 0.178 | 0.148 |
| 5e | WikiText-103 sample | 200k | 8000 | 50 | `--neg-exclude` | 0.195 | 0.170 |
| 6a | WikiText-103 sample | 100k | 1500 | 50 | `--lrqdropout` | 0.158 | 0.152 |
| 6b | WikiText-103 sample | 100k | 1500 | 50 | `--ignore_linear w` | 0.167 | 0.153 |
| 7a | WikiText-103 sample | 500k | 3000 | 50 | `--ignore_linear c` | 0.209 | 0.161 |
| 7b | WikiText-103 sample | 500k | 3000 | 50 | `--power_t 0 -l 0.05` | 0.196 | 0.161 |
| 7c | WikiText-103 sample | 500k | 3000 | 50 | `--power_t 0 -l 0.5` | 0.203 | 0.161 |
| 7d | WikiText-103 sample | 100k | 1500 | 50 | `--lrq ww50` (shared) | 0.180 | 0.149 |

Everything sits ~1.1-1.4x the random baseline. The training loss settles around
0.42-0.58 (random is 0.693), so the model learns *something*, but the nearest
neighbours of content words are noise. Frequent **function** words do cluster
(`the` -> `in, of, a, by, from, and`), which is why purity is slightly above
random at all; content words (`london`, `year`, `war`) do not. More data, lower
rank, smaller vocabulary, negative-sampling hygiene, dropout, a constant
learning rate and a shared `ww` matrix all leave it essentially unchanged.

## The control: PPMI + SVD beats it on identical data

Same 500k words, same 3000-word vocabulary, same window. `ppmi_svd.py` counts
co-occurrence, takes positive PMI, and truncated-SVDs to 50 dims.

| method | kNN UPOS purity | random | example neighbours of `london` |
|---|---|---|---|
| **PPMI + SVD** | **0.295** | 0.156 | centre, station, town, york, hotel |
| VW `--lrq wc50` | 0.209 | 0.161 | led, money, begun, river., radio, middle |

The count-based baseline is clearly better and its neighbours are sensible
(`york` -> new, london, city, washington; `dog` -> friend, rescue, doctor). VW's
low-rank quadratic — a *secondary interaction learner* bolted onto a linear
model, not a tuned word2vec — loses to a few lines of numpy.

## Conclusion

VW can be made to train and expose word embeddings (`--lrq` + the JSON dump),
and it learns coarse structure (synthetic clusters 0.88-0.99; POS categories
0.66). But on real fine-grained skip-gram data at any scale this machine
afforded, it is a poor embedding trainer: a plain PPMI+SVD baseline beats it on
identical text and vocabulary, and its content-word neighbours are noise.

Recommendation: **do not use VW `--lrq` to train embeddings.** If embeddings are
wanted as tagger/NER features, use the count-based route (PPMI+SVD, the same
idea `clusters.py` already uses) or an external word2vec/fastText/GloVe, then
feed the vectors in as the `c` namespace. VW's value in this repo stays where it
is strong: the sequence tagger and NER model.

Not tried: `-q a b --rank K` with the C++ `gd_mf_weights` dumper (would need
building VW from source); `--nn` hidden layers; a genuinely large corpus
(hundreds of millions of words) where the skip-gram signal is strong enough to
change the picture. The last one is the only result here I would expect to move
much, but the PPMI control suggests the ranking would not flip.

---

# Round 2: fixing the bug, and every knob that followed

After round 1 I listed the untried knobs and ranked them. This is what they
did. Full numbers in `results2.tsv`. New tooling: `similarity.py` (Spearman on
SimLex-999 and WordSim-353), `similarity_metric.py` (the factorization-natural
metric `cos(E_c e_w)` vs raw cosine), `pairs_count.py` (count-derived labels),
`data/` (the datasets).

## The evaluation upgrade paid for itself immediately

Round 1's kNN-UPOS-purity was too coarse. SimLex-999 / WordSim-353 (Spearman)
show the gap plainly, and on the *same* 500k words and 3000-word vocabulary:

| method | train s | purity | SimLex | WordSim |
|---|---|---|---|---|
| PPMI + SVD (control) | 23 | 0.295 | +0.202 | +0.468 |
| VW `--lrq` skip-gram, 1 ep (round-1 setup) | 478 | 0.205 | -0.010 | -0.014 |

The VW embeddings had **zero** correlation with human similarity. The
round-1 conclusion held, but now it was measurable.

## The real bug: `--ignore_linear` cripples `--lrq`

Every round-1 run used `--ignore_linear c`. A controlled rank-5 matrix
(160k examples, `--loss_function squared`) settles what that flag does:

| config | 1 epoch recon corr | 20 epochs recon corr |
|---|---|---|
| no ignore | 0.706 | **0.987** |
| `--ignore_linear c` | 0.074 | 0.107 |
| `--ignore_linear w` | 0.085 | 0.098 |

Ignoring *either* namespace stops `--lrq` learning that namespace entirely —
the model never gets below the label variance, even after 20 epochs. This is a
VW 9.11.9 behaviour (bug?) I had been tripping over the whole time. Round-1
numbers are all from a crippled model.

Two things follow. First, drop `--ignore_linear` and keep a linear term on both
namespaces. Second, **`--lrq` needs many epochs**: at 1 epoch it reaches only
0.71 on a clean rank-5 matrix; at 20 epochs it reaches 0.99. The default
`--power_t 0.5` schedule does get there, just slowly.

## With the bug fixed: real data improves a little, not much

Same benchmark, `--lrq wc50`, skip-gram, no ignore:

| run | train s | purity | SimLex | WordSim |
|---|---|---|---|---|
| skip-gram, 1 ep | 483 | 0.213 | +0.019 | +0.216 |
| PPMI+SVD control | 23 | 0.295 | +0.202 | +0.468 |

WordSim jumps (0.01 -> 0.22) but SimLex stays flat, and PPMI+SVD is still far
ahead. So the bug mattered, but fixing it does not close the gap.

## Objective changes: fit the matrix, still bad embeddings

I handed VW the co-occurrence matrix directly (`pairs_count.py`), with the
linear terms in place:

| objective | train s | final loss | purity | SimLex | WordSim |
|---|---|---|---|---|---|
| PPMI regression (squared), 20 ep | 540 | 0.239 | 0.176 | -0.049 | +0.193 |
| log-count (squared), 20 ep | 598 | 0.250 | 0.249 | -0.049 | -0.007 |
| Poisson (count label), 20 ep | 717 | diverged | 0.184 | +0.016 | +0.006 |
| PPMI regression, `--ignore_linear c`, 1 ep | 37 | 1.444 | 0.184 | +0.011 | +0.120 |

PPMI regression fits well (loss 0.24 vs label variance 1.36) yet the extracted
vectors still do not track human similarity. I checked the obvious metric
objection: in `M ~= E_w E_c^T` the natural similarity is `cos(E_c e_w)`, not
raw `cos(e_w)`. Recomputing with the full model (linear weights included, both
raw and column/row-centred) gave SimLex +0.09 / WordSim +0.08 — no better. The
problem is the factorization VW finds, not the metric.

## Optimizer sweep

1-epoch screen on the PPMI target (fast), then the two best at 10 epochs:

| optimizer | 1 ep loss | 10 ep train s | 10 ep loss | purity | SimLex | WordSim |
|---|---|---|---|---|---|---|
| gd (default) | 1.539 | - | - | 0.158 | +0.026 | +0.171 |
| --invariant | 0.738 | 261 | 0.189 | 0.263 | +0.112 | -0.049 |
| --ftrl | 0.527 | 381 | **0.154** | 0.241 | +0.066 | +0.105 |
| --normalized | 1.730 | - | - | - | - | - |
| --adaptive | 9.399 | - | - | - | - | - | diverged |
| --coin | 2.546 | - | - | - | - | - | 915k NaN warnings |
| --pistol | 6.447 | - | - | - | - | - | diverged |
| --svrg | 38.3 | - | - | - | - | - | diverged |
| --OjaNewton | 1.711 | - | - | - | - | - | no factors extracted |
| --bfgs | - | - | - | - | - | - | exits immediately |

FTRL and `--invariant` fit much faster and push purity up (0.24-0.26 vs 0.16),
but neither beats the PPMI+SVD control on similarity. `--random_seed` has no
effect (lrq is deterministic); `--random_weights` does not help; `--lrqfa` hung.

## Speed: the name dumps, not the learning

| 1M lines, `--lrq wc50` | time |
|---|---|
| no dump | 6.2 s |
| `--readable_model` (hashed) | 6.4 s |
| `--invert_hash` / `--dump_json_weights` (named) | 90 s |

On 5.4M lines: 32 s to train, 467 s with `--dump_json`. The named dumps do a
hash-map lookup per feature per example (O(examples)); `--readable_model` does
not. I tried to build a fast name map via `--audit` (which prints
`lrq^w^word^k:index`) but its output was 1.2 GB for 30 examples, so that path
is out. A `--readable_model` + reimplemented-hash route is possible but VW's
hash is not plain murmur3 with the seeds I tried.

## Round-2 conclusion

Even after removing the `--ignore_linear` bug, training many epochs, switching
the objective to the co-occurrence matrix itself, and trying ten optimizers and
several inits, VW's `--lrq` embeddings stay well below a 23-second PPMI+SVD
baseline on the same text and vocabulary:

| best of each family | purity | SimLex | WordSim |
|---|---|---|---|
| PPMI+SVD | 0.295 | +0.202 | +0.468 |
| VW best (FTRL / invariant) | 0.263 | +0.112 | +0.105 |
| VW skip-gram (bug fixed) | 0.213 | +0.019 | +0.216 |

`--lrq` can fit a low-rank matrix (corr 0.99 on synthetic, loss 0.15 on PPMI),
but the factors it lands on are not the similarity-preserving ones that SVD
finds. The recommendation stands and is now much better supported: **use
PPMI+SVD (or an external word2vec) for embeddings; keep VW for the tagger.**

## Round 3: giving VW its best shot — `-q wc --rank K` (gd_mf)

`--lrq` is a low-rank-quadratic *reduction*; `-q a b --rank K` is VW's actual
gradient-descent **matrix factorization** (gd_mf, the Netflix implementation),
with random init and its own LR regime. That is the fairer test of "can VW's MF
produce embeddings," so I ran it.

**Extraction was the blocker, and I solved it.** For gd_mf, `--invert_hash` and
`--readable_model` print raw pointers, and `--dump_json_weights` gives
`"terms": null` (anonymous indices) — all useless. But **`--audit` prints each
factor with its name and index** (`w^alpha:43569185(...):1:0.09022`). Training
with `-f`, then running `-i model -d chunks.vw --audit` on a tiny file holding
every vocabulary feature, dumps all factors. `extract_audit.py` parses it (the
lowest index per feature is its linear weight; the next K are the factors).
Validated on the controlled rank-5 matrix: recon corr **0.998**.

**Result** — gd_mf rank 50 on the *same PPMI matrix* as the control, movielens
regime (`-l 0.015 --power_t 0 --decay_learning_rate 0.97 --l2 0.001`,
20 passes):

| run | train s | final loss | purity | SimLex | WordSim |
|---|---|---|---|---|---|
| PPMI+SVD (control) | 23 | - | 0.295 | +0.202 | +0.468 |
| **gd_mf `--rank 50` on PPMI** | 785 | **0.119** | 0.238 | +0.057 | -0.001 |
| lrq on PPMI (round 2) | 540 | 0.239 | 0.176 | -0.049 | +0.193 |

gd_mf fits the matrix *better than* lrq (loss 0.119 vs 0.239) and is far more
expensive, yet its embeddings are just as poor. The neighbours are the clincher:
`london` -> runs, placed, operation, division (gd_mf) vs centre, station, town,
york, hotel (PPMI+SVD). The factorization-natural metric `cos(E_c e_w)` does not
rescue it either (+0.062).

**This closes the question.** The gap is not `--lrq` vs `--rank`, and not
optimization: gd_mf reaches a lower loss than the exact-SVD control's objective
and still lands in a non-similarity-preserving basis. The exact SVD's `U·S` is
the right embedding; SGD (of either reduction) finds a different factorization
of the same matrix. PPMI+SVD remains the answer.
