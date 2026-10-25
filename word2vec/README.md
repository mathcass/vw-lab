# Word embeddings with Vowpal Wabbit

Can VW train word vectors as good as word2vec's? **Yes, at equal data, with
the right setup, but slowly.** On the same tokens and the same benchmarks:

- **10M tokens of WikiText, 5 epochs:** CBOW examples streamed into
  `--lrq wc100 --pistol` slightly beat gensim word2vec on all three
  benchmarks (SimLex 0.274 vs 0.273, WordSim 0.653 vs 0.624, analogies 35.0%
  vs 31.9%). That took 2.4 hours on one core against gensim's 3 minutes on
  two. The practical recipe, gd_mf (`-q wc --rank 100`), takes 21 minutes and
  nearly matches word2vec (0.268 / 0.607 / 28.2%).
- **100M tokens (WikiText + C4), 1 epoch, 189k-word vocabulary:** gd_mf
  matches gensim skip-gram on similarity (SimLex 0.337 vs 0.338, WordSim
  0.661 vs 0.650) but trails on analogies (42.2% vs 45.8%). It takes 60
  minutes against gensim's 7, in 347 MB against gensim's 421 MB.
- **VW's niche is fixed memory and streams.** With the words hashed into a
  table smaller than the vocabulary, VW used 228 MB (54% of gensim's) for a
  small loss. With `gen_online.py` it trains in a single pass over raw text
  with no vocabulary at all. gensim can't, because it builds its vocabulary
  before training.

For plain word vectors from text, gensim is still the better tool: 8-45x
faster, multi-threaded, and free of the gotchas listed below. Pretrained
vectors (fastText, GloVe, spaCy's `en_core_web_lg`), trained on 60-1000x more
text, beat anything trained here. See
[When VW is the better tool](#when-vw-is-the-better-tool).

## Running it

Everything runs from this folder with `make` (or `make -C word2vec TARGET`
from the repo root). This folder is its own [uv](https://docs.astral.sh/uv/)
project on Python 3.12-3.13, because gensim has no wheel for Python 3.14,
which `nlp/` uses. Package versions are pinned to the ones the results
came from (VW 9.11.9, numpy 2.5.4, scipy 1.18.1, gensim 4.4.0). The original
runs used Python 3.14 for VW and 3.12 for gensim; a VW rerun here on 3.13
gives identical results.

```
make setup benchmarks corpus  # env, benchmark files (pinned + checksummed), 10M-token corpus (~2 min)
make smoke                    # whole pipeline on the dev slice, 1 epoch (~2 min)
make fast                     # VW gd_mf rank 100, 5 epochs: the practical recipe (~21 min)
make best                     # VW lrq + PiSTOL rank 100, 5 epochs: the best vectors (~2.4 h)
make baselines                # gensim word2vec + PPMI+SVD on the same tokens (~10 min)
make dev-optimizers           # PiSTOL vs FTRL vs VW's default update, 2M-token slice (~13 min)
make dev                      # the key dev-slice runs (~1.3 h)
make big-corpus scale probes  # 100M tokens: memory, time and quality (~3.5 h)
make table                    # the latest result of each run above
```

Vectors land in `data/word2vec-work/runs/NAME.wc.npz` (`words`, `vectors`;
`.w`, `.c` are the two sides separately). `make` targets reuse the run names
already in `results/results.tsv`, so a rerun adds a row next to the recorded
one. Train time is wall time on a shared 4-core machine, so treat times as
rough.

Inputs come from the repo's shared `data/` directory: the WikiText and C4
shards are the root Makefile's pinned downloads, and `make benchmarks` puts
SimLex-999, WordSim-353 and the Google analogies in `data/word-benchmarks/`.
A rebuild from scratch reproduces the original corpus byte for byte, and a
rerun of `dev_gdmf_cbow_e5_l0.05` reproduces its loss and scores exactly
(`repro_dev_gdmf_cbow_e5_l0.05` in `results/results.tsv`).

## Method

### Data and evaluation

- **Corpus.** `corpus.py` tokenizes the root Makefile's 10M-word WikiText
  sample (`data/wikitext/wiki-10m.txt`) once: lowercase runs of letters and
  digits, inner apostrophes kept. Every method, VW or not, trains on these
  tokens, restricted to the 20k most frequent words (93.4% of tokens). The
  **dev slice** is the first 22,000 paragraphs (about 2M tokens), where all
  the settings were chosen.
- **Benchmarks** (`evalvec.py`), always over the same 20k-word vocabulary:
  - **SimLex-999:** similarity, 896 pairs covered.
  - **WordSim-353:** relatedness, 314 pairs covered.
  - **Google analogies:** 3CosAdd top-1 over the 9,470 questions inside the
    vocabulary, searching all 20k words.

  Words a method didn't produce count as zero vectors, so coverage is
  identical across methods. 95% bootstrap intervals are about ±0.065 for
  SimLex and ±0.085 for WordSim, so smaller differences are noise. Seed
  reruns vary by ±0.01-0.02.
- **Results** are appended to `results/results.tsv`: name, train time, dim,
  missing words, each score with its interval, the loss and flags, and the
  date. Peak memory goes to `results/memory.tsv`.

### Training pipeline

```
corpus.py ──> corpus.ids ──> gen.py cbow ──stdin──> python -m vowpalwabbit ──> model
                              (fresh samples         (--lrq wcK or -q wc --rank K)
                               each epoch)                       │
                                       extract.py <──────────────┘
                                       (factors read from the weights)
                                            │
                                       evalvec.py ──> results/results.tsv
```

`train.sh NAME K "GENERATOR ARGS" "VW ARGS"` runs that chain and logs time
and peak memory. `baseline.sh` does the same for gensim and PPMI+SVD.

1. **Examples (`gen.py`).** Words are written as integer ids, which VW uses
   as-is (no hashing), so each word's weights sit at a computable index.
   Each epoch resamples word2vec's subsampling (`sample` 1e-3), dynamic
   window (5) and unigram^0.75 negatives (5), the way word2vec does. The best
   format is **CBOW**: one line per target with its whole window as a bag,
   plus one line per negative that swaps in a random target:
   ```
    1 |w 29 |c 1052 3982 40 21 4245
   -1 |w 2475 |c 1052 3982 40 21 4245
   ```
   That's about 6 lines per token, against about 40 for skip-gram pairs, and
   it scores as well or better. The lines are streamed into VW's stdin, so
   nothing is written to disk and each epoch gets new samples.
2. **Model.** The prediction is a dot product between the target's factors
   (`w`) and the summed factors of the context bag (`c`), with logistic loss,
   which is word2vec's negative-sampling objective. Two VW reductions do this:
   - **`--lrq wcK`** works with any optimizer. **`--pistol`** is what makes
     it good.
   - **gd_mf (`-q wc --rank K`)** is about 4x faster but only has plain SGD.
     It needs a constant learning rate (`--power_t 0 -l 0.02`).
3. **Extraction (`extract.py`).** The factors are read straight out of the
   saved model with `Workspace.get_weight` at the computed indices, with a
   collision check. The vectors are the sum of both sides (`w` + `c`), as
   word2vec users often do, which is usually best.
4. **References (`baselines.py`):** gensim word2vec (SGNS and CBOW) and
   PPMI+SVD (context smoothing 0.75, U·S^0.5) on the same tokens.

### Scale runs (100M tokens)

`bigcorpus.py` streams the WikiText-103 parquet shard and then C4 through the
same tokenizer until 100M tokens. The vocabulary is every word seen at least
5 times (gensim's default): 189,267 of 730k distinct tokens. Every method
streams from disk and trains for one epoch at dim 100. Each run is capped at
3 GB with swap off, so a method that needs more is killed rather than
swapping. Peak resident memory is measured per process (generator and VW
separately) and includes the Python interpreter. Quality is scored on the same
20k-word benchmark vocabulary (68 benchmark words aren't in the 100M
vocabulary, for every method). Two VW-only variants:

- **Hashed (`gen.py --words`, `extract.py --hashed`).** Words are written as
  strings, so VW hashes them into `-b 17` (131k slots for 189k words): a hard
  memory cap, at the cost of collisions.
- **One pass (`gen_online.py`).** Raw text is read once, with no vocabulary
  and no counting pass, and every token goes to VW. Running counts for
  subsampling live in a fixed hashed array. Negatives come from a ring buffer
  of the last 1M tokens, thinned to word2vec's unigram^0.75. Memory is fixed
  whatever the vocabulary or corpus size.

## Results

### 10M tokens

| method | dim | train | SimLex | WordSim | analogy |
|---|---|---|---|---|---|
| **VW `--lrq` CBOW + `--pistol`, 5 epochs** (`make best`) | 100 | 8508 s | **+0.274** | **+0.653** | **0.350** |
| gensim word2vec SGNS, 5 epochs, 2 threads | 100 | 183 s | +0.273 | +0.624 | 0.319 |
| **VW gd_mf CBOW, `-l 0.02`, 5 epochs** (`make fast`) | 100 | 1253 s | +0.268 | +0.607 | 0.282 |
| PPMI + SVD | 100 | 64 s | +0.230 | +0.618 | 0.280 |
| VW `--lrq` CBOW + `--pistol`, 5 epochs | 50 | 5227 s | +0.259 | +0.604 | 0.270 |
| VW gd_mf CBOW, `-l 0.05 --l2 1e-6`, 5 epochs | 50 | 1117 s | +0.253 | +0.600 | 0.190 |
| VW gd_mf CBOW, `-l 0.02`, 5 epochs | 50 | 1154 s | +0.240 | +0.587 | 0.204 |
| gensim word2vec SGNS, 5 epochs, 2 threads | 50 | 197 s | +0.249 | +0.579 | 0.270 |
| gensim word2vec CBOW, 5 epochs, 2 threads | 50 | 62 s | +0.215 | +0.517 | 0.215 |
| PPMI + SVD | 50 | 32 s | +0.210 | +0.601 | 0.183 |
| VW gd_mf CBOW, `-l 0.05` (no L2) | 50 | 1459 s | NaN | NaN | NaN (diverged) |

VW's win at dim 100 is within noise on SimLex and WordSim. Only analogies
(+3.1 points over 9,470 questions, about 6 standard errors) is a clear gap,
and gensim ran with defaults while VW's settings were tuned on the dev slice.

### 2M-token dev slice (dim 50): what mattered

| run | train | SimLex | WordSim | analogy |
|---|---|---|---|---|
| gensim SGNS, 5 epochs (3 seeds) | 42-84 s | +0.230..0.233 | +0.414..0.436 | 0.076..0.084 |
| gensim SGNS, 1 epoch | 25 s | +0.146 | +0.144 | 0.014 |
| gensim SGNS, 10 epochs | 197 s | +0.227 | +0.502 | 0.107 |
| gensim CBOW, 5 epochs | 34 s | +0.183 | +0.286 | 0.050 |
| PPMI + SVD | 19 s | +0.176 | +0.528 | 0.108 |
| numpy GloVe reference, 20 epochs | 1104 s | +0.088 | +0.123 | 0.016 |
| **optimizer, 1 epoch, `--lrq wc50`** (CBOW rows: `make dev-optimizers`) | | | | |
| VW skip-gram, default update | 635 s | +0.095 | +0.231 | 0.006 |
| VW skip-gram, `--ftrl` | 450 s | +0.092 | +0.426 | 0.011 |
| VW skip-gram, `--pistol` | 413 s | +0.131 | +0.457 | 0.040 |
| VW CBOW, default update | 317 s | +0.097 | +0.293 | 0.001 |
| VW CBOW, `--ftrl` | 243 s | +0.094 | +0.383 | 0.005 |
| VW CBOW, `--pistol` | 224 s | +0.137 | +0.458 | 0.034 |
| **5 epochs, `--lrq wc50`** | | | | |
| VW skip-gram, default update | 2782 s | +0.137 | +0.376 | 0.019 |
| VW skip-gram, `--ftrl` | 2429 s | +0.134 | +0.450 | 0.022 |
| VW skip-gram, `--pistol` | 2268 s | +0.185 | +0.522 | 0.086 |
| VW CBOW, default update | 1451 s | +0.117 | +0.385 | 0.003 |
| **VW CBOW, `--pistol`** (3 seeds) | 1078-1150 s | +0.161..0.178 | +0.507..0.548 | 0.074..0.086 |
| ... 10 epochs | 2011 s | +0.184 | +0.557 | 0.094 |
| ... rank 100 | 2078 s | +0.172 | +0.538 | 0.127 |
| ... 2 negatives | 485 s | +0.173 | +0.557 | 0.071 |
| ... 10 negatives | 2546 s | +0.208 | +0.545 | 0.096 |
| ... window 10 | 1577 s | +0.178 | +0.551 | 0.094 |
| ... mean of context instead of sum (`sgbag`) | 970 s | +0.164 | +0.440 | 0.051 |
| **gd_mf (`-q wc --rank 50`), 5 epochs, constant lr** | | | | |
| skip-gram, `-l 0.05` | 971 s | +0.199 | +0.478 | 0.069 |
| skip-gram, `-l 0.1` | 973 s | +0.192 | +0.523 | 0.055 |
| CBOW, `-l 0.05` | 247 s | +0.213 | +0.504 | 0.064 |
| CBOW, `-l 0.02` | 241 s | +0.222 | +0.442 | 0.052 |
| CBOW, `-l 0.05`, 10 epochs | 402 s | +0.192 | +0.525 | 0.067 |
| CBOW, `-l 0.05`, rank 100 | 281 s | +0.221 | +0.498 | 0.092 |

What did **not** work (all in `results/results.tsv`; the scripts are in
`experiments/`):

- **Other optimizers:** `--sgd`, `--adaptive` alone, `--normalized` alone,
  `--adaptive --invariant`, `--power_t 0`, `-l 0.1` / `-l 2`, FreeGrad (with
  or without `--restart`), and `--coin` (NaN). PiSTOL with `--ftrl_alpha`
  0.25 or `--ftrl_beta` 2 gave NaN; alpha 2 or beta 0.25 trained but scored
  worse.
- **Other objectives** (`experiments/gen_counts.py`, `gen_docs.py`):
  - GloVe (default update, FTRL, xmax 10; PiSTOL went NaN). The numpy
    reference (`experiments/glove_ref.py`) is just as weak at this size, so
    that's the objective, not VW.
  - PPMI regression on positive cells (the earlier attempt's objective).
  - Full-matrix PPMI with sampled zeros, i.e. the SVD objective: best with
    FTRL at 0.15 / 0.41 / 0.010; PiSTOL never learned.
  - The label-dependent ranking losses `--csoaa_ldf` and `--wap_ldf`
    (`gen.py ldf`).
  - `--lda` (`experiments/train_lda.sh`): with stop words dropped it reaches
    SimLex ~0.15, WordSim 0.04-0.21.
- **Other knobs:**
  - A centred initialization in place of lrq's non-negative one
    (`experiments/init_model.py`).
  - Post-hoc centring, top-component removal and per-dimension scaling.
  - `--lrqdropout`.
  - gd_mf with VW's default schedule, or with `-l 0.5`.
- **Not pursued, with reasons:**
  - `--lrqfa` runs fine, but with only two namespaces field-aware factors are
    the same model as `--lrq`.
  - `--nn` puts the inputs through a shared hidden layer, which can't
    represent a word x context dot product.
  - `--bfgs`, `--OjaNewton` and `--svrg` failed in the earlier attempt and
    don't fit streamed training (bfgs needs `--passes` over a cache).

### 100M tokens: memory, time and quality

| method | corpus | vocab | peak memory | train | SimLex | WordSim | analogy |
|---|---|---|---|---|---|---|---|
| gensim SGNS, 2 threads | 10M | 20k | 137 MB | 36 s | +0.224 | +0.419 | 0.113 |
| gensim CBOW, 2 threads | 10M | 20k | 137 MB | 14 s | +0.178 | +0.285 | 0.058 |
| VW gd_mf CBOW, `-b 18` | 10M | 20k | 332 MB (VW 293) | 312 s | +0.204 | +0.409 | 0.091 |
| VW lrq CBOW + PiSTOL, `-b 25` | 10M | 20k | 124 MB (VW only) | not timed (memory probe) | | | |
| PPMI + SVD | 10M | 20k | 889 MB | 38 s | +0.230 | +0.618 | 0.280 |
| gensim SGNS, 2 threads | 100M | 189k | 421 MB | 439 s | **+0.338** | +0.650 | **0.458** |
| gensim CBOW, 2 threads | 100M | 189k | 421 MB | 154 s | +0.294 | +0.618 | 0.399 |
| **VW gd_mf CBOW, `-b 18`** | 100M | 189k | **347 MB** (VW 292) | 3591 s | +0.337 | **+0.661** | 0.422 |
| VW gd_mf CBOW, hashed words, `-b 17` | 100M | 189k | **228 MB** (VW 165) | 3526 s | +0.326 | +0.642 | 0.363 |
| VW gd_mf CBOW, **one pass, no vocabulary**, `-b 17` | 100M | 730k (all) | 291 MB (VW 165) | 3667 s | +0.318 | +0.656 | 0.362 |
| VW lrq CBOW + PiSTOL, `-b 26` | 100M | 189k | 777 MB (VW only) | not trained (about 5 h estimated) | | | |
| PPMI + SVD | 100M | 189k | over 3 GB, killed after 25 s | | | | |

All rows are one epoch at dim 100, with gd_mf flags
`--loss_function logistic --noconstant --power_t 0 -l 0.02`. The lrq rows are
memory probes (`touch_probe.sh`; see gotcha 18).

**How each method's memory grows.** Apart from PPMI, none depends on corpus
length:

- **gensim:** about 120 MB plus **1.7 KB per vocabulary word** at dim 100
  (two 400-byte vectors plus vocabulary bookkeeping). Before pruning, its
  vocabulary scan keeps a count for every distinct token; `max_vocab_size`
  caps that.
- **VW gd_mf:** a fixed table of 2^b slots x 2^ceil(log2(2K+1)) floats x 4
  bytes: 1 KB per slot at rank 100, 512 bytes at rank 50. gd_mf initializes
  every weight randomly, so the whole table is resident from the start. With
  integer ids, 2^b only has to cover the vocabulary once (gotcha 10): 189k
  words fit in `-b 18` (256 MB).
- **VW lrq:** 2 KB per word per namespace at K=100 with PiSTOL (128 slots x
  4 floats of optimizer state x 4 bytes), and the table must fit both
  namespaces (`-b 26`, 1.1 GB reserved). Resident memory grows as words first
  appear and levels off at 777 MB.
- **The VW generators:** word counts and the negative table, 39-63 MB. For
  `gen_online.py` it's 126 MB, mostly its 1M-token buffer.
- **PPMI + SVD:** the co-occurrence matrix lives in memory. My
  implementation counts all pairs at once; a chunked counter would need less,
  but the matrix still grows with corpus and vocabulary.

**Memory alone is not a reason to choose VW:** both tools stream, and both
scale with the vocabulary. VW's lever is that `-b` is a hard cap. In the
hashed run, 76% of words share a slot with another word, yet similarity stays
within noise and only analogies drop (6 points). Most collisions pair a
frequent word with a rare one, which barely moves the frequent word's vector:
only 14% of the 20k benchmark words share a slot with another top-20k word.
The one-pass run scores the same as the two-pass hashed run.

### What the vectors look like

Nearest neighbours, VW gd_mf at 100M tokens (gensim's are very similar):

```
     dog: cat, pet, puppy, sitter, dogs, grooming, cats, kitten
  france: spain, italy, belgium, portugal, germany, sweden, austria, netherlands
   happy: thrilled, delighted, thankful, glad, grateful, pleased, excited
 quickly: easily, swiftly, efficiently, effectively, instantly, eventually, rapidly
    said: says, mr, spokeswoman, trudeau, krell, spokesman, spokesperson
```

VW's neighbours lean topical where gensim's skip-gram ones are more
grammatical: gensim's **said** gives told, explained, replied. That fits VW
trailing on analogies, half of which are grammatical. Summing both sides
lets phrases bleed in (**good** → luck). Against published 300-d vectors
trained on 6B-100B tokens (SimLex ~0.37-0.45, WordSim ~0.65-0.75, analogies
~70-75%), these are small-corpus vectors. The gap is almost all data. Our
analogy set is a 9,470-question subset searched over 20k words, so it isn't
directly comparable.

## When VW is the better tool

- **The embedding is part of a supervised model.** lrq and gd_mf learn
  factorized interactions between any two namespaces from your own labels:
  query x document, user x item, word x tag. Linear features sit in the same
  model. gensim only learns from co-occurrence.
- **You need side information, weights or another loss.** Extra namespaces
  (POS tags, document source), per-example importance weights, and weighted
  pairs from non-text data (graphs, sessions, counts) use the same input
  format. Losses: logistic, squared, hinge, quantile.
- **Memory must stay under a fixed cap whatever the vocabulary, or the text
  is a stream you can read only once** (the hashed and one-pass rows).
- **The model keeps learning after deployment.** `-i model -f model`
  continues on new data with the optimizer state kept, and with hashing a new
  word just gets a slot. gensim can add words, but its learning-rate schedule
  assumes one planned run. *Not benchmarked here.*
- **VW is already in your stack**, and the embeddings should share its
  feature hashing.

## Not tried yet

Roughly by expected payoff:

1. **Use the vectors as features in this repo's POS tagger.** That's the
   real test of "usable"; everything above is intrinsic benchmarks. The
   tagger already has suffixes, prefixes and shapes (`conllu-to-vw.py`), so
   the gain would have to come from distributional knowledge of rare words.
2. **Learn the embeddings inside the tagger** (an lrq word x tag
   interaction), VW's main advantage over gensim.
3. **Extra information in the vectors.** Features in one namespace sum, so
   annotations can be added beside the word. `data/wikitext/wiki-10m.conllu`
   has spaCy's lemma, POS, morphology and dependencies for the same text
   (with spaCy's tokenization, so the corpus and baselines would need
   rebuilding from it). Four ways to use them:
   - **Additive features**, spaCy MultiHashEmbed style:
     `|w ducks lem=duck pos=NOUN suf=cks`.
   - **Sense keys**, sense2vec style: `duck|NOUN` plus the plain word.
   - **Dependency contexts**: Levy & Goldberg 2014 report better similarity
     this way, i.e. SimLex.
   - **Multi-task** word → tag examples, with `-q wc -q wt` sharing the `w`
     side.
4. **Subword features**, fastText's trick: character n-grams in the target's
   namespace, giving vectors for rare and unseen words and likely better
   grammatical analogies.
   - **Cost:** about 2x slower on the target side (10x on the context side).
   - **Memory:** n-grams need a big shared table (about 2 GB at fastText's
     2M buckets, rank 100), which works against the fixed-memory case.
   - **Built-in options:** `--affix` gives prefixes and suffixes only;
     `--ngram` is token n-grams, not character n-grams.
5. **Position-aware context namespaces** (left-1, right-1, ...), as in
   structured skip-gram, for grammatical analogies.
6. **Two hashes per word** (a sum of two slots, as in spaCy's bloom
   embeddings) to win back the hashed runs' analogy loss at the same memory.
7. **Multi-core training** with allreduce (`--span_server` exists, but the pip
   wheel lacks the `spanning_tree` server), or shard training with periodic
   weight averaging.
8. **Large-corpus settings** for both tools: `sample` 1e-4 or 1e-5, a
   decaying learning rate, rank 200-300, more than one epoch.

## Gotchas

### Getting good vectors out of VW

1. **The optimizer decides the outcome.** With VW's default update
   (adaptive + normalized + invariant), lrq embeddings are mediocre.
   `--pistol` is clearly best, then `--ftrl`, for skip-gram and CBOW alike.
2. **Use CBOW-style examples:** one line per target with the whole window
   as a bag. Sum the context; the mean (`sgbag`) is worse.
3. **Use the sum of both sides' factors** (w + c). For CBOW the context side
   alone is usually worst.
4. **gd_mf is about 4x faster than lrq, but needs plain-SGD tuning.**
   - It accepts only its own SGD (`--pistol` gives `unrecognised option`).
   - The default schedule (`-l 0.5 --power_t 0.5`) learns little.
   - A constant `-l 0.05 --power_t 0` works on 2M tokens but **diverged to
     NaN after ~67M examples** on 10M tokens.
   - `-l 0.02`, or `-l 0.05 --l2 1e-6`, is stable, including for 479M
     examples at 100M tokens.

### VW behaviour

5. **`--ignore_linear X` on an `--lrq` namespace freezes that side's
   factors.** lrq adds its interaction features *into* the namespace, so
   ignoring the namespace's linear terms also ignores the lrq features. Two
   runs on different data gave identical `c` vectors.
6. **lrq's initialization is non-negative** (average cosine between words
   starts at 0.39). A centred initialization didn't help, so this is a
   curiosity, not a cause.
7. **Several settings fail silently with all-NaN factors.** `--coin`,
   `--pistol --ftrl_alpha 0.25` and `--pistol --ftrl_beta 2` give NaN with no
   warning; the loss just sits at ln 2. Other divergences print
   `NAN prediction` **on stdout, one per example**: one run filled 17 GB of
   log. `train.sh` counts them instead, and `evalvec.py` checks the vectors
   for NaN.
8. **Naming one of `--adaptive`, `--normalized` or `--invariant` turns the
   other two off.** `--invariant` alone is a *subset* of the default.
9. **lrq spaces word slots out by STEP = 2^ceil(log2(K+1)).** The linear
   weight of integer feature *i* in namespace *ns* is at
   `((hash_space(ns) + i) * STEP) mod num_weights`, and its K factors are in
   the next K slots. With string features, K=50 and `-b 22`, there are only
   65,536 slots and ~5% of a 3,000-word vocabulary collides. Integer ids make
   the layout predictable; `extract.py` checks for overlaps.
10. **gd_mf keeps factors inside each word's slot.** Read them with
    `get_weight(i, offset)`: offsets 1..K are the left namespace's factors and
    K+1..2K the right's, with `i = (hash_space(ns) + id) mod num_weights`.
    - **Memory:** 2^b x 2^ceil(log2(2K+1)) floats x 4 bytes, so `-b 25` at
      K=50 would need 16 GB.
    - **Slot sharing:** each side has its own offsets, so a left word and a
      right word in one slot share only the linear weight. 2^b therefore has
      to cover the vocabulary once, not twice. This was checked by rebuilding
      VW's predictions from the extracted weights.
    - **lrq is different:** both namespaces use the same factor slots there,
      so their ranges must not overlap.
11. **The named dumps are the wrong tool.** `--invert_hash` and
    `--dump_json_weights_*` keep one name per slot (silently dropping
    colliding words), cost time proportional to the number of examples, and
    work in a single pass only.
12. **`Workspace.get_weight_from_name` ignores lrq's slot step** and returns
    the wrong weight (usually 0) for lrq models.
13. **`python -m vowpalwabbit` ignores piped input** unless given
    `-d /dev/stdin`. It reports 0 examples and exits.
14. **`--passes` replays the cache in a fixed order**, and `--invert_hash`
    refuses a cache. Streaming freshly generated epochs avoids both.
15. **VW `--lda` collapses on raw text:** one topic takes the stop words.
16. **The pip wheel has no `spanning_tree`**, so multi-process training isn't
    available from this install. Training is single-threaded, which is most
    of the speed gap.
17. **Namespace hashes decide where integer ids land.** VW adds
    `hash_space(ns)` to each id, so two namespaces' id ranges can overlap
    even when 2^b is big enough. With `w` and `c` and 189k words, lrq at K=100
    would need `-b 28` instead of `-b 26`. Renaming the context namespace
    (`gen.py --ctx-ns cb`, `extract.py --ns w,cb`) moves it clear.
18. **Resident memory understates lrq until every word has been seen.** VW
    reserves the table up front, but the OS counts only pages that have been
    written. A 2-minute run showed 37 MB. `touch_probe.sh` feeds one example
    per word to measure the real footprint.
19. **VW clips predictions to the label range** ([-1, 1] here), so a
    prediction rebuilt from the weights can differ from `predict()` by the
    clipped amount.
20. **gensim has no Python 3.14 wheel**, hence this folder's own uv project
    on Python 3.12-3.13.

## The earlier attempt

A first attempt by another agent concluded "don't use VW for this". Its
write-up and results are kept in `results/earlier-attempt/`; its scripts were
superseded by the ones here. What it got wrong:

| problem | fix here |
|---|---|
| skip-gram pairs and the PPMI control used different tokenizers (`london,` vs `london`) | one tokenizer (`corpus.py`) feeds every method |
| `-b 22` with `--lrq wc50` gave ~5% of words a shared vector, and the named dump dropped them | integer word ids + `-b 25`; `extract.py` refuses overlapping weight blocks |
| vectors read through `--invert_hash`/JSON dumps: slow, single pass only | factors read straight from the saved model (4 s for 20k words) |
| epochs baked into one big file; only 1 epoch for the fixed skip-gram | `gen.py` streams freshly sampled epochs into VW |
| only compared against PPMI+SVD on 500k words | gensim word2vec, PPMI+SVD and a numpy GloVe on the same 2M, 10M and 100M tokens |
| no confidence intervals; "improvements" of +0.03 were noise | bootstrap 95% CIs, plus seed reruns |
| never tried `--pistol`, nor gd_mf with a working learning rate | optimizer and learning-rate sweeps |

## Files

| file | what it does |
|---|---|
| `Makefile` | every step and recorded run (see [Running it](#running-it)) |
| `paths.py` | where data, work files and results live (overridable by environment variable) |
| `corpus.py` | tokenize the 10M-token corpus once: `corpus.txt`, `corpus.ids`, `corpus.vocab.txt`, `vocab.tsv` |
| `bigcorpus.py` | the 100M-token WikiText + C4 corpus, vocabulary = count >= 5 |
| `gen.py` | stream VW examples (`cbow`, `sgns`, `sgbag`, `ldf`), ids or words (`--words`) |
| `gen_online.py` | one-pass CBOW examples with no vocabulary |
| `train.sh` | generator -> VW -> `extract.py` -> `evalvec.py`, logging time and peak memory |
| `baseline.sh` | the same for `baselines.py` (gensim word2vec, PPMI+SVD) |
| `extract.py` | read lrq or gd_mf factors from a saved model (ids or hashed words), with collision checks |
| `evalvec.py` | SimLex / WordSim (bootstrap CIs) / analogies -> `results/results.tsv` |
| `touch_probe.sh` | VW's resident memory once every word's weights exist |
| `neighbours.py` | nearest neighbours for a few words |
| `summarize.py` | markdown table of chosen `results/results.tsv` rows |
| `experiments/` | the dead ends above: GloVe/PPMI objectives (`gen_counts.py`, `glove_ref.py`), LDA (`gen_docs.py`, `train_lda.sh`, `extract_lda.py`), centred init (`init_model.py`) |
| `results/results.tsv` | every evaluated run: time, scores with CIs, loss, flags, date |
| `results/memory.tsv` | peak memory and time of the scale runs and probes |
| `results/earlier-attempt/` | the earlier attempt's write-up and results |
