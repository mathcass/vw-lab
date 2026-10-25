# Vowpal Wabbit 9.11.9: bugs and gotchas

Collected from the three projects in this repo. All of them use the
`vowpalwabbit` 9.11.9 wheel from PyPI (Linux x86_64). Each item says which
project found it; that project's README or findings have the details. The
three bugs with a standalone demo are in [`bug-repros/`](bug-repros/README.md).

## Bugs

Each of these gave a wrong result or a crash, with no warning.

| What | Found in | Workaround |
|---|---|---|
| `--csoaa_ldf` / `--wap_ldf` **segfault** when predicting on multi-line examples with no cost labels | `decisions/` (bug-repros 1) | give every option a dummy `i:0` cost |
| `--csoaa_ldf mc --csoaa_rank`: `predict()` scores are **shifted by one** (the shared line counts as an action, the last option is dropped) | `decisions/` (bug-repros 2) | read each option's `get_partial_prediction()` |
| Reloading with an extra `-q` (`-i model.vw -q cd`) **silently drops the model's saved interactions** | `decisions/` (bug-repros 3) | restate every saved `-q` when adding one |
| `--search_task dep_parser` leaves the **last token of most sentences unattached** (`9999999:9999999`) | `nlp/` | attach it to the root (`evaluate.attach_unattached`): +5 LAS |
| `--search_span_bilou` returns labels in the internal BILOU numbering | `nlp/` | use plain BIO with `--search_task sequence` |
| `Workspace.get_weight_from_name` ignores `--lrq`'s slot step and returns the wrong weight (usually 0) | `word2vec/` | compute the index yourself (`word2vec/extract.py`) |
| `--coin`, `--pistol --ftrl_alpha 0.25` and `--pistol --ftrl_beta 2` train to **all-NaN weights** with no warning (loss sits at ln 2) | `word2vec/` | check outputs for NaN |
| A diverging run prints `NAN prediction` **on stdout once per example** (one run wrote 17 GB of log) | `word2vec/` | count the lines instead of keeping them |

## Surprising behaviour

- **The wheel has no `vw` binary.** `python -m vowpalwabbit` takes the same
  flags but ignores piped input unless given `-d /dev/stdin`; without that it
  reports 0 examples and exits. The wheel also has no `spanning_tree`, so
  training is single-threaded. (`nlp/`, `word2vec/`)
- **VW 8.x models load in 9.x without error but predict differently** (56%
  instead of 74% on POS dev). Score old models with the VW that saved them.
  (`nlp/`)
- **The default 18-bit table is shared by every class**, so a 50-tag tagger
  collides badly: `-b 24` alone took POS from 72% to 87%. (`nlp/`)
- **`--l1` is applied on every update,** so the same value shrinks a model
  far more when training makes more updates (more data, rollouts, more
  passes). Retune it whenever the data or search settings change. (`nlp/`)
- **`--predict_only_model` halves the file** at identical accuracy, and only
  nonzero weights cost anything. Without it VW 9 saves optimizer state.
  (`nlp/`)
- **Naming one of `--adaptive`, `--normalized` or `--invariant` turns the
  other two off;** the default is all three. (`word2vec/`)
- **`--ignore_linear X` on an `--lrq` namespace freezes that side's
  factors.** (`word2vec/`)
- **`--passes` replays the cache in a fixed order**, and `--invert_hash`
  refuses a cache. (`word2vec/`)
- **`--invert_hash` and `--dump_json_weights_*` keep one name per slot**,
  silently dropping colliding features, and take time proportional to the
  number of examples. (`word2vec/`)
- **Namespace hashes decide where integer feature ids land,** so two
  namespaces' ids can collide even in a big table. (`word2vec/`)
- **Resident memory understates a model's size** until every slot has been
  written. (`word2vec/`)
- **Predictions are clipped to the label range,** so a prediction rebuilt
  from the weights can differ from `predict()`. (`word2vec/`)
- **`--probabilities` output is very flat** (0.58 at most on a perfectly
  separable toy), so calibrate confidence separately. (`decisions/`)
- **A `--csoaa_ldf` model can't be loaded with `--cb_adf`.** The error says
  "model content is corrupted"; it isn't. (`decisions/`)
- **`--cb_adf` allows only one costed option per example,** and
  `--csoaa_ldf mc` with costs other than 0/1 switches from classification to
  regression. (`decisions/`)

## What worked across projects

- **Teacher forcing** (`--search_rollin ref --search_rollout none`) beat VW's
  default search settings for tags, entities, sentences and parses, and
  trained 7-10x faster. (`nlp/`)
- **A big table plus `--l1`** beats a small table: `-b 26` keeps collisions
  rare and L1 drops the weights that don't matter. (`nlp/`)
- **`--pistol`** was the best optimizer for embeddings (`--lrq`); `gd_mf`
  only takes plain SGD and needs a constant, small learning rate.
  (`word2vec/`)
- **Feature crosses learned from words memorize:** they help on seen domains
  and add nothing on new ones. (`decisions/`)
