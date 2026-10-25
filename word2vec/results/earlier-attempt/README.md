> **Kept for the record.** This is the first attempt (by another agent, in a
> scratch directory since removed). Its scripts were superseded by the ones in
> `word2vec/` and are not kept, so the commands and paths below no longer run.
> What it got wrong, and the fixes, are in
> [the main README](../../README.md#the-earlier-attempt).

# `.pi/embeddings` — VW word-embedding proof of concept

A self-contained experiment: can Vowpal Wabbit train word embeddings? It lives
in `.pi/` with its own uv project and venv so it never touches the main repo's
environment or data. See `narrative.md` for the story and `results.tsv` for
every run.

**Short answer:** mechanically yes (`--lrq` + the JSON weight dump), and it
learns coarse/synthetic structure well, but on real text it is a poor embedding
trainer — a count-based PPMI+SVD baseline beats it on identical data. Don't use
VW for this; use PPMI+SVD (or an external word2vec) and feed the vectors in as
features.

Round 2 (see `narrative.md` and `results2.tsv`) fixed a VW bug I had been
tripping over — `--ignore_linear` on either namespace stops `--lrq` learning
that namespace entirely — and then tried every remaining knob: count-based
objectives (PPMI / log-count / Poisson), ten optimizers (FTRL and `--invariant`
fit best), epochs, rank, init and seed. VW's best still loses to the 23-second
PPMI+SVD baseline (SimLex +0.11 vs +0.20, WordSim +0.11 vs +0.47).

## Environment

```
cd .pi && uv sync          # creates .pi/.venv (vw 9.11.9, numpy, pyarrow)
```

All commands below run from the repo root and use `.pi/.venv/bin/python`, never
the repo's `uv run`.

## Scripts

| file | what it does |
|---|---|
| `pairs.py` | corpus (CoNLL-U or `# text = `) -> VW skip-gram pairs `+1/-1 \|w TARGET \|c CONTEXT` |
| `train.sh` | `--lrq wcK` training + `--invert_hash` dump (uses `.pi/.venv`) |
| `extract.py` | parse `--invert_hash` -> `word<TAB>vec` / `.npz` |
| `extract_json.py` | parse the JSON weight dump -> `.npz` (matches `extract.py`) |
| `evaluate.py` | nearest neighbours + kNN UPOS purity vs a random baseline |
| `similarity.py` | SimLex-999 / WordSim-353 Spearman (the real intrinsic metric) |
| `similarity_metric.py` | raw cosine vs the factorization-natural `cos(E_c e_w)` |
| `pairs_count.py` | count-based examples for PPMI / log-count / Poisson objectives |
| `ppmi_svd.py` | the count-based PPMI+SVD control |
| `wiki_text.py` | sample a WikiText-103 parquet shard to `# text = ` lines |

## Minimal end-to-end

```bash
PY=.pi/.venv/bin/python
# 1. pairs from any CoNLL-U
$PY .pi/embeddings/pairs.py --conllu data/spacy/train.conllu \
    --out .pi/embeddings/work/demo.vw --max-tokens 100000 --vocab 4000 \
    --window 5 --negatives 5 --subsample 1e-4 --epochs 3
# 2. train rank-50 lrq factors and dump them with names
$PY -m vowpalwabbit -d .pi/embeddings/work/demo.vw \
    --lrq wc50 --noconstant --ignore_linear c \
    --loss_function logistic --holdout_off -b 22 \
    --dump_json_weights_experimental .pi/embeddings/work/demo.json \
    --dump_json_weights_include_feature_names_experimental
# 3. extract and look at neighbours
$PY .pi/embeddings/extract_json.py .pi/embeddings/work/demo.json .pi/embeddings/work/demo
$PY .pi/embeddings/evaluate.py .pi/embeddings/work/demo data/spacy/train.conllu --side w
```

## Gotchas found

- **`--ignore_linear` on either namespace stops `--lrq` learning that
  namespace.** Loss never drops below the label variance, even after 20 epochs
  on a clean rank-5 matrix (recon corr 0.07-0.11 vs 0.99 without). Keep linear
  terms on both sides. Ignoring *both* also emits `update is NAN`.
- **`--lrq` needs many epochs** (~20) to fit; 1 epoch reaches only corr 0.71 on
  a clean rank-5 matrix.
- `--invert_hash` does not name `-q a b --rank K` factors (it prints raw
  pointers). Use `--lrq`, and prefer the JSON dump for names.
- Both `--invert_hash` and the JSON dump are **single-pass only**: `--invert_hash`
  refuses a cache file and the JSON dump hangs under `--passes`. `pairs.py`
  bakes the epochs into the pairs file instead.
- The named dumps are O(examples) (90 s per 1M lines); `--readable_model` is
  O(table) and fast (6 s) but hashed. `--audit` prints names + indices but its
  output is ~40 MB per example.
