# Vowpal Wabbit for NLP

Can [Vowpal Wabbit](https://vowpalwabbit.org/) (VW) learn NLP models cheaply,
in training time and in model size? This repo has two projects that try,
sharing one data folder:

| Folder | What it does |
|---|---|
| [`nlp/`](nlp/README.md) | A POS tagger trained on UD English-EWT, and VW copies of spaCy's tagger, sentence splitter, entity tagger and dependency parser, chained into a pipeline on raw text |
| [`word2vec/`](word2vec/README.md) | Word embeddings trained with VW, against gensim word2vec and PPMI+SVD |

## Results

**`nlp/`.** For POS tagging, yes; for a whole spaCy-style pipeline, no. Each
piece trains quickly on a CPU, but together they don't beat spaCy's
`en_core_web_sm` (~12 MB) on accuracy, size or speed. Test scores; "vs spaCy"
rows are agreement with `en_core_web_sm` on the same text:

| Task | Model size | Train time | Test |
|---|---|---|---|
| POS tags (gold UD XPOS) | 8.4 MB | ~35 s | 93.5% (UDPipe 1: 93.9%) |
| POS tags, smallest | 0.40 MB | 33 s | 90.9% |
| POS tags vs spaCy | 8.3 MB | ~1 min | 94.5% |
| Sentence splitting vs spaCy | 0.62 MB | 29 s | 95.5 F1 (`. ! ?` rule: 85.7) |
| Named entities vs spaCy | 9.3 MB | 14 min | 71.8 F1 |
| Parser vs spaCy (given spaCy's tags) | 4.9 MB | 12 min | 85.7 UAS / 82.6 LAS |
| Whole pipeline from raw text vs spaCy | 23.2 MB | ~28 min | 94.4% tags, 70.6 entity F1, 77.7 LAS |

The pipeline runs at 4,400 tokens/s against spaCy's 7,200, mostly because of
the Python glue around VW.

**`word2vec/`.** VW can train word vectors as good as word2vec's, but slowly.
On 100M tokens (one epoch), VW's `gd_mf` matches gensim skip-gram on word
similarity (SimLex 0.337 vs 0.338) and trails on analogies (42.2% vs 45.8%),
taking 60 minutes against gensim's 7. Its niche is fixed memory and
single-pass training on a stream.

## Layout

```
README.md, Makefile   this overview, and the shared downloads
wikitext-to-text.py   WikiText-103 sample (used by both projects)
pyproject.toml        uv project for the shared data prep (pyarrow only)
data/                 all downloads and work files (not committed)
nlp/                  its own Makefile, uv project (Python 3.14) and README
word2vec/             its own Makefile, uv project (Python 3.12-3.13) and README
```

Each project's Makefile runs from its own folder (or `make -C nlp TARGET`
from here) and calls the root Makefile for shared downloads when it needs
them. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

## Shared data

Downloads are pinned to a dataset revision and checked against a sha256
(except UD, which comes from the LINDAT release).

| File in `data/` | Source | License | Used by |
|---|---|---|---|
| `ud-treebanks-v2.7/` (`make download`, then untar) | [UD v2.7](https://universaldependencies.org/), English treebanks | CC BY-SA / CC BY-NC-SA, per treebank | `nlp/` |
| `c4/c4-train-00000.json.gz` | [C4](https://huggingface.co/datasets/allenai/c4) en, first training shard (319 MB) | ODC-BY | `nlp/` (NER text), `word2vec/` (100M corpus) |
| `wikitext/train-00000.parquet` | [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) raw, first training shard (157 MB) | CC BY-SA 4.0 | `word2vec/` |
| `wikitext/wiki-10m.txt` | 10M-word seed-0 sample of the shard | CC BY-SA 4.0 | `word2vec/` (10M corpus), `nlp/` (earlier NER recipe) |

Each project's README describes the files it builds from these.
