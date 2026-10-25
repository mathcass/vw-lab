# vw-lab

Where does [Vowpal Wabbit](https://vowpalwabbit.org/) (VW) hold up against
modern tools, in accuracy, training time and model size? Three projects test
it against a strong baseline each:

| Folder | Question | Baseline |
|---|---|---|
| [`nlp/`](nlp/README.md) | Can VW learn a POS tagger, and copy spaCy's tagger, sentence splitter, entity tagger and parser? | spaCy `en_core_web_sm`, published UD taggers |
| [`word2vec/`](word2vec/README.md) | Can VW train word embeddings? | gensim word2vec, PPMI+SVD |
| [`decisions/`](decisions/README.md) | Can VW answer Jev-style decisions (yes/no, pick one, score)? | OpenAI's Decisions API |

[`VW-NOTES.md`](VW-NOTES.md) collects the VW 9.11.9 bugs and gotchas all
three ran into; [`bug-repros/`](bug-repros/README.md) has standalone demos
of three of them.

## Results

| Project | Verdict |
|---|---|
| `nlp/` | **Tagging yes, a whole pipeline no.** Each piece trains quickly on a CPU, but together they don't beat spaCy's small pipeline on accuracy, size or speed. |
| `word2vec/` | **Yes, but slowly.** As good as word2vec at equal data, 8x slower; its niche is fixed memory and single-pass training on a stream. |
| `decisions/` | **Not zero-shot; yes as a cost-cutter for keyword-style tasks.** Behind OpenAI, VW takes the traffic it is sure of. |

**`nlp/`.** Test scores; "vs spaCy" rows are agreement with
`en_core_web_sm` (~12 MB) on the same text:

| Task | Model size | Train time | Test |
|---|---|---|---|
| POS tags (gold UD XPOS) | 8.4 MB | ~35 s | 93.5% (UDPipe 1: 93.9%) |
| POS tags, smallest | 0.40 MB | 33 s | 90.9% |
| POS tags vs spaCy | 8.3 MB | ~1 min | 94.5% |
| Sentence splitting vs spaCy | 0.62 MB | 29 s | 95.5 F1 (`. ! ?` rule: 85.7) |
| Named entities vs spaCy | 9.3 MB | 14 min | 71.8 F1 |
| Parser vs spaCy (given spaCy's tags) | 4.9 MB | 12 min | 85.7 UAS / 82.6 LAS |
| Whole pipeline from raw text vs spaCy | 23.2 MB | ~28 min | 94.4% tags, 70.6 entity F1, 77.7 LAS |

The pipeline is slower than spaCy, mostly because of the Python glue around
VW.

**`word2vec/`.** On 100M tokens (one epoch), VW's `gd_mf` matches gensim
skip-gram on word similarity (SimLex 0.337 vs 0.338) and trails on analogies
(42.2% vs 45.8%), taking 60 minutes to train against gensim's 7.

**`decisions/`.** Macro-F1, fold 0 (VW trains in 1.5-2.5 min per model):

| Set | VW | OpenAI, no training | Random |
|---|---|---|---|
| Domains VW trained on | 0.600 | 0.677 | 0.359 |
| Domains VW never saw | 0.293 | 0.640 | 0.325 |
| Lockbox (a separate benchmark) | 0.290 | 0.617 | 0.209 |

Trained on OpenAI's answers and gated by its own confidence, VW takes about
a third of the traffic for a small loss (0.641 to 0.629); for spam it takes
89%, at better quality than OpenAI's.

## Layout

```
README.md, Makefile   this overview, and the shared downloads (UD, C4, WikiText)
VW-NOTES.md           VW 9.11.9 bugs and gotchas from all three projects
bug-repros/           standalone demos of three VW bugs
wikitext-to-text.py   WikiText-103 sample (used by nlp/ and word2vec/)
pyproject.toml        uv project for the shared data prep (pyarrow only)
data/                 shared downloads and work files (not committed)
nlp/                  own Makefile, uv project (Python 3.14), README, hill-climb records
word2vec/             own Makefile, uv project (Python 3.12-3.13), README
decisions/            own uv project (Python 3.12), README, FINDINGS, REPRODUCE
```

Each project is its own [uv](https://docs.astral.sh/uv/) project and runs
from its own folder. `nlp/` and `word2vec/` share `data/` and call the root
Makefile for downloads; `decisions/` keeps its data and runs in its own
`data/` and `runs/` (also not committed), since nothing else uses them.

## Shared data

Every download is checked against a sha256; C4 and WikiText are also pinned
to a dataset revision. Each project's Makefile fetches what it needs.

| File in `data/` | Source | License | Used by |
|---|---|---|---|
| `ud-treebanks-v2.7/` (`make download`; the English treebanks only) | [UD v2.7](https://universaldependencies.org/) from [LINDAT](http://hdl.handle.net/11234/1-3424) (414 MB) | CC BY-SA / CC BY-NC-SA, per treebank | `nlp/` |
| `c4/c4-train-00000.json.gz` | [C4](https://huggingface.co/datasets/allenai/c4) en, first training shard (319 MB) | ODC-BY | `nlp/` (NER text), `word2vec/` (100M corpus) |
| `wikitext/train-00000.parquet` | [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) raw, first training shard (157 MB) | CC BY-SA 4.0 | `word2vec/` |
| `wikitext/wiki-10m.txt` | 10M-word seed-0 sample of the shard | CC BY-SA 4.0 | `word2vec/` (10M corpus), `nlp/` (earlier NER recipe) |

`decisions/` downloads its two Hugging Face datasets itself, pinned to a
revision (see its REPRODUCE.md).

## On a new machine

Needs `make`, `curl`, `uv` and GNU `time`; uv installs a Python in each
project's pinned range and the locked packages. Every step downloads what it
needs. Run one job at a time on a 16 GB machine. Times are wall time from a
fresh clone on a 4-core CPU (2026-10-24/25), downloads included:

```
cd nlp
make prepare train eval                         # gold POS tagger: 2 min (UD download, train 34 s)
make spacy-pos-prepare spacy-pos-train spacy-pos-eval   # 1.5 min
make sent-prepare sent-train sent-eval          # 2 min
make ner-prepare                                # 13 min (C4 download, spaCy labels 5.1M tokens)
make ner-train ner-eval                         # 12.5 min
make dep-prepare dep-train dep-eval             # 8.5 min
make pipeline-eval c4-eval                      # 7.5 min
cd ../word2vec
make setup benchmarks corpus                    # 30 s (WikiText download)
make fast                                       # 21 min
make baselines                                  # 6 min
cd ../decisions                                 # REPRODUCE.md; data and fold-0 model: 4 min
```

That fresh clone rebuilt every `nlp/` model and data file byte for byte,
word2vec's VW and PPMI+SVD scores exactly (gensim, on 2 threads, varies a
little between runs), and the decisions data, splits, fold-0 model and
predictions byte for byte. The OpenAI steps in `decisions/` cost about
$1.25 and need a key; they weren't rerun. Another CPU may differ in the last
decimal places of VW's floating-point math.
