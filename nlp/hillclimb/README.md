# Hill-climb records

How the settings in `../run.sh` were chosen. Each climb trained one config at
a time, scored it on dev, and kept the best; test was scored once per final
model. The [main README](../README.md) has the tables; these folders have
the full story and every run.

| Folder | Climb | Story | Every run |
|---|---|---|---|
| `vw-tagger/` | first POS climb: features, `-b 26` (the ~770 MB model) | `narrative.md` | `results.tsv` |
| `vw-tagger-small/` | small POS model, spaCy POS, then NER (rounds 1-8) | `narrative.md` | `results.tsv`, `results-ner.tsv`, `test-results*.tsv` |
| `vw-parser/` | `dep_parser` settings | `narrative.md` | `results.tsv`, `test-results.tsv` |

Each folder has its harness: `hc.py` (train one config, score dev, append a
row), `round.sh CONFIG.cfg` (run a file of configs one after another, skipping
names already in the results), and the `.cfg` files of each round. Config
lines are `NAME|FEATS|PASSES|VW ARGS`.

## Rerunning a config

The harness reads data from the repo's `data/`, runs VW from `nlp/`'s uv
project, and writes each run's data and model to `data/hillclimb/`
(`HC_SCRATCH` overrides it). It appends to the results files here, so
rerunning a recorded name is skipped; use a new name.

```
cd nlp && make prepare           # data/work, the gold POS data every tagger climb uses
cd hillclimb/vw-tagger-small
python3 hc.py rerun-final w,l,s4,p3,h,f,n1 3 \
  --search_neighbor_features -1:w,1:w,-2:w,2:w,-1:s,1:s,-1:h,1:h \
  --search_rollin ref --search_rollout none -b 26 -q ln --l1 3e-7 --predict_only_model
```

`HC_DATA` points `hc.py` at another work folder. These are the ones the
records name, and how to rebuild them:

| `HC_DATA` in the records | Rebuild |
|---|---|
| `data/work` (default) | `make prepare` |
| `data/spacy-ewt-work` (spaCy POS, EWT text) | `make spacy-pos-prepare`, then use `data/spacy-pos-work`; the final model it trains is byte-identical |
| `data/spacy-ner-c4-work` (the final NER data, e8a) | `make ner-prepare`, then use `data/ner-work`; byte-identical |
| `data/spacy-work`, `data/spacy-ner-work`, `data/spacy-ner-all-work`, `data/spacy-ner-all-pos-work`, `data/spacy-ner-wiki-work`, `data/spacy-ner-wiki10-work`, `data/spacy-ner-c4wiki-work` | not kept: intermediate experiments, built by hand as the narratives describe (`conllu-to-vw.py` on `data/spacy/` and the WikiText or C4 samples). `run-e4.sh` is one such build, with the session's original paths |

The parser climb builds its own data from `data/spacy/` (`make dep-prepare`
creates it), so its configs rerun as they are.

The `run-e*.sh` and `final*.sh` scripts are kept as the record of the final
runs; some name folders that are no longer kept.
