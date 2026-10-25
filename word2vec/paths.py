"""Where everything lives. Data goes in the repo's shared data/ (not committed).

    DATA     ../data                   shared with the rest of the repo
    WORK     DATA/word2vec-work        10M-token corpus, its 20k vocabulary, vectors
    BIG      DATA/word2vec-big         100M-token corpus (memory/scale runs)
    BENCH    DATA/word-benchmarks      SimLex-999, WordSim-353, Google analogies
    RESULTS  word2vec/results          results.tsv and memory.tsv (committed)

Each can be overridden with an environment variable of the same name
(W2V_DATA, W2V_WORK, ...).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.environ.get("W2V_DATA", os.path.join(HERE, "..", "data")))
WORK = os.environ.get("W2V_WORK", os.path.join(DATA, "word2vec-work"))
BIG = os.environ.get("W2V_BIG", os.path.join(DATA, "word2vec-big"))
BENCH = os.environ.get("W2V_BENCH", os.path.join(DATA, "word-benchmarks"))
RESULTS = os.environ.get("W2V_RESULTS", os.path.join(HERE, "results"))
