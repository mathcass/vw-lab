#!/usr/bin/env python
"""Non-VW reference embeddings on the exact same tokens (WORK/corpus.ids, see paths.py).

    baselines.py ppmi  OUT.npz [--dim 100 --window 5 --cds 0.75 --power 0.5]
    baselines.py w2v   OUT.npz [--dim 100 --window 5 --epochs 5 --sg 1 --threads 2]
    baselines.py w2v_stream OUT.npz [same options; streams WORK/corpus.vocab.txt]
    (all modes take --work DIR, default WORK; BIG is the 100M-token corpus)

ppmi: symmetric-window co-occurrence counts -> PPMI (context-distribution
      smoothing `cds`) -> truncated SVD; vectors are U * S^power.
w2v : gensim word2vec (SGNS by default). This is the reference for "what a real
      skip-gram trainer gets on this data", which the earlier attempt lacked.

Needs scipy + gensim (in this folder's uv project, Python 3.12-3.13: gensim has no
Python 3.14 wheel, which is why word2vec/ has its own project).
Prints `train_s=<seconds>`.
"""
import argparse
import time

import numpy as np

from paths import WORK


def load(work, max_paras=0):
    vocab = [l.split("\t")[1] for l in open(f"{work}/vocab.tsv")]
    paras = [np.array(l.split(), dtype=np.int32) for l in open(f"{work}/corpus.ids")]
    paras = [p for p in paras if len(p) > 1]
    return vocab, paras[:max_paras] if max_paras else paras


def cooc(paras, V, window):
    import scipy.sparse as sp
    ids = np.concatenate(paras)
    pid = np.concatenate([np.full(len(p), i, dtype=np.int32) for i, p in enumerate(paras)])
    C = sp.csr_matrix((V, V), dtype=np.float64)
    for d in range(1, window + 1):
        ok = pid[:-d] == pid[d:]
        a, b = ids[:-d][ok], ids[d:][ok]
        m = sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(V, V)).tocsr()
        C = C + m + m.T
    return C


def ppmi_svd(args, vocab, paras):
    import scipy.sparse as sp
    from scipy.sparse.linalg import svds
    V = len(vocab)
    C = cooc(paras, V, args.window).tocoo()
    # PMI = log P(w,c) / (P(w) P_cds(c)); the corpus total cancels out.
    row = np.asarray(C.sum(1)).ravel()
    col = np.asarray(C.sum(0)).ravel() ** args.cds
    pmi = np.log(C.data * col.sum() / (row[C.row] * col[C.col]))
    keep = pmi > 0
    P = sp.csr_matrix((pmi[keep], (C.row[keep], C.col[keep])), shape=(V, V))
    U, S, _ = svds(P, k=args.dim)
    order = np.argsort(-S)
    return U[:, order] * (S[order] ** args.power), f"nnz {P.nnz}"


def w2v(args, vocab, paras):
    from gensim.models import Word2Vec
    sents = [[vocab[i] for i in p] for p in paras]
    m = Word2Vec(sents, vector_size=args.dim, window=args.window, negative=5, sg=args.sg,
                 sample=args.sample, min_count=1, workers=args.threads, epochs=args.epochs, seed=args.seed)
    return np.stack([m.wv[w] if w in m.wv else np.zeros(args.dim) for w in vocab]), ""


def w2v_stream(args, vocab):
    """gensim streaming from disk (LineSentence), the usual way to run it at scale.
    Reads WORK/corpus.vocab.txt (in-vocab tokens only), so the vocabulary is
    identical to the one VW sees."""
    from gensim.models import Word2Vec
    from gensim.models.word2vec import LineSentence
    m = Word2Vec(LineSentence(f"{args.work}/corpus.vocab.txt"), vector_size=args.dim, window=args.window,
                 negative=5, sg=args.sg, sample=args.sample, min_count=1, workers=args.threads,
                 epochs=args.epochs, seed=args.seed)
    return np.stack([m.wv[w] if w in m.wv else np.zeros(args.dim) for w in vocab]), f"vocab {len(m.wv)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["ppmi", "w2v", "w2v_stream"])
    ap.add_argument("out")
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--cds", type=float, default=0.75)
    ap.add_argument("--power", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--sg", type=int, default=1)
    ap.add_argument("--sample", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-paras", type=int, default=0, help="same meaning as gen.py --max-paras")
    ap.add_argument("--work", default=WORK, help="directory with vocab.tsv and corpus.ids")
    args = ap.parse_args()
    if args.method == "w2v_stream":
        vocab = [l.split("\t")[1] for l in open(f"{args.work}/vocab.tsv")]
        t = time.time()
        E, info = w2v_stream(args, vocab)
    else:
        vocab, paras = load(args.work, args.max_paras)
        t = time.time()
        E, info = (ppmi_svd if args.method == "ppmi" else w2v)(args, vocab, paras)
    dt = time.time() - t
    np.savez(args.out, words=np.array(vocab), vectors=E.astype(np.float32))
    print(f"{args.method}: {E.shape} {info} train_s={dt:.0f}")


if __name__ == "__main__":
    main()
