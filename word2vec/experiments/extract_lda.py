#!/usr/bin/env python
"""Word vectors from a VW --lda model.

    extract_lda.py MODEL K OUT_PREFIX

VW stores the topic-word parameters lambda for integer feature i at weight
index i: `get_weight(i, k)` is lambda[k, i] (readable_model adds the 0.1 prior).
Writes three variants:
    .lam    raw lambda row
    .ptw    p(topic | word): lambda[:, w] / topic size, renormalized per word
    .hel    sqrt(p(topic | word)), whose cosine is a Hellinger-style similarity
"""
import os
import sys

import numpy as np
from vowpalwabbit import Workspace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import WORK  # noqa: E402


def main():
    model, K, prefix = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    vocab = [l.split("\t")[1] for l in open(f"{WORK}/vocab.tsv")]
    vw = Workspace(f"-i {model} --quiet -t")
    lam = np.array([[vw.get_weight(i, k) for k in range(K)] for i in range(len(vocab))]) + 0.1
    ptw = lam / lam.sum(0, keepdims=True)          # p(word | topic)
    ptw = ptw / ptw.sum(1, keepdims=True)          # -> p(topic | word), uniform topic prior
    for tag, M in (("lam", lam), ("ptw", ptw), ("hel", np.sqrt(ptw))):
        np.savez(f"{prefix}.{tag}.npz", words=np.array(vocab), vectors=M.astype(np.float32))
    print(f"lda: {len(vocab)} x {K}")


if __name__ == "__main__":
    main()
