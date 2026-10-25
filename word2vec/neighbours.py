#!/usr/bin/env python
"""Print cosine nearest neighbours for a few query words.

    neighbours.py VECS.npz [word ...]
"""
import sys

import numpy as np

QUERIES = ["london", "monday", "dog", "war", "music", "good", "france", "seven", "river", "said"]


def main():
    z = np.load(sys.argv[1], allow_pickle=True)
    words = [str(w) for w in z["words"]]
    M = z["vectors"].astype(np.float32)
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    idx = {w: i for i, w in enumerate(words)}
    for q in sys.argv[2:] or QUERIES:
        if q in idx:
            s = M @ M[idx[q]]
            nb = [words[j] for j in np.argsort(-s)[1:9]]
            print(f"{q:>8}: {', '.join(nb)}")


if __name__ == "__main__":
    main()
