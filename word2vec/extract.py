#!/usr/bin/env python
"""Read `--lrq` word factors straight out of a saved VW model (`-f MODEL`).

    extract.py MODEL K OUT_PREFIX [--ns wc]

Works because gen.py uses integer feature ids, which VW does not hash: the
linear weight of id i in namespace ns sits at
    ((hash_space(ns) + i) * STEP) mod num_weights,     STEP = 2^ceil(log2(K+1))
and its K lrq factors are the next K slots. The formula is checked against
VW's own name lookup, and the script refuses to run if any two words' weight
blocks collide (raise -b if so). This replaces --invert_hash /
--dump_json_weights, which are O(examples), single-pass only, and keep just one
name per hash slot (silently dropping colliding words).

Writes OUT_PREFIX.w.npz, .c.npz and .wc.npz (w + c, the word2vec trick).
"""
import argparse

import numpy as np
from vowpalwabbit import Workspace

from paths import WORK


def probe(model, namespaces):
    """Index of ids 0-2 per namespace, from VW's own --invert_hash on one example.
    (Workspace.get_weight_from_name is no use here: it ignores lrq's slot step.)"""
    import re
    import subprocess
    import sys
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        with open(f"{d}/p.vw", "w") as f:
            f.write("1 " + " ".join(f"|{ns} 0 1 2" for ns in namespaces) + "\n")
        subprocess.run([sys.executable, "-m", "vowpalwabbit", "-i", model, "-t", "-d", f"{d}/p.vw",
                        "--invert_hash", f"{d}/p.ih", "--quiet"], check=True)
        found = {}
        for line in open(f"{d}/p.ih"):
            m = re.match(r"^(\w+)\^(\d+):(\d+):", line)
            if m:
                found[(m.group(1), int(m.group(2)))] = int(m.group(3))
        return found


def extract_gdmf(vw, vocab, K, args):
    """gd_mf keeps a word's factors inside its own weight's stride block:
    get_weight(i, 0) is the linear weight, offsets 1..K the left-namespace
    factors and K+1..2K the right-namespace factors (checked against --audit)."""
    N, V = vw.num_weights(), len(vocab)
    out, slots = {}, []
    for side, ns in enumerate(args.ns):
        h = vw.hash_space(ns)
        if args.hashed:  # --words runs: VW hashed the word strings
            base = [vw.hash_feature(w, h) % N for w in vocab]
        else:
            base = [(h + i) % N for i in range(V)]
        slots.append(base)
        off = 1 + side * K
        out[ns] = np.array([[vw.get_weight(b, off + k) for k in range(K)] for b in base],
                           dtype=np.float32)
        np.savez(f"{args.prefix}.{ns}.npz", words=np.array(vocab), vectors=out[ns])
    # Only collisions inside one namespace share factors (each side has its own
    # offsets in the slot); a w word and a c word in one slot share just the
    # linear weight.
    shared = [V - len(set(b)) for b in slots]
    cross = len(set(slots[0]) & set(slots[1]))
    if any(shared) and not args.hashed:
        raise SystemExit(f"gd_mf factor slots collide within a namespace {shared}; raise -b")
    np.savez(f"{args.prefix}.wc.npz", words=np.array(vocab), vectors=out[args.ns[0]] + out[args.ns[1]])
    print(f"extracted gd_mf {V} x {K} per namespace; words sharing a slot with another word "
          f"of the same namespace {shared}; w/c slots that share a linear weight {cross}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("k", type=int)
    ap.add_argument("prefix")
    ap.add_argument("--ns", default="wc", help='namespaces: "wc", or comma-separated names like "w,cb"')
    ap.add_argument("--vocab", default=f"{WORK}/vocab.tsv")
    ap.add_argument("--hashed", action="store_true", help="gd_mf model trained on words (gen.py --words)")
    ap.add_argument("--gdmf", action="store_true",
                    help="model is `-q wc --rank K` (gd_mf), not --lrq")
    args = ap.parse_args()
    args.ns = args.ns.split(",") if "," in args.ns else list(args.ns)
    vocab = [l.split("\t")[1] for l in open(args.vocab)]
    V, K = len(vocab), args.k
    vw = Workspace(f"-i {args.model} --quiet -t")
    N = vw.num_weights()
    if args.gdmf:
        return extract_gdmf(vw, vocab, K, args)
    step = 1 << int(np.ceil(np.log2(K + 1)))
    named = probe(args.model, args.ns)
    out = {}
    blocks = []
    for ns in args.ns:
        h = vw.hash_space(ns)
        base = [((h + i) * step) % N for i in range(V)]
        for i in (0, 1, 2):  # formula check against VW's own named dump
            if (ns, i) in named:
                assert named[(ns, i)] == base[i], (ns, i, named[(ns, i)], base[i])
        assert named, "probe found no named weights"
        blocks += base
        M = np.array([[vw.get_weight(b + 1 + k) for k in range(K)] for b in base],
                     dtype=np.float32)
        out[ns] = M
        np.savez(f"{args.prefix}.{ns}.npz", words=np.array(vocab), vectors=M)
    b = np.sort(np.array(blocks))
    gaps = np.diff(np.append(b, b[0] + N))
    if gaps.min() < step:
        raise SystemExit(f"weight blocks collide (min gap {gaps.min()} < {step}); raise -b")
    if len(args.ns) == 2:
        np.savez(f"{args.prefix}.wc.npz", words=np.array(vocab), vectors=out[args.ns[0]] + out[args.ns[1]])
    zero = {ns: int((np.abs(M).sum(1) == 0).sum()) for ns, M in out.items()}
    print(f"extracted {V} x {K} per namespace, all-zero rows {zero}")


if __name__ == "__main__":
    main()
