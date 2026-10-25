#!/usr/bin/env python
"""Train one VW dep_parser config on spaCy's parses, score agreement on dev.

Usage: hc.py NAME FEATS PASSES [VW ARGS...]

FEATS is a dep-vw.py feature spec (e.g. "w,p"). Data: spaCy-labeled CoNLL-U
in data/spacy/ (HC_TRAIN, default train.conllu = all UD text). Dev tags are
spaCy's own XPOS unless HC_POS=vw (then a VW tagger's predictions, from
HC_POS_PRED/HC_POS_TAGS). Scores UAS/LAS (all tokens, and without punct) and
appends a row to results.tsv; HC_TEST=1 also scores test and writes to
test-results.tsv (final models only).
"""
import os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
NLP = HERE.parents[1]
# Run folders (data and models per config); not committed.
SCRATCH = Path(os.environ.get("HC_SCRATCH", REPO / "data/hillclimb/dep"))
SPACY = REPO / "data/spacy"
TRAIN = os.environ.get("HC_TRAIN", "train.conllu")
RELS = HERE / "rels.tsv"
VW = ["uv", "run", "--project", str(NLP), "python", "-m", "vowpalwabbit"]


def featurized(split, feats, pos=None):
    src = SPACY / (TRAIN if split == "train" else f"{split}.conllu")
    tag = f"{src.stem}-{feats}" + (f"-pos{pos}" if pos else "")
    out = SCRATCH / "data" / f"{tag}.vw"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        extra = []
        if pos:
            extra = ["--pos-pred", os.environ[f"HC_POS_PRED_{split.upper()}"],
                     "--pos-tags", os.environ["HC_POS_TAGS"]]
        subprocess.run(["python3", str(HERE / "dep-vw.py"), str(src), str(out) + ".part",
                        "--rels", str(RELS), "--feats", feats, *extra], check=True)
        os.rename(str(out) + ".part", out)
    return out


def read_gold(path):
    sents, cur = [], []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            if cur: sents.append(cur); cur = []
            continue
        label, feats = line.split(" ", 1)
        h, r = label.split(",")
        tag = feats.split("|p ")[1].split()[0] if "|p " in feats else None
        cur.append((int(h), int(r), tag))
    if cur: sents.append(cur)
    return sents


def read_pred(path):
    sents, cur = [], []
    for line in open(path):
        if not line.strip():
            if cur: sents.append(cur); cur = []
            continue
        h, r = line.split(":")
        cur.append((int(h), int(r)))
    if cur: sents.append(cur)
    return sents


def score(gold_path, pred_path, gold_conllu):
    """UAS/LAS over all tokens and without punctuation (spaCy's own XPOS of
    the reference parse decides what's punctuation)."""
    punct = {"punct"}
    rels = {int(v): k for k, v in (l.split("\t") for l in RELS.read_text().splitlines())}
    gold, pred = read_gold(gold_path), read_pred(pred_path)
    assert len(gold) == len(pred), (len(gold), len(pred))
    n = u = l = n_np = u_np = l_np = 0
    for g, p in zip(gold, pred):
        assert len(g) == len(p)
        for (gh, gr, _), (ph, pr) in zip(g, p):
            ok_u = gh == ph; ok_l = ok_u and gr == pr
            n += 1; u += ok_u; l += ok_l
            if rels[gr] not in punct:
                n_np += 1; u_np += ok_u; l_np += ok_l
    return 100 * u / n, 100 * l / n, 100 * u_np / n_np, 100 * l_np / n_np


def main():
    name, feats, passes, extra = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    work = SCRATCH / "runs" / name
    work.mkdir(parents=True, exist_ok=True)
    pos = os.environ.get("HC_POS")
    train, dev = featurized("train", feats), featurized("dev", feats, pos)
    L = len(RELS.read_text().splitlines())
    root = next(int(v) for k, v in (l.split("\t") for l in RELS.read_text().splitlines()) if k == "root")
    base = ["--search_task", "dep_parser", "--search", str(L), "--num_label", str(L),
            "--root_label", str(root), "--holdout_off"]
    model = work / "model.vw"
    t0 = time.time()
    subprocess.run(VW + ["-k", "-c", "--cache_file", str(work / "cache"), "-d", str(train),
                         "--passes", passes, *base, *extra, "-f", str(model), "--quiet"], check=True)
    train_s = time.time() - t0
    (work / "cache").unlink(missing_ok=True)
    t0 = time.time()
    subprocess.run(VW + ["-t", "-i", str(model), "-d", str(dev), "-p", str(work / "dev.pred"), "--quiet"], check=True)
    pred_s = time.time() - t0
    m = score(dev, work / "dev.pred", None)
    size_mb = model.stat().st_size / 1e6
    final = os.environ.get("HC_TEST") == "1"
    test_cols = []
    if final:
        test = featurized("test", feats, pos)
        subprocess.run(VW + ["-t", "-i", str(model), "-d", str(test), "-p", str(work / "test.pred"), "--quiet"], check=True)
        test_cols = [f"{x:.2f}" for x in score(test, work / "test.pred", None)]
    else:
        model.unlink()
    row = [name, feats, passes, " ".join(extra), *(f"{x:.2f}" for x in m), f"{train_s:.0f}",
           f"{pred_s:.1f}", f"{size_mb:.2f}", TRAIN, pos or "spacy", time.strftime("%H:%M"), *test_cols]
    results = HERE / (("test-" if final else "") + "results.tsv")
    if not results.exists():
        results.write_text("\t".join(["name", "feats", "passes", "vw_args", "dev_uas", "dev_las",
                                      "dev_uas_np", "dev_las_np", "train_s", "predict_s", "model_mb",
                                      "train", "pos", "finished"]
                                     + (["test_uas", "test_las", "test_uas_np", "test_las_np"] if final else []))
                           + "\n")
    with open(results, "a") as f:
        f.write("\t".join(row) + "\n")
    print("\t".join(row), flush=True)


if __name__ == "__main__":
    main()
