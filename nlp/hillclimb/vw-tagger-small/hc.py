#!/usr/bin/env python
"""Train one tagger config, score it on dev, and append a row to results.tsv.

Usage: hc.py NAME FEATS PASSES [VW ARGS...]

FEATS is a featurize.py spec (e.g. "w" or "w,l,s4,p3,h,f"). The base search
flags from run.sh are always included; VW ARGS are added on top. Test is not
scored unless HC_TEST=1 (final runs only): then the model is kept, test is
scored too, and the row goes to test-results.tsv instead of results.tsv.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
NLP = HERE.parents[1]
# Run folders (data and models per config); not committed.
SCRATCH = Path(os.environ.get("HC_SCRATCH", REPO / "data/hillclimb/hc"))
# Data folder (relative to the repo) with train/dev .conllu files from
# conllu-to-vw.py; HC_DATA=data/spacy-work scores agreement with spaCy instead.
DATA = REPO / os.environ.get("HC_DATA", "data/work")
VW = ["uv", "run", "--project", str(NLP), "python", "-m", "vowpalwabbit"]
BASE = ["--search_task", "sequence", "--search", "50", "--search_alpha", "1e-6",
        "--search_max_bias_ngram_length", "2", "--search_max_quad_ngram_length", "1",
        "--holdout_off"]

sys.path.insert(0, str(NLP))
from evaluate import read_predictions, read_vw  # noqa: E402


def featurized(split, feats):
    out = SCRATCH / "data" / DATA.name / feats / f"{split}.vw"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{os.getpid()}.tmp")  # parallel runs may build the same file
        subprocess.run([sys.executable, str(HERE / "featurize.py"), feats,
                        str(DATA / f"{split}.conllu"), str(tmp)], check=True)
        tmp.rename(out)
    return out


def score(gold_path, pred_path, train_path):
    gold, preds = read_vw(gold_path), read_predictions(pred_path)
    assert len(gold) == len(preds), (len(gold), len(preds))
    seen = {w for s in read_vw(train_path) for _, w in s}
    tok = ok = oov = oov_ok = sent_ok = 0
    for g, p in zip(gold, preds):
        assert len(g) == len(p)
        hits = [pl == gl for (gl, _), pl in zip(g, p)]
        tok += len(g); ok += sum(hits); sent_ok += all(hits)
        for (gl, w), h in zip(g, hits):
            if w not in seen:
                oov += 1; oov_ok += h
    return 100 * ok / tok, 100 * oov_ok / oov, 100 * sent_ok / len(gold)


def spans(labels, names):
    """Entity spans (type, start, end) from BIO label ids. An I- tag that
    doesn't continue a span of its type starts a new one (conlleval style)."""
    out, cur = [], None
    for i, y in enumerate(labels + [1]):
        tag = names[y]
        if cur and (tag == "O" or tag.startswith("B-") or tag[2:] != cur[0]):
            out.append((cur[0], cur[1], i)); cur = None
        if tag != "O" and cur is None:
            cur = (tag[2:], i)
    return out


def score_ner(gold_path, pred_path):
    """Entity F1 / precision / recall (exact span and type) and token accuracy."""
    names = {int(i): t for t, i in (l.split("\t") for l in
                                     (DATA / "train.tags").read_text().splitlines())}
    gold, preds = read_vw(gold_path), read_predictions(pred_path)
    assert len(gold) == len(preds), (len(gold), len(preds))
    tp = n_gold = n_pred = tok = ok = 0
    for g, p in zip(gold, preds):
        gl = [int(y) for y, _ in g]; pl = [int(y) for y in p]
        gs, ps = set(spans(gl, names)), set(spans(pl, names))
        tp += len(gs & ps); n_gold += len(gs); n_pred += len(ps)
        tok += len(gl); ok += sum(a == b for a, b in zip(gl, pl))
    prec = 100 * tp / max(n_pred, 1); rec = 100 * tp / max(n_gold, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return f1, prec, rec, 100 * ok / tok


def main():
    name, feats, passes, extra = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    work = SCRATCH / "runs" / name
    work.mkdir(parents=True, exist_ok=True)
    train, dev = featurized("train", feats), featurized("dev", feats)
    model = work / "model.vw"

    # Base flags give way to the same flag in VW ARGS (BASE is flag/value pairs,
    # then the lone --holdout_off, which slicing keeps as-is).
    base = [x for i in range(0, len(BASE), 2) if BASE[i] not in extra
            for x in BASE[i:i + 2]]
    t0 = time.time()
    subprocess.run(VW + ["-k", "-c", "--cache_file", str(work / "cache"), "-d", str(train),
                         "--passes", passes, *base, *extra, "-f", str(model), "--quiet"],
                   check=True)
    train_s = time.time() - t0
    (work / "cache").unlink(missing_ok=True)

    t0 = time.time()
    subprocess.run(VW + ["-t", "-i", str(model), "-d", str(dev), "-p", str(work / "dev.pred"),
                         "--quiet"], check=True)
    pred_s = time.time() - t0

    ner = any(l.startswith("B-") for l in (DATA / "train.tags").read_text().splitlines())
    if ner:
        metrics = score_ner(DATA / "dev.conllu", work / "dev.pred")
    else:
        metrics = score(DATA / "dev.conllu", work / "dev.pred", DATA / "train.conllu")
    size_mb = model.stat().st_size / 1e6
    final = os.environ.get("HC_TEST") == "1"
    test_cols = []
    if final:
        test = featurized("test", feats)
        subprocess.run(VW + ["-t", "-i", str(model), "-d", str(test), "-p",
                             str(work / "test.pred"), "--quiet"], check=True)
        t = (score_ner(DATA / "test.conllu", work / "test.pred") if ner else
             score(DATA / "test.conllu", work / "test.pred", DATA / "train.conllu"))
        test_cols = [*(f"{x:.2f}" for x in t), DATA.name]
    else:
        model.unlink()  # models are cheap to rebuild; keep disk use low

    row = [name, feats, passes, " ".join(extra), *(f"{x:.2f}" for x in metrics),
           f"{train_s:.0f}", f"{pred_s:.1f}", f"{size_mb:.2f}", time.strftime("%H:%M"),
           *test_cols]
    results = HERE / (("test-" if final else "") + ("results-ner.tsv" if ner else "results.tsv"))
    m = ["f1", "prec", "rec", "tok_acc"] if ner else ["acc", "unseen", "sent"]
    if not results.exists():
        results.write_text("\t".join(["name", "feats", "passes", "vw_args", *(f"dev_{x}" for x in m),
                                      "train_s", "predict_s", "model_mb", "finished"]
                                     + ([*(f"test_{x}" for x in m), "data"] if final else []))
                           + "\n")
    with open(results, "a") as f:
        f.write("\t".join(row) + "\n")
    print("\t".join(row))


if __name__ == "__main__":
    main()
