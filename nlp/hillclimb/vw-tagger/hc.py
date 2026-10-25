#!/usr/bin/env python
"""Train one tagger config, score it on dev, and append a row to results.tsv.

Usage: hc.py NAME FEATS PASSES [VW ARGS...]

FEATS is a featurize.py spec (e.g. "w" or "w,l,s4,p3,h,f"). The base search
flags from run.sh are always included; VW ARGS are added on top. Test is never
scored here.
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
VW = ["uv", "run", "--project", str(NLP), "python", "-m", "vowpalwabbit"]
BASE = ["--search_task", "sequence", "--search", "50", "--search_alpha", "1e-6",
        "--search_max_bias_ngram_length", "2", "--search_max_quad_ngram_length", "1",
        "--holdout_off"]

sys.path.insert(0, str(NLP))
from evaluate import read_predictions, read_vw  # noqa: E402


def featurized(split, feats):
    out = SCRATCH / "data" / feats / f"{split}.vw"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{os.getpid()}.tmp")  # parallel runs may build the same file
        subprocess.run([sys.executable, str(HERE / "featurize.py"), feats,
                        str(REPO / "data/work" / f"{split}.conllu"), str(tmp)], check=True)
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

    acc, unseen, sent = score(REPO / "data/work/dev.conllu", work / "dev.pred",
                              REPO / "data/work/train.conllu")
    size_mb = model.stat().st_size / 1e6
    model.unlink()  # models are cheap to rebuild; keep disk use low

    row = [name, feats, passes, " ".join(extra), f"{acc:.2f}", f"{unseen:.2f}", f"{sent:.2f}",
           f"{train_s:.0f}", f"{pred_s:.1f}", f"{size_mb:.0f}", time.strftime("%H:%M")]
    results = HERE / "results.tsv"
    if not results.exists():
        results.write_text("name\tfeats\tpasses\tvw_args\tdev_acc\tdev_unseen\tdev_sent\t"
                           "train_s\tpredict_s\tmodel_mb\tfinished\n")
    with open(results, "a") as f:
        f.write("\t".join(row) + "\n")
    print("\t".join(row))


if __name__ == "__main__":
    main()
