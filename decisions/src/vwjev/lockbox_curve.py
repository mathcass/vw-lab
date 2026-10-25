"""How many labeled rows does VW need on a brand-new domain? (lockbox)

For each fast-decisions domain (100 texts each), repeat REPS times:
    split the texts 50/50 into a test half and a learning pool
    start a model (pre-trained fold-0 model, or empty), then feed it the
    pool's rows one at a time with feedback, scoring the test half after
    0, 5, 10, 20, 30 and 50 rows

Feedback modes are the same as feedback.py (correction / bandit). Each
learning step is a single online update: what a product would do.

Scoring pools the test predictions of all repetitions, per domain, and
leaves out multi-label questions so OpenAI (no multi-select) can be scored
on exactly the same questions. OpenAI's zero-shot answers are read from an
openai_decide run, if given.

The lockbox is still never used to choose settings: everything here was
fixed in advance and every adapted model is thrown away.

    uv run python -I -m vwjev.lockbox_curve --model runs/v6_generic_shuf --openai luna_f0
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import time
from pathlib import Path

from vowpalwabbit import Workspace

from vwjev.data import PROCESSED
from vwjev.features import FeatureConfig, shuffle_options, supervised_costs
from vwjev.feedback import lines_with_f, load_pretrained, negative_block
from vwjev.metrics import summarize
from vwjev.run import decide, option_scores

CHECKPOINTS = [0, 5, 10, 20, 30, 50]
REPS = 10
TEST_ROWS = 50


def learn(ws, r, cfg, mode, rng, eps=0.1):
    scores = option_scores(ws, lines_with_f(shuffled := shuffle_options(r, rng), cfg))
    pred = decide(shuffled, scores)
    if mode == "bandit" and r["qtype"] != "multi" and rng.random() < eps:
        pred = [rng.randrange(len(r["options"]))]
    right = set(pred) == set(shuffled["gold"])
    if mode == "correction" or r["qtype"] == "multi" or right:
        ws.learn(lines_with_f(shuffled, cfg, supervised_costs(shuffled)))
    else:
        ws.learn(negative_block(shuffled, cfg, pred[0]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/v6_generic_shuf")
    ap.add_argument("--scratch-vw", default="-q sl -q qm -q qp --ignore t")
    ap.add_argument("--openai", help="openai_decide run name, for the zero-shot reference")
    args = ap.parse_args()
    cfg = FeatureConfig(**json.loads((Path(args.model) / "feature_config.json").read_text()))

    by_dom = collections.defaultdict(lambda: collections.defaultdict(list))
    with (PROCESSED / "fast.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            by_dom[r["domain"]][r["id"].rsplit("/", 1)[0]].append(r)

    # pooled[(start, mode, n)] -> list of (rec, pred) over all reps and domains
    pooled = collections.defaultdict(list)
    test_ids = []  # the test questions in pooled order (same for every key)
    t0 = time.time()
    for dom, rows in sorted(by_dom.items()):
        keys = sorted(rows)
        for rep in range(REPS):
            rng = random.Random(f"{dom}/{rep}")
            order = keys[:]
            rng.shuffle(order)
            test = [r for k in order[:TEST_ROWS] for r in rows[k] if r["qtype"] != "multi"]
            pool = [rows[k] for k in order[TEST_ROWS:]]
            test_ids += [r["id"] for r in test]
            for start in ("pretrained", "scratch"):
                for mode in ("correction", "bandit"):
                    if start == "pretrained":
                        ws = load_pretrained(args.model)
                    else:
                        ws = Workspace(f"--csoaa_ldf mc -b 24 -q fl {args.scratch_vw} --quiet")
                    lrng = random.Random(f"{dom}/{rep}/{start}/{mode}")
                    for n in range(CHECKPOINTS[-1] + 1):
                        if n in CHECKPOINTS:
                            pooled[(start, mode, n)] += [(r, decide(r, option_scores(ws, lines_with_f(r, cfg)))) for r in test]
                        if n < len(pool):
                            for r in pool[n]:  # every question about this text
                                learn(ws, r, cfg, mode, lrng)
                    ws.finish()
        print(f"{dom:20s} done  {time.time() - t0:.0f}s", flush=True)

    out = {"checkpoints_rows": CHECKPOINTS, "reps": REPS, "test_rows": TEST_ROWS, "model": args.model, "curves": {}}
    for (start, mode, n), items in sorted(pooled.items()):
        s = summarize(items)
        out["curves"].setdefault(f"{start}/{mode}", []).append(
            {"n": n, "macro_f1": s["macro_f1"], "acc": s["acc"], "domains": {d: x["macro_f1"] for d, x in s["domains"].items()}})

    if args.openai:
        from vwjev.openai_report import openai_preds, score

        recs = {}
        with (PROCESSED / "fast.jsonl").open() as fh:
            for line in fh:
                r = json.loads(line)
                recs[r["id"]] = r
        oa, *_ = openai_preds(Path("runs/openai") / args.openai / "raw_lockbox.jsonl", recs)
        # score OpenAI on the same pooled list of test questions
        s = score(recs, test_ids, oa)
        out["openai"] = {"macro_f1": s["macro_f1"], "acc": s["acc"], "domains": {d: x["macro_f1"] for d, x in s["domains"].items()}}

    dest = Path(args.model) / "lockbox_curve.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"\n{'curve':24s} " + " ".join(f"n={n:<5d}" for n in CHECKPOINTS) + "  (rows learned; macro-F1)")
    for k, pts in out["curves"].items():
        print(f"{k:24s} " + " ".join(f"{p['macro_f1']:.3f}  " for p in pts))
    if "openai" in out:
        print(f"{'OpenAI zero-shot':24s} {out['openai']['macro_f1']:.3f}")
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
