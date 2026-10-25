"""Simulate a product that starts cold and learns from feedback.

For one domain held out of a pre-trained model's fold:
    stream  that domain's BEV train rows (never seen in pre-training)
    eval    that domain's dev rows

Each streamed question is answered by the current model; the "user" then
gives feedback and the model updates online. Feedback modes:

    correction  the user supplies the right answer every time
    bandit      the user only says whether the chosen answer was right;
                the model explores with epsilon-greedy

A wrong bandit pick only tells us that one option is bad, so that update
regresses just the chosen option toward "bad" (a 0/1-cost block with a
null option). A right pick on a single-answer question implies every other
option is wrong, so it becomes a full update.

The text is added in a fresh namespace `f` with a new `-q fl` interaction,
so in-domain vocabulary can be learned without disturbing pre-trained weights
(see load_pretrained: the saved interactions must be restated).

    uv run python -I -m vwjev.feedback --model runs/v3_generic_rel --domain "Spam detection"
"""

from __future__ import annotations

import argparse
import json
import random
import time
import zlib
from pathlib import Path

from vowpalwabbit import Workspace

from vwjev.data import PROCESSED
from vwjev.features import FeatureConfig, shuffle_options, supervised_costs, to_lines, tokens
from vwjev.metrics import summarize
from vwjev.run import MAX_CHARS, decide, option_scores

CHECKPOINTS = [0, 25, 50, 100, 200, 400, 800, 1600, 3200]


def load_domain(fold: int, domain: str, stream_cap: int, seed: int):
    roles = json.loads((PROCESSED / "splits" / f"fold{fold}.json").read_text())
    stream, ev = [], []
    with (PROCESSED / "bev.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            if r["domain"] != domain:
                continue
            role = roles.get(r["id"])
            if role in ("train", "seen_test"):
                raise SystemExit(f"{domain!r} is not held out in fold {fold}")
            r["text"] = r["text"][:MAX_CHARS]
            # same per-question option shuffle as run.py
            r = shuffle_options(r, random.Random(zlib.crc32(r["id"].encode())))
            if role == "dev":
                ev.append(r)
            elif r["orig_split"] == "train":
                stream.append(r)
    random.Random(seed).shuffle(stream)
    return stream[:stream_cap], ev


def lines_with_f(rec: dict, cfg: FeatureConfig, costs=None) -> list[str]:
    lines = to_lines(rec, cfg, costs)
    f_toks = tokens(rec["text"])[: cfg.max_text_tokens]
    lines[0] += f" |f {' '.join(f_toks)}"
    return lines


def load_pretrained(model_dir: str | Path, extra: str = "-q fl") -> Workspace:
    """Reload a saved model and add interactions.

    VW drops a model's saved -q interactions when any -q is passed on the
    command line at load time (the command line wins), so restate the saved
    ones alongside the new ones.
    """
    model_dir = Path(model_dir)
    saved = json.loads((model_dir / "results.json").read_text())["vw_args"].split()
    inter = " ".join(f"{a} {b}" for a, b in zip(saved, saved[1:]) if a in ("-q", "--interactions", "--cubic"))
    return Workspace(f"-i {model_dir / 'model.vw'} {inter} {extra} --quiet")


def negative_block(rec: dict, cfg: FeatureConfig, a: int) -> list[str]:
    """Only option `a` is known to be wrong: regress it toward bad."""
    full = lines_with_f(rec, cfg)
    opt = full[1 + a].split(" ", 1)[1]  # drop the dummy "i:0" label
    return [full[0], f"1:1 {opt}", "2:0 |l __null__"]


def eval_set(ws, recs, cfg):
    items = [(r, decide(r, option_scores(ws, lines_with_f(r, cfg)))) for r in recs]
    s = summarize(items)
    return {"macro_f1": round(s["macro_f1"], 4), "acc": round(s["acc"], 4)}


def run_one(start: str, mode: str, args, cfg, stream, ev, rng):
    if start == "pretrained":
        ws = load_pretrained(args.model)
    else:
        ws = Workspace(f"--csoaa_ldf mc -b 24 -q fl {args.scratch_vw} --quiet")
    curve, online_correct = [], 0
    for n, r in enumerate(stream + [None]):
        if n in CHECKPOINTS or r is None:
            point = {"n": n, **eval_set(ws, ev, cfg), "online_acc": round(online_correct / max(1, n), 4)}
            curve.append(point)
            print(f"    {start:10s} {mode:10s} n={n:5d} macroF1={point['macro_f1']:.3f} acc={point['acc']:.3f} online={point['online_acc']:.3f}", flush=True)
        if r is None or n >= CHECKPOINTS[-1]:
            break
        scores = option_scores(ws, lines_with_f(r, cfg))
        pred = decide(r, scores)
        if mode == "bandit" and r["qtype"] != "multi" and rng.random() < args.epsilon:
            pred = [rng.randrange(len(r["options"]))]
        right = set(pred) == set(r["gold"])
        online_correct += right
        if mode == "correction" or r["qtype"] == "multi":
            ws.learn(lines_with_f(r, cfg, supervised_costs(r)))
        elif right:
            ws.learn(lines_with_f(r, cfg, supervised_costs(r)))
        else:
            ws.learn(negative_block(r, cfg, pred[0]))
    ws.finish()
    return curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="run directory of a pre-trained model")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--stream-cap", type=int, default=CHECKPOINTS[-1])
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--scratch-vw", default="-q sl -q qm -q qp --ignore t", help="interactions for the from-scratch model")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = FeatureConfig(**json.loads((Path(args.model) / "feature_config.json").read_text()))
    stream, ev = load_domain(args.fold, args.domain, args.stream_cap, args.seed)
    print(f"{args.domain}: stream={len(stream)} eval={len(ev)}", flush=True)

    results = {"domain": args.domain, "model": args.model, "epsilon": args.epsilon, "curves": {}}
    t0 = time.time()
    for start in ("pretrained", "scratch"):
        for mode in ("correction", "bandit"):
            results["curves"][f"{start}/{mode}"] = run_one(start, mode, args, cfg, stream, ev, random.Random(args.seed))
    out = Path(args.model) / "feedback"
    out.mkdir(exist_ok=True)
    slug = "".join(c if c.isalnum() else "_" for c in args.domain).lower()
    (out / f"{slug}.json").write_text(json.dumps(results, indent=1))
    print(f"done in {time.time() - t0:.0f}s -> {out / (slug + '.json')}")


if __name__ == "__main__":
    main()
