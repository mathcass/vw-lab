"""Train a csoaa_ldf model on one domain fold and score it.

    uv run python -I -m vwjev.run --name v0 --fold 0 --vw "-q tl -q ql"

Writes runs/<name>/results.json, per-question scores for each eval set,
and the model file.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import time
import zlib
from pathlib import Path

from vowpalwabbit import Workspace, pyvw

from vwjev.data import PROCESSED
from vwjev.features import FeatureConfig, compute_idf, shuffle_options, supervised_costs, to_lines
from vwjev.metrics import summarize

MAX_CHARS = 4000  # tokens beyond max_text_tokens are dropped anyway; keeps memory low


def load_fold(fold: int, limit_train: int | None, seed: int):
    roles = json.loads((PROCESSED / "splits" / f"fold{fold}.json").read_text())
    sets = {"train": [], "dev": [], "seen_test": []}
    with (PROCESSED / "bev.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            role = roles.get(r["id"])
            if role is None:
                continue
            r["text"] = r["text"][:MAX_CHARS]
            sets[role].append(r)
    with (PROCESSED / "fast.jsonl").open() as fh:
        sets["lockbox"] = [json.loads(line) for line in fh]
    # fixed per-question option shuffle for BEV eval sets (removes the
    # answer-position bias); the lockbox keeps its original order
    for name in ("dev", "seen_test"):
        sets[name] = [shuffle_options(r, random.Random(zlib.crc32(r["id"].encode()))) for r in sets[name]]
    rng = random.Random(seed)
    rng.shuffle(sets["train"])
    if limit_train:
        sets["train"] = sets["train"][:limit_train]
    return sets


def option_scores(ws: Workspace, lines: list[str]) -> list[float]:
    """Per-option scores in input order (lower = better).

    pyvw's predict() reads the prediction off the shared line, so read each
    option's partial_prediction instead.
    """
    exs = ws.parse(lines)
    pyvw.pylibvw.vw.predict_multi(ws, exs)
    scores = [e.get_partial_prediction() for e in exs[1:]]
    ws.finish_example(exs)
    return scores


def decide(rec: dict, scores: list[float]) -> list[int]:
    best = min(range(len(scores)), key=scores.__getitem__)
    if rec["qtype"] != "multi":
        return [best]
    # csoaa_ldf mc scores sit near -1 (good) and +1 (bad); 0 is the midpoint
    picked = [i for i, s in enumerate(scores) if s < 0]
    return picked or [best]


def evaluate(ws, recs, cfg, out_path: Path):
    items = []
    with out_path.open("w") as fh:
        for r in recs:
            s = option_scores(ws, to_lines(r, cfg))
            pred = decide(r, s)
            items.append((r, pred))
            fh.write(json.dumps({"id": r["id"], "scores": s, "gold": r["gold"], "qtype": r["qtype"]}) + "\n")
    return summarize(items)


def baselines(recs, seed=0):
    rng = random.Random(seed)
    rand = [(r, [rng.randrange(len(r["options"]))]) for r in recs]
    first = [(r, [0]) for r in recs]
    return {"random": summarize(rand)["macro_f1"], "first_option": summarize(first)["macro_f1"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--vw", default="-q tl -q ql", help="extra VW arguments")
    ap.add_argument("--bits", type=int, default=24)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--max-text-tokens", type=int, default=256)
    ap.add_argument("--no-match", action="store_true")
    ap.add_argument("--no-position", action="store_true")
    ap.add_argument("--no-yesno-crossed", action="store_true")
    ap.add_argument("--no-idf", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = FeatureConfig(
        max_text_tokens=args.max_text_tokens,
        match=not args.no_match,
        position=not args.no_position,
        yesno_crossed=not args.no_yesno_crossed,
    )
    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    sets = load_fold(args.fold, args.limit_train, args.seed)
    print(f"loaded in {time.time() - t0:.0f}s: " + ", ".join(f"{k}={len(v)}" for k, v in sets.items()), flush=True)

    if not args.no_idf:
        seen, texts = set(), []
        for r in sets["train"]:
            if r["group_id"] not in seen:
                seen.add(r["group_id"])
                texts.append(r["text"])
        cfg.idf = compute_idf(texts, args.max_text_tokens)
        print(f"IDF from {len(texts)} unique training texts, {len(cfg.idf)} words", flush=True)
    # everything needed to rebuild features for this model later
    (out / "feature_config.json").write_text(json.dumps(dataclasses.asdict(cfg)))

    vw_args = f"--csoaa_ldf mc -b {args.bits} --quiet {args.vw} -f {out / 'model.vw'}"
    ws = Workspace(vw_args)
    rng = random.Random(args.seed)
    t0 = time.time()
    for p in range(args.passes):
        if p:
            rng.shuffle(sets["train"])
        for i, r in enumerate(sets["train"]):
            r = shuffle_options(r, rng)
            ws.learn(to_lines(r, cfg, supervised_costs(r)))
            if i and i % 50000 == 0:
                print(f"  pass {p + 1}: {i} questions, {i / (time.time() - t0):.0f}/s", flush=True)
    train_s = time.time() - t0

    results = {"args": vars(args), "vw_args": vw_args, "train_seconds": round(train_s)}
    for name in ("dev", "seen_test", "lockbox"):
        t0 = time.time()
        s = evaluate(ws, sets[name], cfg, out / f"preds_{name}.jsonl")
        s["baselines"] = baselines(sets[name])
        s["eval_ms_per_question"] = round(1000 * (time.time() - t0) / max(1, len(sets[name])), 3)
        results[name] = s
    ws.finish()
    results["model_mb"] = round((out / "model.vw").stat().st_size / 1e6, 1)
    (out / "results.json").write_text(json.dumps(results, indent=1))

    print(f"\n{args.name}  fold={args.fold}  train {train_s:.0f}s  model {results['model_mb']} MB  [{vw_args}]")
    print(f"{'set':10s} {'macroF1':>8s} {'acc':>6s} {'random':>7s} {'first':>6s}   by type (macro-F1)")
    for name in ("dev", "seen_test", "lockbox"):
        s = results[name]
        bt = "  ".join(f"{t}={v['macro_f1']:.3f}" for t, v in sorted(s["by_qtype"].items()))
        print(
            f"{name:10s} {s['macro_f1']:8.3f} {s['acc']:6.3f} {s['baselines']['random']:7.3f} "
            f"{s['baselines']['first_option']:6.3f}   {bt}"
        )


if __name__ == "__main__":
    main()
