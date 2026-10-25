"""Score an OpenAI Decisions run against VW on exactly the same questions.

Reads runs/openai/<name>/raw_<set>.jsonl (from openai_decide) and VW's
per-question scores from runs/<vw run>/preds_<set>.jsonl. Only questions
OpenAI answered are scored, for every model, so the comparison is like for
like. Multi-label questions are excluded (the API has no multi-select).

OpenAI prediction per type:
- predicate: "yes" if probability >= 0.5
- choice:    the returned choice
- score:     the most probable level (the API's `score` is a weighted mean)
Refusals are counted and scored as wrong (prediction = an invalid index
mapped to a label no gold uses).

    uv run python -I -m vwjev.openai_report --name luna_f0
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from statistics import mean

from vwjev.cost_estimate import PRICE_PER_M
from vwjev.metrics import summarize
from vwjev.openai_decide import OUT, load_set

VW_RUNS = {"VW full": "v5_full_shuf", "VW generic": "v6_generic_shuf"}
REFUSED = "__refused__"


def openai_preds(raw: Path, recs: dict[str, dict]):
    preds, conf, stats = {}, {}, collections.Counter()
    tokens, est = 0, 0
    with raw.open() as fh:
        for line in fh:
            d = json.loads(line)
            if d["status"] != 200:
                stats[f"http_{d['status']}"] += 1
                continue
            stats["requests"] += 1
            tokens += d["response"].get("usage", {}).get("input_tokens", 0)
            est += d["est_tokens"]
            answers = {a["name"]: a for a in d["response"].get("answers", [])}
            for i, qid in enumerate(d["ids"]):
                a, r = answers.get(f"q{i}"), recs[qid]
                if a is None or a["type"] == "refusal":
                    stats["refusal" if a else "missing"] += 1
                    preds[qid] = None
                    continue
                if a["type"] == "predicate":
                    # options are shuffled, so "yes" is not always first
                    p = a["probability"]
                    keys = [o["key"] for o in r["options"]]
                    preds[qid], conf[qid] = [keys.index("yes" if p >= 0.5 else "no")], max(p, 1 - p)
                elif a["type"] == "choice":
                    keys = [o["key"] for o in r["options"]]
                    preds[qid] = [keys.index(a["choice"])] if a["choice"] in keys else None
                    conf[qid] = a.get("confidence")
                    if preds[qid] is None:
                        stats["unknown_choice"] += 1
                else:
                    best = max(a["probabilities"], key=lambda x: x["probability"])
                    preds[qid], conf[qid] = [int(best["value"])], a.get("confidence")
    return preds, conf, stats, tokens, est


def vw_preds(path: Path) -> dict[str, list[int]]:
    out = {}
    with path.open() as fh:
        for line in fh:
            d = json.loads(line)
            out[d["id"]] = [min(range(len(d["scores"])), key=d["scores"].__getitem__)]
    return out


def score(recs, ids, preds):
    items = []
    for q in ids:
        r = recs[q]
        p = preds.get(q)
        if p is None:  # refusal / unparseable: count as a wrong answer
            r = dict(r, options=r["options"] + [{"key": REFUSED, "desc": ""}])
            p = [len(r["options"]) - 1]
        items.append((r, p))
    return summarize(items)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    run = OUT / args.name
    results = {}
    total_tok = total_est = 0
    for set_name in ("lockbox", "dev", "seen_test"):
        raw = run / f"raw_{set_name}.jsonl"
        if not raw.exists():
            continue
        recs = {r["id"]: r for r in load_set(set_name, args.fold)}
        oa, conf, stats, tok, est = openai_preds(raw, recs)
        total_tok += tok
        total_est += est
        ids = sorted(oa)
        rows = {"OpenAI gpt-6-luna": score(recs, ids, oa)}
        for label, vw_run in VW_RUNS.items():
            rows[label] = score(recs, ids, vw_preds(Path("runs") / vw_run / f"preds_{set_name}.jsonl"))
        rand = [score(recs, ids, {q: [random.Random(s * 7919 + i).randrange(len(recs[q]["options"]))] for i, q in enumerate(ids)}) for s in range(5)]
        rows["random (5 seeds)"] = {"macro_f1": mean(x["macro_f1"] for x in rand), "acc": mean(x["acc"] for x in rand), "domains": {}}

        print(f"\n== {set_name}: {len(ids)} questions, {stats['requests']} requests, {tok:,} billed tokens (${tok / 1e6 * PRICE_PER_M:.3f}), billed/est {tok / max(est, 1):.2f}  {dict(stats)}")
        print(f"{'model':20s} {'macroF1':>8s} {'acc':>6s}")
        for label, s in rows.items():
            print(f"{label:20s} {s['macro_f1']:8.3f} {s['acc']:6.3f}")
        doms = sorted(rows["OpenAI gpt-6-luna"]["domains"])
        print(f"\n  {'domain (macro-F1)':42s} {'n':>5s} {'OpenAI':>7s} {'VWfull':>7s} {'VWgen':>7s}")
        for d in doms:
            o = rows["OpenAI gpt-6-luna"]["domains"][d]
            print(f"  {d[:42]:42s} {o['n']:5d} {o['macro_f1']:7.3f} {rows['VW full']['domains'][d]['macro_f1']:7.3f} {rows['VW generic']['domains'][d]['macro_f1']:7.3f}")

        # selective accuracy: OpenAI accuracy on its most confident answers
        cq = sorted((q for q in ids if conf.get(q) is not None and oa[q] is not None), key=lambda q: -conf[q])
        sel = {}
        for frac in (0.25, 0.5, 1.0):
            top = cq[: max(1, int(len(cq) * frac))]
            sel[frac] = mean(set(oa[q]) == set(recs[q]["gold"]) for q in top)
        print("  OpenAI accuracy on its most confident answers: " + "  ".join(f"top {int(f * 100)}%={a:.3f}" for f, a in sel.items()))
        results[set_name] = {
            "n_questions": len(ids), "stats": dict(stats), "billed_tokens": tok, "est_tokens": est,
            "models": {k: {"macro_f1": v["macro_f1"], "acc": v["acc"], "domains": {d: x["macro_f1"] for d, x in v["domains"].items()}} for k, v in rows.items()},
            "openai_selective_acc": sel,
        }
    print(f"\nTotal billed: {total_tok:,} tokens = ${total_tok / 1e6 * PRICE_PER_M:.3f}  (billed/estimate {total_tok / max(total_est, 1):.2f})")
    (run / "report.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
