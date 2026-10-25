"""Collect every number for the comparison report into one JSON file.

Fold 0 throughout, so all models are scored on the same questions:
    seen_test  BEV test rows of domains VW trained on       (VW in-domain)
    dev        BEV test rows of 7 domains VW never saw      (VW out-of-domain)
    lockbox    fast-decisions, a separate dataset            (VW out-of-domain)
OpenAI and GLiNER never trained on any of it (for them every set is
out-of-domain, as far as we know). Multi-label questions are left out
everywhere (OpenAI has no multi-select).

GLiNER ran on all of the lockbox but only a 400-text random sample of each
BEV set (CPU time); those rows are scored for every model separately.

    uv run python -I -m vwjev.final_report
"""

from __future__ import annotations

import collections
import json
import random
import statistics
from pathlib import Path

from vwjev.data import PROCESSED
from vwjev.openai_report import openai_preds, score, vw_preds

RUNS = Path("runs")
OPENAI_RUN = RUNS / "openai" / "luna_f0"
GLINER_DIR = RUNS / "gliner" / "f0"
VW = {"VW full": RUNS / "v5_full_shuf", "VW generic": RUNS / "v6_generic_shuf"}
OUT = RUNS / "final_report.json"


def load_eval(name: str) -> dict[str, dict]:
    with (PROCESSED / f"eval_f0_{name}.jsonl").open() as fh:
        return {r["id"]: r for r in map(json.loads, fh)}


def gliner_preds(path: Path):
    preds, conf = {}, {}
    if path.exists():
        with path.open() as fh:
            for d in map(json.loads, fh):
                preds[d["id"]] = d["pred"]
                conf[d["id"]] = d["conf"]
    return preds, conf


def vw_conf(path: Path) -> dict[str, float]:
    """Margin between the best and second-best option score (lower = better)."""
    out = {}
    with path.open() as fh:
        for d in map(json.loads, fh):
            s = sorted(d["scores"])
            out[d["id"]] = s[1] - s[0] if len(s) > 1 else 0.0
    return out


def random_score(recs, ids, seeds=5):
    runs = [score(recs, ids, {q: [random.Random(s * 7919 + i).randrange(len(recs[q]["options"]))] for i, q in enumerate(ids)}) for s in range(seeds)]
    return {"macro_f1": statistics.mean(x["macro_f1"] for x in runs), "acc": statistics.mean(x["acc"] for x in runs)}


def selective(recs, ids, preds, conf, fracs=(0.1, 0.25, 0.5, 0.75, 1.0)):
    """Accuracy on the most confident fraction of answers."""
    ranked = sorted((q for q in ids if preds.get(q) is not None and conf.get(q) is not None), key=lambda q: -conf[q])
    out = {}
    for f in fracs:
        top = ranked[: max(1, int(len(ranked) * f))]
        out[str(f)] = statistics.mean(set(preds[q]) == set(recs[q]["gold"]) for q in top)
    return out


def brief(s):
    return {"macro_f1": s["macro_f1"], "acc": s["acc"], "n": s.get("n"),
            "domains": {d: {"macro_f1": x["macro_f1"], "acc": x["acc"], "n": x["n"]} for d, x in s.get("domains", {}).items()}}


def table(set_name: str, recs: dict, ids: list[str], models: dict) -> dict:
    rows = {name: brief(score(recs, ids, p)) for name, p in models.items()}
    rows["Random"] = random_score(recs, ids)
    return {"set": set_name, "n_questions": len(ids), "models": rows}


def lockbox_pooled_ids(recs: dict) -> list[str]:
    """Same pooled test questions as lockbox_curve (must match its split)."""
    from vwjev.lockbox_curve import REPS, TEST_ROWS

    by_dom = collections.defaultdict(lambda: collections.defaultdict(list))
    with (PROCESSED / "fast.jsonl").open() as fh:
        for r in map(json.loads, fh):
            by_dom[r["domain"]][r["id"].rsplit("/", 1)[0]].append(r)
    ids = []
    for dom, rows in sorted(by_dom.items()):
        keys = sorted(rows)
        for rep in range(REPS):
            order = keys[:]
            random.Random(f"{dom}/{rep}").shuffle(order)
            ids += [r["id"] for k in order[:TEST_ROWS] for r in rows[k] if r["qtype"] != "multi"]
    return ids


def main() -> None:
    report = {"tables": {}, "selective": {}, "curves": {}, "cost": {}}

    for name in ("seen_test", "dev", "lockbox"):
        recs = load_eval(name)
        oa, oa_conf, stats, tok, _ = openai_preds(OPENAI_RUN / f"raw_{name}.jsonl", recs)
        vw = {k: vw_preds(p / f"preds_{name}.jsonl") for k, p in VW.items()}
        ids = sorted(recs)
        models = {"OpenAI gpt-6-luna": oa, **vw}
        gl_name = name if name == "lockbox" else f"{name}_s400"
        gl, gl_conf = gliner_preds(GLINER_DIR / f"preds_{gl_name}.jsonl")
        report["tables"][name] = table(name, recs, ids, models)
        report["tables"][name]["openai_refusals"] = stats.get("refusal", 0)
        report["cost"][name] = {"requests": stats["requests"], "billed_tokens": tok}

        # rows GLiNER answered (all of lockbox; 400-text sample of BEV sets)
        sample = load_eval(gl_name) if name != "lockbox" else recs
        sids = sorted(set(sample) & set(gl)) if gl else []
        if sids:
            report["tables"][f"{name}_gliner_rows"] = table(f"{name} (GLiNER rows)", recs, sids, {**models, "GLiNER2.5-Decide": gl})
            report["tables"][f"{name}_gliner_rows"]["complete"] = len(sids) == len(sample)

        sel = {"OpenAI gpt-6-luna": selective(recs, ids, oa, oa_conf)}
        for k, p in VW.items():
            sel[k] = selective(recs, ids, vw[k], vw_conf(p / f"preds_{name}.jsonl"))
        if sids:
            sel["GLiNER2.5-Decide (its rows)"] = selective(recs, sids, gl, gl_conf)
        report["selective"][name] = sel

    # learning curves: BEV held-out domains (feedback.py) with OpenAI per domain
    dev_oa = report["tables"]["dev"]["models"]["OpenAI gpt-6-luna"]["domains"]
    dev_vw0 = report["tables"]["dev"]["models"]["VW generic"]["domains"]
    for f in sorted((VW["VW generic"] / "feedback").glob("*.json")):
        d = json.loads(f.read_text())
        report["curves"][d["domain"]] = {
            "curves": {k: [{"n": p["n"], "macro_f1": p["macro_f1"], "acc": p["acc"]} for p in v] for k, v in d["curves"].items()},
            "openai": dev_oa.get(d["domain"], {}).get("macro_f1"),
            "vw_zero_shot": dev_vw0.get(d["domain"], {}).get("macro_f1"),
        }

    # lockbox curve, with OpenAI and GLiNER on the same pooled test questions
    lc = json.loads((VW["VW generic"] / "lockbox_curve.json").read_text())
    lb = load_eval("lockbox")
    pooled = lockbox_pooled_ids(lb)
    oa, *_ = openai_preds(OPENAI_RUN / "raw_lockbox.jsonl", lb)
    check = score(lb, pooled, oa)["macro_f1"]
    assert abs(check - lc["openai"]["macro_f1"]) < 1e-9, (check, lc["openai"]["macro_f1"])
    gl, _ = gliner_preds(GLINER_DIR / "preds_lockbox.jsonl")
    lc["gliner"] = brief(score(lb, pooled, gl)) if gl and all(q in gl for q in pooled) else None
    report["lockbox_curve"] = lc

    # how much in-domain data VW full had: fold-0 training questions per domain
    roles = json.loads((PROCESSED / "splits" / "fold0.json").read_text())
    per_dom = collections.Counter()
    with (PROCESSED / "bev.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            if roles.get(r["id"]) == "train":
                per_dom[r["domain"]] += 1
    report["train_questions_per_domain"] = dict(per_dom)

    # VW across all four folds (zero-shot), from each run's results.json
    four = {}
    for label, base in (("VW full", "v5_full_shuf"), ("VW generic", "v6_generic_shuf")):
        rows = [json.loads((RUNS / (base + ("" if k == 0 else f"_f{k}")) / "results.json").read_text()) for k in range(4)]
        four[label] = {s: statistics.mean(r[s]["macro_f1"] for r in rows) for s in ("dev", "seen_test", "lockbox")}
        four["Random"] = {s: statistics.mean(r[s]["baselines"]["random"] for r in rows) for s in ("dev", "seen_test", "lockbox")}
        four[label]["ms_per_question"] = statistics.mean(r["dev"]["eval_ms_per_question"] for r in rows)
    report["four_fold"] = four

    OUT.write_text(json.dumps(report, indent=1))
    for name, t in report["tables"].items():
        print(f"\n{t['set']}  ({t['n_questions']} questions)")
        for m, s in t["models"].items():
            print(f"  {m:22s} macroF1={s['macro_f1']:.3f} acc={s['acc']:.3f}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
