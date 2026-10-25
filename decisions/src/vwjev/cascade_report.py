"""Summarize the cold-start cascade runs (vwjev.cascade) and the cost model.

Quality is macro-F1 of what the user was served, against the true labels,
pooled over the seeds (each seed is a different arrival order). Cost is the
billed input tokens of the texts sent to OpenAI.

    uv run python -I -m vwjev.cascade_report    # -> runs/cascade/report.json
"""

from __future__ import annotations

import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from statistics import mean

from vwjev.cascade import OUT, POLICIES, load_texts
from vwjev.cost_estimate import PRICE_PER_M
from vwjev.metrics import pooled_scores

DOMAINS = [
    "Spam detection",
    "Tool and workflow decisions",
    "Sentiment, emotion, and moderation",
    "Rule-conditioned decisions",
    "Calibrated probability",
    "Sensory quality rating",
    "Response preference and quality",
]
# text-index windows of the stream (arrival order)
EDGES = [0, 50, 100, 200, 400, 800, 1600, 2400, 3200, 10**9]
STEADY = 1000  # "later on" = the last this many texts of each stream

# --- cost model (illustrative assumptions, stated on the report) ---
# monthly spend scenarios on decision calls: Ramp AI Index (April 2026) median
# and mean monthly business AI spend, plus a round $100K
SPEND_SCENARIOS = {"Ramp median": 2246, "$100K / month": 100_000, "Ramp average": 140_842}
ENG_ONE_OFF = 20_000  # build + rollout: ~4 engineer-weeks, fully loaded
ENG_MONTHLY = 3_000  # upkeep: ~10% of an engineer
VCPU_MONTH = 40  # one always-on cloud vCPU
VCPU_HEADROOM = 3  # provision 3x the measured CPU need (peaks, redundancy)
VW_MS_PER_QUESTION = 0.55  # measured: predict ~0.47 ms; learning on OpenAI-routed texts adds a little


def slug(d: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in d).lower()


def window_of(i: int) -> int:
    for w in range(len(EDGES) - 1):
        if EDGES[w] <= i < EDGES[w + 1]:
            return w
    raise ValueError(i)


def domain_summary(domain: str) -> dict:
    texts = load_texts(domain)
    res = json.loads(gzip.decompress((OUT / f"{slug(domain)}.json.gz").read_bytes()))
    n = len(texts)
    seeds = sorted({int(k.rsplit("/", 1)[1]) for k in res})
    nq = sum(len(t["qs"]) for t in texts)
    out = {"n_texts": n, "n_questions": nq, "seeds": len(seeds), "tokens_per_text": mean(t["tokens"] for t in texts), "policies": {}}

    def items(policy, field, lo=0, hi=10**9, routed=None):
        its = []
        for s in seeds:
            r = res[f"{policy}/{s}"]
            for i, ti in enumerate(r["order"]):
                if not lo <= i < hi or (routed is not None and r["routes"][i] != routed):
                    continue
                for q, p in zip(texts[ti]["qs"], r[field][i]):
                    its.append((q["rec"], p))
        return its

    def f1(its):
        return round(pooled_scores(its)["macro_f1"], 4) if its else None

    def acc(its):
        return round(mean(set(p) == set(r["gold"]) for r, p in its), 4) if its else None

    # OpenAI alone, refusals counted wrong (as in the benchmark tables)
    oa = []
    for t in texts:
        for q in t["qs"]:
            r = q["rec"]
            if q["oa"] is None:
                r = {**r, "options": r["options"] + [{"key": "__refused__", "desc": ""}]}
                oa.append((r, [len(r["options"]) - 1]))
            else:
                oa.append((r, q["oa"]))
    out["openai_alone"] = round(pooled_scores(oa)["macro_f1"], 4)
    out["openai_alone_acc"] = round(mean(set(p) == set(r["gold"]) for r, p in oa), 4)

    for policy in POLICIES:
        if not all(f"{policy}/{s}" in res for s in seeds):  # still running
            continue
        runs = [res[f"{policy}/{s}"] for s in seeds]
        share = mean(sum(r["routes"]) / n for r in runs)
        tok = mean(sum(texts[ti]["tokens"] for i, ti in enumerate(r["order"]) if r["routes"][i]) for r in runs)
        lo = max(0, n - STEADY)
        late_share = mean(sum(r["routes"][lo:]) / (n - lo) for r in runs)
        p = {
            "served_f1": f1(items(policy, "served")),
            "openai_share": round(share, 4),
            "cost_usd": round(tok / 1e6 * PRICE_PER_M, 4),
            "late_served_f1": f1(items(policy, "served", lo)),
            "late_openai_share": round(late_share, 4),
            "late_vw_served_f1": f1(items(policy, "served", lo, routed=0)),
            "served_acc": acc(items(policy, "served")),
            "late_served_acc": acc(items(policy, "served", lo)),
            "windows": [],
        }
        for w in range(len(EDGES) - 1):
            a, b = EDGES[w], min(EDGES[w + 1], n)
            if a >= n:
                break
            # macro-F1 on a short stretch drops rare classes (MIN_SUPPORT), so
            # stretches are compared by accuracy; F1 is kept for reference
            sv, vv = items(policy, "served", a, b), items(policy, "vw", a, b)
            p["windows"].append(
                {
                    "from": a,
                    "to": b,
                    "served_f1": f1(sv),
                    "vw_f1": f1(vv),
                    "served_acc": acc(sv),
                    "vw_acc": acc(vv),
                    "openai_share": round(mean(sum(r["routes"][a:b]) / (b - a) for r in runs), 4),
                }
            )
        if "threshold" in runs[0]:
            # when VW first answers on its own (not counting audit-free gaps
            # before any threshold): first text routed to VW, per order
            p["first_vw_text"] = [next((i for i, x in enumerate(r["routes"]) if x == 0), None) for r in runs]
        out["policies"][policy] = p
    return out


def cost_model(late_share: float, tokens_per_text: float, qs_per_text: float) -> list[dict]:
    """Monthly saving and payback for each spend scenario, if a share
    (1 - late_share) of texts moves from OpenAI to VW."""
    usd_per_text = tokens_per_text / 1e6 * PRICE_PER_M
    rows = []
    for name, spend in SPEND_SCENARIOS.items():
        texts_month = spend / usd_per_text
        cpu_s = texts_month * qs_per_text * VW_MS_PER_QUESTION / 1000
        vcpus = max(1, round(cpu_s / (30 * 86400) * VCPU_HEADROOM + 0.5))
        serving = vcpus * VCPU_MONTH
        saved = spend * (1 - late_share)
        net = saved - serving - ENG_MONTHLY
        rows.append(
            {
                "scenario": name,
                "spend": spend,
                "texts_per_month": texts_month,
                "texts_per_second": texts_month / (30 * 86400),
                "openai_after": spend * late_share,
                "vcpus": vcpus,
                "serving": serving,
                "saved": saved,
                "net_monthly": net,
                "payback_months": ENG_ONE_OFF / net if net > 0 else None,
            }
        )
    return rows


def main() -> None:
    rep = {"domains": {}, "edges": EDGES, "steady": STEADY}
    todo = [d for d in DOMAINS if (OUT / f"{slug(d)}.json.gz").exists()]
    with ProcessPoolExecutor(3) as ex:  # ~0.5 GB each
        for d, summary in zip(todo, ex.map(domain_summary, todo)):
            rep["domains"][d] = summary
    doms = rep["domains"]

    print("\nWhat users were served, whole stream (macro-F1 / share of texts sent to OpenAI / OpenAI cost):")
    print(f"{'policy':16s}" + "".join(f"{d[:12]:>22s}" for d in doms) + f"{'mean F1':>9s}{'share':>7s}")
    agg = {}
    for policy in POLICIES:
        if not all(policy in v["policies"] for v in doms.values()):
            continue
        ps = [v["policies"][policy] for v in doms.values()]
        cells = "".join(f"{p['served_f1']:>8.3f} {p['openai_share']:>5.0%} ${p['cost_usd']:<6.3f}" for p in ps)
        agg[policy] = {
            "served_f1": mean(p["served_f1"] for p in ps),
            "openai_share": mean(p["openai_share"] for p in ps),
            "late_served_f1": mean(p["late_served_f1"] for p in ps),
            "late_openai_share": mean(p["late_openai_share"] for p in ps),
        }
        print(f"{policy:16s}{cells}{agg[policy]['served_f1']:>9.3f}{agg[policy]['openai_share']:>7.0%}")
    print(f"{'OpenAI alone':16s}" + "".join(f"{v['openai_alone']:>8.3f}{'':14s}" for v in doms.values()))
    rep["mean"] = agg

    print(f"\nLater on (last {STEADY} texts): served F1 / share to OpenAI / F1 of the answers VW served")
    for policy in POLICIES:
        if policy not in agg:
            continue
        ps = [v["policies"][policy] for v in doms.values()]
        print(
            f"{policy:16s}"
            + "".join(f"{p['late_served_f1']:>8.3f} {p['late_openai_share']:>5.0%} {p['late_vw_served_f1'] or float('nan'):>6.3f} " for p in ps)
        )

    print("\nVW in the background, learning from OpenAI vs from true labels (macro-F1 per window of the stream):")
    for d, v in doms.items():
        o, g = v["policies"]["openai"]["windows"], v["policies"]["gold"]["windows"]
        print(f"  {d[:30]:30s} OpenAI alone {v['openai_alone']:.3f}")
        print("    " + " ".join(f"{w['from']:>5d}+" for w in o))
        print("    " + " ".join(f"{w['vw_f1']:6.3f}" for w in o) + "   learns from OpenAI")
        print("    " + " ".join(f"{w['vw_f1']:6.3f}" for w in g) + "   learns from true labels")

    # cost model on a quality-preserving rule: the best gate rule whose
    # later-on served F1 stays within 0.02 of always-OpenAI, averaged over domains
    base = agg["openai"]["late_served_f1"]
    gates = [p for p in agg if p.startswith("gate") and agg[p]["late_served_f1"] >= base - 0.02]
    pick = min(gates, key=lambda p: agg[p]["late_openai_share"]) if gates else "gate:0.95:0.05"
    tpt = mean(v["tokens_per_text"] for v in doms.values())
    qpt = mean(v["n_questions"] / v["n_texts"] for v in doms.values())
    rep["cost_model"] = {
        "policy": pick,
        "late_openai_share": agg[pick]["late_openai_share"],
        "late_served_f1": agg[pick]["late_served_f1"],
        "openai_late_served_f1": base,
        "tokens_per_text": tpt,
        "questions_per_text": qpt,
        "assumptions": {
            "eng_one_off": ENG_ONE_OFF,
            "eng_monthly": ENG_MONTHLY,
            "vcpu_month": VCPU_MONTH,
            "vcpu_headroom": VCPU_HEADROOM,
            "vw_ms_per_question": VW_MS_PER_QUESTION,
            "price_per_m": PRICE_PER_M,
        },
        "scenarios": cost_model(agg[pick]["late_openai_share"], tpt, qpt),
        "break_even_spend": ENG_MONTHLY / (1 - agg[pick]["late_openai_share"]),
    }
    cm = rep["cost_model"]
    print(f"\nCost model on {pick}: later-on share to OpenAI {cm['late_openai_share']:.0%}, served F1 {cm['late_served_f1']:.3f} vs {base:.3f} always-OpenAI")
    print(f"  ${tpt / 1e6 * PRICE_PER_M * 1e6:.0f} per million texts; break-even monthly decision spend ${cm['break_even_spend']:,.0f}")
    for s in cm["scenarios"]:
        pb = f"{s['payback_months']:.1f} months" if s["payback_months"] else "never"
        print(
            f"  {s['scenario']:14s} ${s['spend']:>9,.0f}/mo = {s['texts_per_second']:>7,.0f} texts/s; saves ${s['saved']:>9,.0f}, "
            f"VW servers {s['vcpus']} vCPU ${s['serving']:,.0f}, net ${s['net_monthly']:>9,.0f}/mo, payback {pb}"
        )
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    print(f"\n-> {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
