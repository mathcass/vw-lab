"""Cold-start cascade: OpenAI answers first, VW learns from its answers and
takes over the texts it is confident about.

Product scenario: no labelled data on day one. Each incoming text is routed
either to the OpenAI Decisions API (paid; the user gets OpenAI's answer and
VW learns from it as if it were the label) or to VW (free; the user gets
VW's answer). True labels are used only to score what the user was served,
never for training (except in the `gold` comparison run).

VW answers every text in the background either way, so each OpenAI call
also tells us whether VW would have agreed. That agreement log is what a
product can use to decide when VW is ready, without any true labels.

Routing rules (one request per text, so routing is per text):

    openai          always OpenAI; VW learns in the background
    gold            always OpenAI, but VW learns from the true labels
                    (what VW could reach with human corrections instead)
    warmup:N:R      OpenAI for the first N texts, then VW; afterwards a random
                    R share of texts still goes to OpenAI (audits), and VW
                    keeps learning from those
    gate:A:R        VW answers a text when its confidence clears a threshold
                    picked from the agreement log so that, on past OpenAI
                    calls above that threshold, VW agreed with OpenAI at least
                    A of the time; everything else goes to OpenAI. A random R
                    share always goes to OpenAI so the log keeps seeing
                    confident texts.

VW's confidence on a text is the smallest gap between its best and second-
best option scores over the text's questions. When OpenAI refuses a
question, the user gets VW's answer for it and VW does not learn from it.

VW starts from scratch (no pre-training): in the earlier feedback runs the
pre-trained and scratch models were level after ~100 examples in every
domain, and a scratch model is small and fast to simulate.

    uv run python -I -m vwjev.cascade prepare            # after openai_decide --sets stream
    uv run python -I -m vwjev.cascade run --domain "Spam detection"
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import random
import time
from pathlib import Path

from vowpalwabbit import Workspace

from vwjev.data import PROCESSED
from vwjev.features import FeatureConfig
from vwjev.feedback import lines_with_f
from vwjev.openai_decide import load_set, row_key
from vwjev.openai_report import openai_preds
from vwjev.run import decide, option_scores

FOLD = 0
OPENAI_RUN = Path("runs/openai/luna_f0")
CFG_FROM = Path("runs/v6_generic_shuf/feature_config.json")
OUT = Path("runs/cascade")
CACHE = PROCESSED / f"cascade_f{FOLD}.jsonl"
SCRATCH_VW = "--csoaa_ldf mc -b 24 -q fl -q sl -q qm -q qp --ignore t --quiet"
SEEDS = 3

POLICIES = (
    ["openai", "gold"]
    + [f"warmup:{n}:{r}" for n in (50, 100, 200, 400, 800, 1600) for r in (0, 0.1)]
    + [f"gate:{a}:{r}" for a in (0.8, 0.9, 0.95) for r in (0.02, 0.05, 0.1, 0.2)]
)
GATE_MIN_N = 30  # log entries above a threshold needed before trusting it
GATE_WINDOW = 1000  # most recent OpenAI calls kept in the agreement log
GATE_EVERY = 25  # recompute the threshold every this many texts


def prepare() -> None:
    """One line per text with OpenAI answers: {domain, row, tokens, qs: [{rec, oa}]}."""
    n = 0
    with CACHE.open("w") as out:
        for name in ("stream", "dev"):
            recs = {r["id"]: r for r in load_set(name, FOLD)}
            preds, _, stats, _, _ = openai_preds(OPENAI_RUN / f"raw_{name}.jsonl", recs)
            print(name, dict(stats))
            tokens = {}
            with (OPENAI_RUN / f"raw_{name}.jsonl").open() as fh:
                for d in map(json.loads, fh):
                    if d["status"] == 200:
                        tokens[d["row"]] = d["response"]["usage"]["input_tokens"]
            by_row = collections.defaultdict(list)
            for qid, r in recs.items():
                by_row[row_key(qid)].append(r)
            for row, rs in by_row.items():
                if row not in tokens:  # never sent (e.g. budget stop)
                    continue
                qs = [{"rec": r, "oa": preds.get(r["id"])} for r in rs]
                out.write(json.dumps({"domain": rs[0]["domain"], "row": row, "set": name, "tokens": tokens[row], "qs": qs}) + "\n")
                n += 1
    print(f"{n} texts -> {CACHE}")


def load_texts(domain: str) -> list[dict]:
    with CACHE.open() as fh:
        return [t for t in map(json.loads, fh) if t["domain"] == domain]


def margin(scores: list[float]) -> float:
    if len(scores) < 2:
        return float("inf")
    a, b = sorted(scores)[:2]
    return b - a


def teacher_costs(rec: dict, pred: list[int]) -> list[float]:
    return [0.0 if i in pred else 1.0 for i in range(len(rec["options"]))]


class Gate:
    """Pick the lowest confidence threshold at which VW agreed with OpenAI
    on at least `target` of logged OpenAI calls above it."""

    def __init__(self, target: float):
        self.target = target
        self.log: collections.deque = collections.deque(maxlen=GATE_WINDOW)
        self.threshold = float("inf")

    def update(self) -> None:
        entries = sorted(self.log, key=lambda e: -e[0])
        agree, best = 0, float("inf")
        for k, (m, ok) in enumerate(entries, 1):
            agree += ok
            # thresholds sit between distinct margins
            if k >= GATE_MIN_N and agree / k >= self.target and (k == len(entries) or entries[k][0] < m):
                best = m
        self.threshold = best


def simulate(texts: list[dict], policy: str, seed: int, cfg: FeatureConfig) -> dict:
    kind, *params = policy.split(":")
    rng = random.Random(f"{policy}/{seed}")
    order = list(range(len(texts)))
    random.Random(f"order/{texts[0]['domain']}/{seed}").shuffle(order)
    ws = Workspace(SCRATCH_VW)
    gate = Gate(float(params[0])) if kind == "gate" else None
    audit = float(params[1]) if kind in ("warmup", "gate") else 0.0
    routes, served, vwp, thresholds = [], [], [], []
    for i, ti in enumerate(order):
        t = texts[ti]
        preds, gaps = [], []
        for q in t["qs"]:
            s = option_scores(ws, lines_with_f(q["rec"], cfg))
            preds.append(decide(q["rec"], s))
            gaps.append(margin(s))
        conf = min(gaps)
        if kind in ("openai", "gold"):
            to_oa = True
        elif rng.random() < audit:
            to_oa = True
        elif kind == "warmup":
            to_oa = i < int(params[0])
        else:
            if i % GATE_EVERY == 0:
                gate.update()
            to_oa = conf < gate.threshold
        if gate is not None:
            thresholds.append(gate.threshold)
        routes.append(int(to_oa))
        out = []
        for q, p in zip(t["qs"], preds):
            oa = q["oa"]
            out.append(oa if to_oa and oa is not None else p)
        served.append(out)
        vwp.append(preds)
        if to_oa:
            answered = [(q, p) for q, p in zip(t["qs"], preds) if q["oa"] is not None]
            if gate is not None and answered:
                gate.log.append((conf, all(q["oa"] == p for q, p in answered)))
            for q, _ in answered:
                label = q["rec"]["gold"] if kind == "gold" else q["oa"]
                ws.learn(lines_with_f(q["rec"], cfg, teacher_costs(q["rec"], label)))
    ws.finish()
    res = {"order": order, "routes": routes, "served": served, "vw": vwp}
    if gate is not None:
        res["threshold"] = [x if x != float("inf") else None for x in thresholds]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    r = sub.add_parser("run")
    r.add_argument("--domain", required=True)
    r.add_argument("--policies", nargs="+", default=POLICIES)
    r.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare()
        return

    cfg = FeatureConfig(**json.loads(CFG_FROM.read_text()))
    texts = load_texts(args.domain)
    print(f"{args.domain}: {len(texts)} texts, {sum(len(t['qs']) for t in texts)} questions", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() else "_" for c in args.domain).lower()
    path = OUT / f"{slug}.json.gz"
    results = json.loads(gzip.decompress(path.read_bytes())) if path.exists() else {}
    for seed in range(args.seeds):
        for policy in args.policies:
            key = f"{policy}/{seed}"
            if key in results:  # resume
                continue
            t0 = time.time()
            results[key] = simulate(texts, policy, seed, cfg)
            share = sum(results[key]["routes"]) / len(texts)
            print(f"  {key:20s} to OpenAI {share:6.1%}  {time.time() - t0:5.0f}s", flush=True)
            path.write_bytes(gzip.compress(json.dumps(results).encode()))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
