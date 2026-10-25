"""Scoring.

Macro-F1 is pooled per domain: every prediction becomes a class label
"<task base>=<option key>", and macro-F1 is taken over the classes that
appear in gold or predictions. Domain scores are then averaged, so big
domains don't dominate. Numbered task variants ("request_intent_record_3",
"question_5") are merged into their base name; per-task F1 was mostly noise
when a domain splits into many tiny tasks.

Multi-label questions (fast-decisions only) are scored by exact set match
for accuracy; for F1 each (task, key) is a separate yes/no class.
"""

from __future__ import annotations

import collections
import re
from statistics import mean

from sklearn.metrics import f1_score

_NUMBERED = re.compile(r"(_record)?_\d+$")
# classes with fewer gold examples than this are left out of the macro
# average (they still count toward accuracy); one-off classes made the
# pooled macro-F1 swing on lucky guesses
MIN_SUPPORT = 5


def task_base(task: str) -> str:
    return _NUMBERED.sub("", task)


def _labels(rec: dict, idxs: list[int]) -> list[str]:
    base = task_base(rec["task"])
    return [f"{base}={rec['options'][i]['key']}" for i in idxs]


def pooled_scores(items: list[tuple[dict, list[int]]]) -> dict:
    single = [(r, p) for r, p in items if r["qtype"] != "multi"]
    multi = [(r, p) for r, p in items if r["qtype"] == "multi"]
    f1s, weights = [], []
    if single:
        y_true = [_labels(r, r["gold"])[0] for r, _ in single]
        y_pred = [_labels(r, p)[0] for r, p in single]
        support = collections.Counter(y_true)
        labels = sorted(lbl for lbl, c in support.items() if c >= MIN_SUPPORT)
        if labels:
            f1s.append(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
            weights.append(len(single))
    if multi:
        keys = sorted({lbl for r, _ in multi for lbl in _labels(r, list(range(len(r["options"]))))})
        y_true = [[int(k in set(_labels(r, r["gold"]))) for k in keys] for r, _ in multi]
        y_pred = [[int(k in set(_labels(r, p))) for k in keys] for r, p in multi]
        f1s.append(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weights.append(len(multi))
    f1 = sum(f * w for f, w in zip(f1s, weights)) / sum(weights) if weights else float("nan")
    acc = mean(set(p) == set(r["gold"]) for r, p in items)
    return {"macro_f1": f1, "acc": acc, "n": len(items)}


def summarize(items: list[tuple[dict, list[int]]]) -> dict:
    by_domain = collections.defaultdict(list)
    by_type = collections.defaultdict(list)
    for rec, pred in items:
        by_domain[rec["domain"]].append((rec, pred))
        by_type[rec["qtype"]].append((rec, pred))
    domains = {d: pooled_scores(v) for d, v in by_domain.items()}
    # by question type: pooled within each domain, then averaged over domains
    types = {}
    for t, its in by_type.items():
        per_dom = collections.defaultdict(list)
        for rec, pred in its:
            per_dom[rec["domain"]].append((rec, pred))
        vals = [pooled_scores(v)["macro_f1"] for v in per_dom.values()]
        vals = [x for x in vals if x == x]  # skip domains with no class of MIN_SUPPORT examples
        types[t] = {"macro_f1": mean(vals) if vals else float("nan"), "n": len(its)}
    return {
        "macro_f1": mean(vals) if (vals := [d["macro_f1"] for d in domains.values() if d["macro_f1"] == d["macro_f1"]]) else float("nan"),
        "acc": mean(d["acc"] for d in domains.values()),
        "n": len(items),
        "domains": domains,
        "by_qtype": types,
    }
