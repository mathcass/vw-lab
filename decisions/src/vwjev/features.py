"""Turn a decision record into VW multi-line (csoaa_ldf) examples.

Layout (one block per question):

    shared |t <text tokens> |q <question tokens> [|s <shared match feats>]
    1:c1   |l <option 1 tokens> [|p <position feats>] [|m <match feats>]
    2:c2   |l <option 2 tokens> ...

Text-only features add the same score to every option and cancel out, so
every useful signal has to come from option namespaces or interactions
between shared and option namespaces (e.g. -q tl).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)  # splits snake_case and drops |, :, punctuation
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")

STOP = frozenset(
    """a an the is are was were be been of to in on for with and or not no yes this that
    it its as at by from does do did which what how should would could can will there their
    any all some given about into than then so if""".split()
)


def tokens(s: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(_CAMEL.sub(" ", s))]


@dataclass
class FeatureConfig:
    max_text_tokens: int = 256
    # per-option overlap with the text (generic: works for any label set)
    match: bool = True
    # add score-scale position features
    position: bool = True
    # for yes/no questions, put the question words into each option, tagged
    # with its polarity, so -q tl can relate text words to the proposition
    yesno_crossed: bool = True
    # word -> IDF, from training texts only; None weights every word 1
    idf: dict | None = None


def option_tokens(rec: dict, i: int) -> list[str]:
    opt = rec["options"][i]
    return tokens(opt["key"]) + tokens(opt["desc"])


def question_tokens(rec: dict) -> list[str]:
    return tokens(rec["instructions"]) + tokens(rec["task"])


def to_lines(rec: dict, cfg: FeatureConfig, costs: list[float] | None = None) -> list[str]:
    """Build the VW block. `costs` None gives dummy 0 costs (prediction).

    Dummy costs are needed because csoaa_ldf crashes (segfault) in VW 9.11.9
    when predicting on unlabeled multi-line examples.
    """
    text_toks = tokens(rec["text"])[: cfg.max_text_tokens]
    q_toks = question_tokens(rec)
    shared = f"shared |t {' '.join(text_toks)} |q {' '.join(q_toks)}"
    text_set = set(text_toks)
    text_pre = {t[:5] for t in text_toks}
    q_content = [t for t in q_toks if t not in STOP]
    if cfg.match and q_content:
        frac = sum(t in text_set for t in q_content) / len(q_content)
        shared += f" |s q_in_text:{frac:.3f}"
    lines = [shared]

    k = len(rec["options"])
    yesno = rec["qtype"] == "yesno"
    opt_toks = [option_tokens(rec, i) for i in range(k)]
    match = [_match_stats(o, text_set, text_pre, cfg.idf) for o in opt_toks] if cfg.match else None
    if match:
        best = {key: max(m.get(key, 0.0) for m in match) for key in ("w", "w5")}
    for i in range(k):
        cost = 0.0 if costs is None else costs[i]
        o_toks = opt_toks[i]
        if yesno and cfg.yesno_crossed:
            pol = "Y" if rec["options"][i]["key"] == "yes" else "N"
            o_toks = o_toks + [f"{pol}_{t}" for t in q_content]
        parts = [f"{i + 1}:{cost:g} |l {' '.join(o_toks) or 'empty'}"]
        if cfg.position:
            feats = [f"k{min(k, 20)}"]
            if rec["qtype"] == "score" and k > 1:
                feats.append(f"pos:{i / (k - 1):.3f}")
            feats.append("first" if i == 0 else ("last" if i == k - 1 else "mid"))
            parts.append(f"|p {' '.join(feats)}")
        if match:
            m = match[i]
            if m:
                feats = [f"{name}:{v:.3f}" for name, v in m.items()]
                # relative to the other options offered for this question
                for key in ("w", "w5"):
                    gap = m[key] - best[key]
                    feats.append(f"{key}_gap:{gap:.3f}")
                    if best[key] > 0 and gap == 0:
                        feats.append(f"{key}_best")
                parts.append(f"|m {' '.join(feats)}")
            else:
                parts.append("|m no_content")
        lines.append(" ".join(parts))
    return lines


def _match_stats(o_toks, text_set, text_pre, idf) -> dict[str, float]:
    """Overlap between one option's words and the text; empty if no content words."""
    content = [t for t in o_toks if t not in STOP]
    if not content:
        return {}
    wt = [idf.get(t, idf.get("__max__", 1.0)) if idf else 1.0 for t in content]
    total = sum(wt)
    hit = [t in text_set for t in content]
    hit5 = [t[:5] in text_pre for t in content]
    return {
        "ovl": sum(hit) / len(content),
        "ovl5": sum(hit5) / len(content),
        "w": sum(w for w, h in zip(wt, hit) if h) / total,
        "w5": sum(w for w, h in zip(wt, hit5) if h) / total,
    }


def compute_idf(texts, max_tokens: int = 256) -> dict[str, float]:
    """IDF from training texts only (unique texts)."""
    import math
    from collections import Counter

    df = Counter()
    n = 0
    for t in texts:
        df.update(set(tokens(t)[:max_tokens]))
        n += 1
    idf = {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items() if c >= 2}
    idf["__max__"] = math.log(n + 1) + 1  # unseen words count as rare
    return idf


def shuffle_options(rec: dict, rng) -> dict:
    """Randomly reorder options (except score scales, whose order is meaning).

    The BEV mix lists the right answer first far more often than chance in
    several domains, so unshuffled training learns a position shortcut.
    """
    if rec["qtype"] == "score" or len(rec["options"]) < 2:
        return rec
    order = list(range(len(rec["options"])))
    rng.shuffle(order)
    new = dict(rec)
    new["options"] = [rec["options"][i] for i in order]
    new["gold"] = sorted(order.index(g) for g in rec["gold"])
    return new


def supervised_costs(rec: dict) -> list[float]:
    gold = set(rec["gold"])
    return [0.0 if i in gold else 1.0 for i in range(len(rec["options"]))]
