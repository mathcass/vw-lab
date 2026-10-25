"""Estimate what scoring our datasets with the OpenAI Decisions API would cost.

Pricing (developers.openai.com/api/docs/guides/decisions, checked 2026-10-19):
$0.10 per 1M input tokens, no output charge. Long-context multipliers and
regional premiums are not modelled.

Assumptions:
- One request per text; all questions about that text share the request.
- Tokens counted with tiktoken's o200k_base as a stand-in (gpt-6-luna's
  tokenizer isn't published).
- Each request pays the text plus, per question, its name, instructions and
  every option's key + description, plus fixed JSON/request overhead
  (REQUEST_OVERHEAD, QUESTION_OVERHEAD), which are guesses.

    uv run python -I -m vwjev.cost_estimate
"""

from __future__ import annotations

import collections
import json

import tiktoken

from vwjev.data import PROCESSED

PRICE_PER_M = 0.10
REQUEST_OVERHEAD = 20
QUESTION_OVERHEAD = 12
_enc = tiktoken.get_encoding("o200k_base")


def ntok(s: str) -> int:
    return len(_enc.encode(s, disallowed_special=())) if s else 0


def row_key(qid: str) -> str:
    return qid.rsplit("/", 1)[0]


def main() -> None:
    text_tok: dict[str, int] = {}
    q_tok: dict[str, int] = {}
    q_row: dict[str, str] = {}
    group = collections.defaultdict(set)  # label -> question ids
    for path in (PROCESSED / "bev.jsonl", PROCESSED / "fast.jsonl"):
        with path.open() as fh:
            for line in fh:
                r = json.loads(line)
                rk = row_key(r["id"])
                if rk not in text_tok:
                    text_tok[rk] = ntok(r["text"])
                opts = " ".join(f"{o['key']} {o['desc']}" for o in r["options"])
                q_tok[r["id"]] = ntok(f"{r['task']} {r['instructions']} {opts}") + QUESTION_OVERHEAD
                q_row[r["id"]] = rk
                if r["source"] == "bev":
                    group[f"BEV {r['config']} / {r['orig_split']}"].add(r["id"])
                    group[f"BEV all configs / {r['orig_split']}"].add(r["id"])
                    group["BEV everything (train + test)"].add(r["id"])
                else:
                    group["fast-decisions lockbox"].add(r["id"])
    for k in range(4):
        roles = json.loads((PROCESSED / "splits" / f"fold{k}.json").read_text())
        ev = {q for q, role in roles.items() if role in ("dev", "seen_test")}
        group[f"fold {k} eval sets (dev + seen_test)"] = ev
        group["all 4 folds' eval sets"] |= ev

    print(f"{'set':40s} {'requests':>9s} {'questions':>10s} {'tokens':>14s} {'cost USD':>9s} {'max req tok':>11s}")
    for name in sorted(group, key=lambda n: (not n.startswith("fast"), not n.startswith("fold"), n)):
        qids = group[name]
        rows = collections.defaultdict(int)
        for q in qids:
            rows[q_row[q]] += q_tok[q]
        per_req = [text_tok[rk] + qt + REQUEST_OVERHEAD for rk, qt in rows.items()]
        tokens = sum(per_req)
        print(f"{name:40s} {len(rows):9d} {len(qids):10d} {tokens:14,d} {tokens / 1e6 * PRICE_PER_M:9.2f} {max(per_req):11,d}")


if __name__ == "__main__":
    main()
