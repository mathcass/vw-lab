"""Write the exact eval questions (same text cut, same option order) that VW
and OpenAI were scored on, for models run outside this environment, plus a
seeded random sample of texts from each BEV set for slow models.

    uv run python -I -m vwjev.export_eval --fold 0
"""

from __future__ import annotations

import argparse
import collections
import json
import random

from vwjev.data import PROCESSED
from vwjev.openai_decide import load_set


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--sample", type=int, default=400, help="texts per BEV set in the sample files")
    args = ap.parse_args()
    for name in ("lockbox", "dev", "seen_test"):
        recs = load_set(name, args.fold)
        path = PROCESSED / f"eval_f{args.fold}_{name}.jsonl"
        with path.open("w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        print(f"{name}: {len(recs)} questions -> {path}")
        if name == "lockbox":
            continue
        # random sample of texts for slow models (GLiNER on CPU)
        rows = collections.defaultdict(list)
        for r in recs:
            rows[r["id"].rsplit("/", 1)[0]].append(r)
        keep = random.Random(0).sample(sorted(rows), args.sample)
        spath = PROCESSED / f"eval_f{args.fold}_{name}_s{args.sample}.jsonl"
        with spath.open("w") as fh:
            for k in keep:
                for r in rows[k]:
                    fh.write(json.dumps(r) + "\n")
        print(f"{name}: {args.sample}-text sample, {sum(len(rows[k]) for k in keep)} questions -> {spath}")


if __name__ == "__main__":
    main()
