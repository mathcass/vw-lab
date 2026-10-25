"""Assign BEV questions to train / dev / seen_test per domain fold, and check
the fast-decisions lockbox for leaks.

For fold k:
    dev        BEV original-test rows from the fold's held-out domains
               (domains the model never trains on: the zero-shot number)
    train      BEV original-train rows from every other domain
    seen_test  BEV original-test rows from the training domains
               (in-distribution check)
Rows whose text (group_id) appears on both sides are dropped from train.
Dev and seen_test are capped per domain so evaluation stays fast.

The fast-decisions lockbox is never assigned here; it is only scored.
"""

from __future__ import annotations

import collections
import json
import random
import re
from pathlib import Path

from vwjev.data import PROCESSED, norm_text

N_FOLDS = 4
DEV_CAP_PER_DOMAIN = 1500
SEEN_CAP_PER_DOMAIN = 300
SHINGLE = 8


def iter_light(path: Path):
    """Yield the small fields only, to keep memory low."""
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            yield r["id"], r["domain"], r["orig_split"], r["group_id"], r["text"]


def shingles(text: str, n: int = SHINGLE) -> set[int]:
    toks = re.findall(r"\w+", norm_text(text))
    return {hash(tuple(toks[i : i + n])) for i in range(max(0, len(toks) - n + 1))}


def assign_folds(domain_counts: dict[str, int]) -> dict[str, int]:
    folds = [0] * N_FOLDS
    out = {}
    for d, c in sorted(domain_counts.items(), key=lambda kv: -kv[1]):
        k = min(range(N_FOLDS), key=lambda i: folds[i])
        out[d] = k
        folds[k] += c
    return out


def main() -> None:
    bev = PROCESSED / "bev.jsonl"
    fast = PROCESSED / "fast.jsonl"

    # Lockbox shingles and exact-text hashes
    fast_groups, fast_sh = set(), set()
    for _, _, _, gid, text in iter_light(fast):
        fast_groups.add(gid)
        fast_sh |= shingles(text)

    domain_counts = collections.Counter()
    rows = []  # (id, domain, orig_split, group_id)
    leak_exact, leak_shingle = set(), set()
    seen_text = set()
    for qid, dom, split, gid, text in iter_light(bev):
        domain_counts[dom] += 1
        rows.append((qid, dom, split, gid))
        if gid in seen_text:
            continue
        seen_text.add(gid)
        if gid in fast_groups:
            leak_exact.add(gid)
        elif fast_sh and shingles(text) & fast_sh:
            leak_shingle.add(gid)
    leaked = leak_exact | leak_shingle
    print(f"BEV texts matching the lockbox: exact={len(leak_exact)} 8-word-overlap={len(leak_shingle)} (dropped)")

    # Texts that appear in both original splits
    by_split = collections.defaultdict(set)
    for _, _, split, gid in rows:
        by_split[split].add(gid)
    cross = by_split["train"] & by_split["test"]
    print(f"texts in both BEV train and test splits: {len(cross)}")

    fold_of = assign_folds(domain_counts)
    out_dir = PROCESSED / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "domain_folds.json").write_text(json.dumps(fold_of, indent=1, sort_keys=True))

    for k in range(N_FOLDS):
        rng = random.Random(k)
        dev_by_dom = collections.defaultdict(list)
        seen_by_dom = collections.defaultdict(list)
        train = []
        for qid, dom, split, gid in rows:
            if gid in leaked:
                continue
            held = fold_of[dom] == k
            if split == "test":
                (dev_by_dom if held else seen_by_dom)[dom].append((qid, gid))
            elif not held:
                train.append((qid, gid))

        def cap(by_dom, n):
            # sample whole texts so one text's questions stay together
            picked = []
            for dom, items in sorted(by_dom.items()):
                gids = sorted({g for _, g in items})
                rng.shuffle(gids)
                keep, count = set(), 0
                per_gid = collections.Counter(g for _, g in items)
                for g in gids:
                    if count >= n:
                        break
                    keep.add(g)
                    count += per_gid[g]
                picked += [(q, g) for q, g in items if g in keep]
            return picked

        dev = cap(dev_by_dom, DEV_CAP_PER_DOMAIN)
        seen = cap(seen_by_dom, SEEN_CAP_PER_DOMAIN)
        eval_gids = {g for _, g in dev} | {g for _, g in seen}
        train_kept = [q for q, g in train if g not in eval_gids]
        dropped = len(train) - len(train_kept)

        roles = {q: "train" for q in train_kept}
        roles.update({q: "dev" for q, _ in dev})
        roles.update({q: "seen_test" for q, _ in seen})
        (out_dir / f"fold{k}.json").write_text(json.dumps(roles))
        held = sorted(d for d, f in fold_of.items() if f == k)
        print(
            f"fold {k}: train={len(train_kept)} (dropped {dropped} sharing eval text) "
            f"dev={len(dev)} seen_test={len(seen)}\n  held-out domains: {held}"
        )


if __name__ == "__main__":
    main()
