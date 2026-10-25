"""Convert raw datasets into one decision-record format.

Each record is one question about one text:

    id            unique id
    source        "bev" (BEV Decision Mix) or "fast" (fastino/fast-decisions dev)
    config        BEV config name, or the fast-decisions file stem
    orig_split    the split the row came from ("train"/"test"; "dev" for fast)
    domain        BEV domain, or the fast-decisions domain (file stem)
    task          question name, e.g. "citation_intent" or "needs_reply"
    group_id      hash of the normalized text; rows sharing it must stay on
                  one side of any split
    text          the input text (BEV "state", fast-decisions "input")
    qtype         "choice" | "yesno" | "score" | "multi"
    instructions  the question as written (empty for fast-decisions)
    options       list of {"key": str, "desc": str}
    gold          list of gold option indices (length 1 unless qtype == "multi")
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterator

import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
BEV_CONFIGS = ["data", "hard_50k", "numeric_temporal", "skills", "counterfactual_15k"]


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def group_id(text: str) -> str:
    return hashlib.sha1(norm_text(text).encode()).hexdigest()[:16]


def bev_records(root: Path = RAW / "bev-decision") -> Iterator[dict]:
    for cfg in BEV_CONFIGS:
        for split in ("train", "test"):
            df = pd.read_parquet(root / cfg / f"{split}.parquet")
            for i, row in enumerate(df.itertuples(index=False)):
                gid = group_id(row.state)
                for task, q in json.loads(row.questions_json).items():
                    rec = {
                        "id": f"bev/{cfg}/{split}/{i}/{task}",
                        "source": "bev",
                        "config": cfg,
                        "orig_split": split,
                        "domain": row.domain,
                        "task": task,
                        "group_id": gid,
                        "text": row.state,
                        "instructions": q.get("instructions", ""),
                    }
                    t = q["type"]
                    if t == "noul":
                        rec["qtype"] = "yesno"
                        rec["options"] = [{"key": "yes", "desc": ""}, {"key": "no", "desc": ""}]
                        rec["gold"] = [0 if q["label"] else 1]
                    elif t == "choice":
                        keys = list(q["criteria"])
                        rec["qtype"] = "choice"
                        rec["options"] = [{"key": k, "desc": str(q["criteria"][k])} for k in keys]
                        rec["gold"] = [keys.index(q["label"])]
                    elif t == "score":
                        rec["qtype"] = "score"
                        rec["options"] = [{"key": str(j), "desc": str(d)} for j, d in enumerate(q["criteria"])]
                        rec["gold"] = [int(q["label"])]
                    else:
                        raise ValueError(f"unknown question type {t!r}")
                    yield rec


def fast_records(root: Path = RAW / "fast-decisions") -> Iterator[dict]:
    for f in sorted(root.glob("*.jsonl")):
        for i, line in enumerate(f.open()):
            row = json.loads(line)
            gid = group_id(row["input"])
            for h in row["output"]["classifications"]:
                labels = h["labels"]
                gold = [labels.index(g) for g in h["true_label"]]
                if h["multi_label"]:
                    qtype = "multi"
                elif sorted(labels) == ["no", "yes"]:
                    qtype = "yesno"
                    # put yes first to match BEV yes/no records
                    labels = ["yes", "no"]
                    gold = [labels.index(g) for g in h["true_label"]]
                elif all(lbl.isdigit() for lbl in labels):
                    qtype = "score"
                else:
                    qtype = "choice"
                yield {
                    "id": f"fast/{f.stem}/{i}/{h['task']}",
                    "source": "fast",
                    "config": f.stem,
                    "orig_split": "dev",
                    "domain": f.stem,
                    "task": h["task"],
                    "group_id": gid,
                    "text": row["input"],
                    "qtype": qtype,
                    "instructions": "",
                    "options": [{"key": lbl, "desc": ""} for lbl in labels],
                    "gold": gold,
                }


def write_jsonl(records, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def main() -> None:
    n_bev = write_jsonl(bev_records(), PROCESSED / "bev.jsonl")
    n_fast = write_jsonl(fast_records(), PROCESSED / "fast.jsonl")
    print(f"wrote {n_bev} BEV questions and {n_fast} fast-decisions questions to {PROCESSED}")


if __name__ == "__main__":
    main()
