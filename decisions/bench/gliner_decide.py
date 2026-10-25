"""Score exported eval questions with GLiNER2.5-Decide on CPU.

Runs in its own environment (.venv-gliner: CPU PyTorch + gliner2[local]):

    .venv-gliner/bin/python -I bench/gliner_decide.py data/processed/eval_f0_lockbox.jsonl runs/gliner/f0/preds_lockbox.jsonl

Input: records from vwjev.export_eval. All questions about one text go in one
classify call (one "head" per question), as the model card shows. Each head
gets the same information OpenAI got:
    key     the humanised task name (made unique within the text)
    prompt  the instructions, or the task name when there are none
    labels  {option key: description, or the key when there is none}

Texts that share an identical schema are batched together.

Output, one line per question: {"id", "pred": [option index] or null,
"conf": probability of the chosen label or null}. Resumes if the output
exists.
"""

from __future__ import annotations

import collections
import json
import re
import sys
import time
from pathlib import Path

MODEL = "fastino/GLiNER2.5-Decide"
BATCH = 8


def human(task: str) -> str:
    return re.sub(r"[_\s]+", " ", task).strip()


def head(r: dict) -> dict:
    labels = {o["key"]: (o["desc"] or o["key"]) for o in r["options"]}
    instr = r["instructions"].strip()
    return {"labels": labels, "prompt": instr or human(r["task"]) + "?"}


def parse(value, r):
    """GLiNER output for one head -> (index, confidence)."""
    keys = [o["key"] for o in r["options"]]
    conf = None
    if isinstance(value, dict):  # include_confidence=True: {"label": ..., "confidence": ...}
        conf = value.get("confidence")
        value = value.get("label")
    if isinstance(value, list):  # some versions return [{"label", "confidence"}]
        if not value:
            return None, None
        v = value[0]
        if isinstance(v, dict):
            conf, v = v.get("confidence"), v.get("label")
        value = v
    if value in keys:
        return keys.index(value), conf
    return None, conf


def main() -> None:
    src, dest = Path(sys.argv[1]), Path(sys.argv[2])
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if dest.exists():
        with dest.open() as fh:
            done = {json.loads(line)["id"] for line in fh}

    by_row = collections.defaultdict(list)
    with src.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r["id"] not in done:
                by_row[r["id"].rsplit("/", 1)[0]].append(r)

    # one schema per text; group texts whose schema is identical
    groups = collections.defaultdict(list)  # schema json -> [(text, [(name, rec)])]
    for rs in by_row.values():
        tasks, names = {}, []
        for r in rs:
            name = base = human(r["task"])
            k = 2
            while name in tasks:
                name, k = f"{base} {k}", k + 1
            tasks[name] = head(r)
            names.append((name, r))
        groups[json.dumps(tasks, sort_keys=True)].append((rs[0]["text"], names))

    from gliner2 import AutoExtractor

    model = AutoExtractor.from_pretrained(MODEL)
    n_rows = sum(len(v) for v in groups.values())
    print(f"{src.name}: {n_rows} texts in {len(groups)} schema groups", flush=True)

    t0, seen = time.time(), 0
    with dest.open("a") as out:
        for key, items in groups.items():
            tasks = json.loads(key)
            for i in range(0, len(items), BATCH):
                chunk = items[i : i + BATCH]
                results = model.batch_classify_text([t for t, _ in chunk], tasks, batch_size=BATCH, include_confidence=True)
                for (_, names), res in zip(chunk, results):
                    for name, r in names:
                        idx, conf = parse(res.get(name), r)
                        out.write(json.dumps({"id": r["id"], "pred": None if idx is None else [idx], "conf": conf, "raw": res.get(name)}) + "\n")
                seen += len(chunk)
                out.flush()
                if seen % 200 < BATCH:
                    el = time.time() - t0
                    print(f"  {seen}/{n_rows} texts  {el:.0f}s  ({el / seen:.2f}s/text)", flush=True)
    print(f"done: {seen} texts in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
