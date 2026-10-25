"""Score our eval sets with the OpenAI Decisions API (POST /v1/decisions).

One request per text; all questions about that text go in one request, as
the API intends. The text is cut to the same 4,000 characters VW sees.

Question mapping (our qtype -> API type):
- yesno  -> predicate; "yes" when probability >= 0.5
- choice -> choice; options become {value: key, description: desc or key}
- score  -> score; levels in index order (level index = our option index);
            the argmax level is the prediction
- multi  -> skipped (the API has no multi-select); fast-decisions only

Lockbox questions have no instructions, only task names, so the humanised
task name is the instruction. Every raw response is saved, so results can be
re-scored without spending again.

Spend is capped: before each request we reserve a padded token estimate
(tiktoken o200k_base x RESERVE_FACTOR + RESERVE_OVERHEAD); when the response
arrives the reservation is swapped for the billed `usage.input_tokens`. No new
request starts once billed + reserved spend would pass --budget.

The API key is read from OPENAI_API_KEY and never printed or saved.

    uv run python -I -m vwjev.openai_decide --sets lockbox --limit 5 --name smoke
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import os
import random
import re
import threading
import time
import zlib
from pathlib import Path

import requests

from vwjev.cost_estimate import PRICE_PER_M, ntok
from vwjev.data import PROCESSED
from vwjev.features import shuffle_options
from vwjev.run import MAX_CHARS

URL = "https://api.openai.com/v1/decisions"
MODEL = "gpt-6-luna"
OUT = Path("runs/openai")
TYPE = {"yesno": "predicate", "choice": "choice", "score": "score"}
# smoke test: billed input_tokens ran ~130 above our o200k estimate per request
# (305 vs 176 on short texts); reserve generously before each request
RESERVE_OVERHEAD = 200
RESERVE_FACTOR = 1.2


def human(task: str) -> str:
    return re.sub(r"[_\s]+", " ", task).strip()


def row_key(qid: str) -> str:
    return qid.rsplit("/", 1)[0]


# "stream" set: BEV train texts of the fold's held-out domains, for the
# cold-start cascade simulation (vwjev.cascade). Up to this many texts per
# domain, seeded random sample; texts that also appear in dev are excluded.
STREAM_TEXTS = 3000


def load_stream(fold: int) -> list[dict]:
    roles = json.loads((PROCESSED / "splits" / f"fold{fold}.json").read_text())
    held, dev_gids, rows = set(), set(), []
    with (PROCESSED / "bev.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            role = roles.get(r["id"])
            if role == "dev":
                held.add(r["domain"])
                dev_gids.add(r["group_id"])
            elif role is None and r["orig_split"] == "train":
                rows.append(r)
    by_dom = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["domain"] in held and r["group_id"] not in dev_gids:
            by_dom[r["domain"]][r["group_id"]].append(r)
    recs = []
    for dom in sorted(by_dom):
        gids = sorted(by_dom[dom])
        random.Random(f"stream/{fold}/{dom}").shuffle(gids)
        for g in gids[:STREAM_TEXTS]:
            for r in by_dom[dom][g]:
                r["text"] = r["text"][:MAX_CHARS]
                recs.append(shuffle_options(r, random.Random(zlib.crc32(r["id"].encode()))))
    return [r for r in recs if r["qtype"] in TYPE]


def load_set(name: str, fold: int) -> list[dict]:
    """Same records (same text cut, same option shuffle) VW was scored on."""
    if name == "stream":
        return load_stream(fold)
    if name == "lockbox":
        with (PROCESSED / "fast.jsonl").open() as fh:
            recs = [json.loads(line) for line in fh]
    else:
        roles = json.loads((PROCESSED / "splits" / f"fold{fold}.json").read_text())
        recs = []
        with (PROCESSED / "bev.jsonl").open() as fh:
            for line in fh:
                r = json.loads(line)
                if roles.get(r["id"]) == name:
                    r["text"] = r["text"][:MAX_CHARS]
                    recs.append(shuffle_options(r, random.Random(zlib.crc32(r["id"].encode()))))
    return [r for r in recs if r["qtype"] in TYPE]


def question(r: dict, name: str) -> dict:
    instr = r["instructions"].strip()
    task = human(r["task"])
    instructions = f"{task}: {instr}" if instr else task
    q = {"type": TYPE[r["qtype"]], "name": name, "instructions": instructions}
    if r["qtype"] == "choice":
        q["choices"] = [{"value": o["key"], "description": o["desc"] or o["key"]} for o in r["options"]]
    elif r["qtype"] == "score":
        q["levels"] = [{"label": o["key"], "description": o["desc"] or o["key"]} for o in r["options"]]
    return q


def build_requests(recs: list[dict]) -> list[dict]:
    by_row = collections.defaultdict(list)
    for r in recs:
        by_row[row_key(r["id"])].append(r)
    reqs = []
    for rk, rs in by_row.items():
        qs = [question(r, f"q{i}") for i, r in enumerate(rs)]
        body = {"model": MODEL, "input": rs[0]["text"], "questions": qs}
        est = ntok(rs[0]["text"]) + ntok(json.dumps(qs))
        reqs.append({"row": rk, "ids": [r["id"] for r in rs], "body": body, "est_tokens": est})
    return reqs


_KEYLIKE = re.compile(r"sk-[A-Za-z0-9_*\-]+")


def redact(obj):
    """Drop anything key-shaped (auth errors echo a masked key)."""
    return json.loads(_KEYLIKE.sub("sk-[redacted]", json.dumps(obj)))


def call(session: requests.Session, body: dict) -> tuple[int, dict, dict]:
    err: dict = {}
    for attempt in range(6):
        try:
            resp = session.post(URL, json=body, timeout=120)
        except requests.RequestException as e:
            err = {"error": repr(e)}
            time.sleep(2**attempt)
            continue
        hdr = {k: v for k, v in resp.headers.items() if k.lower().startswith(("x-ratelimit", "x-request-id", "openai-processing"))}
        try:
            data = resp.json()
        except ValueError:
            data = {"raw": resp.text[:2000]}
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(min(60, 2**attempt + random.random()))
            err = data
            continue
        return resp.status_code, redact(data), hdr
    return -1, err, {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--sets", nargs="+", default=["lockbox"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--limit", type=int, help="max requests per set")
    ap.add_argument("--sample", type=int, help="random sample of requests per set (seeded)")
    ap.add_argument("--budget", type=float, default=1.0, help="USD cap on estimated spend for this run")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="print request counts and estimates only")
    ap.add_argument("--key-file", type=Path, help="read the key from this file (first line) instead of OPENAI_API_KEY")
    args = ap.parse_args()

    if args.key_file:
        key = args.key_file.read_text().strip().splitlines()[0].removeprefix("OPENAI_API_KEY=").strip()
    else:
        key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set")
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {key}"

    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    stop = threading.Event()  # set on an auth error: no point sending the rest
    spent_est = 0.0
    totals = collections.Counter()

    for set_name in args.sets:
        reqs = build_requests(load_set(set_name, args.fold))
        if args.sample and args.sample < len(reqs):
            reqs = random.Random(0).sample(reqs, args.sample)
        if args.limit:
            reqs = reqs[: args.limit]
        raw_path = out / f"raw_{set_name}.jsonl"
        done = set()
        if raw_path.exists():  # resume: skip rows already answered
            with raw_path.open() as fh:
                done = {json.loads(line)["row"] for line in fh if json.loads(line)["status"] == 200}
        todo = [q for q in reqs if q["row"] not in done]
        est_total = sum(q["est_tokens"] * RESERVE_FACTOR + RESERVE_OVERHEAD for q in todo) / 1e6 * PRICE_PER_M
        print(f"{set_name}: {len(reqs)} requests ({len(todo)} to send), est ${est_total:.3f}", flush=True)
        if args.dry_run:
            continue

        def work(q):
            nonlocal spent_est
            reserve = (q["est_tokens"] * RESERVE_FACTOR + RESERVE_OVERHEAD) / 1e6 * PRICE_PER_M
            with lock:
                if spent_est + reserve > args.budget:
                    return None
                spent_est += reserve
            if stop.is_set():
                return None
            status, data, hdr = call(session, q["body"])
            if status in (401, 403):
                stop.set()
            billed = (data.get("usage") or {}).get("input_tokens") if status == 200 else 0
            with lock:
                spent_est += (billed if billed is not None else q["est_tokens"] * RESERVE_FACTOR + RESERVE_OVERHEAD) / 1e6 * PRICE_PER_M - reserve
            return {"row": q["row"], "ids": q["ids"], "est_tokens": q["est_tokens"], "status": status, "headers": hdr, "response": data}

        t0 = time.time()
        with raw_path.open("a") as fh, cf.ThreadPoolExecutor(args.workers) as ex:
            for i, res in enumerate(ex.map(work, todo), 1):
                if res is None:
                    totals["skipped"] += 1
                    continue
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                totals[f"status_{res['status']}"] += 1
                if i % 100 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}  {time.time() - t0:.0f}s  spent ${spent_est:.3f}  {dict(totals)}", flush=True)
        if stop.is_set():
            print("stopped: authentication failed (see raw file for the error)", flush=True)
            break
    print(f"done. spent ${spent_est:.3f}  {dict(totals)}", flush=True)


if __name__ == "__main__":
    main()
