"""Build the shareable report page from runs/final_report.json.

Every number on the page comes from the data embedded here; nothing in the
template is typed in by hand except the prose.

    uv run python -I -m vwjev.final_report
    uv run python -I -m vwjev.cascade_report      # optional: cold-start section
    uv run python -I -m vwjev.build_report_page   # -> report/can-vw-replace-jev.html
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

from vwjev.cost_estimate import PRICE_PER_M

ROOT = Path("report")
RUNS = Path("runs")
OPENAI_RAW = RUNS / "openai" / "luna_f0"
# cost_estimate.py output for "BEV everything (train + test)"; see FINDINGS.md
BEV_ALL_COST_EST = 11.03


def main() -> None:
    rep = json.loads((RUNS / "final_report.json").read_text())

    ms, est, tok, n_req = [], 0, 0, 0
    for name in ("lockbox", "dev", "seen_test"):
        with (OPENAI_RAW / f"raw_{name}.jsonl").open() as fh:
            for d in map(json.loads, fh):
                if d["status"] != 200:
                    continue
                n_req += 1
                est += d["est_tokens"]
                tok += d["response"]["usage"]["input_tokens"]
                if "openai-processing-ms" in d["headers"]:
                    ms.append(int(d["headers"]["openai-processing-ms"]))

    tables = rep["tables"]
    data = {
        "tables": tables,
        "curves": rep["curves"],
        "lockbox_curve": rep["lockbox_curve"],
        "selective": rep["selective"],
        "four_fold": rep["four_fold"],
        "trainMedian": int(statistics.median(rep["train_questions_per_domain"].values())),
        "refusals": sum(tables[s]["openai_refusals"] for s in ("seen_test", "dev", "lockbox")),
        "nq": sum(tables[s]["n_questions"] for s in ("seen_test", "dev", "lockbox")),
        "facts": {
            "oa_cost": tok / 1e6 * PRICE_PER_M,
            "oa_requests": n_req,
            "oa_tokens": tok,
            "oa_ms_median": int(statistics.median(ms)),
            "oa_ms_p90": int(np.percentile(ms, 90)),
            "oa_ms_p99": int(np.percentile(ms, 99)),
            "vw_ms": rep["four_fold"]["VW full"]["ms_per_question"],
            "tok_ratio": tok / est,
            "bev_all_cost": BEV_ALL_COST_EST,
        },
    }
    cascade = RUNS / "cascade" / "report.json"
    if cascade.exists():  # cold-start cascade simulation (vwjev.cascade_report)
        data["cascade"] = json.loads(cascade.read_text())
    html = (ROOT / "template.html").read_text()
    assert "/*__DATA__*/null" in html
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    out = ROOT / "can-vw-replace-jev.html"
    out.write_text(html)
    print(f"-> {out} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
