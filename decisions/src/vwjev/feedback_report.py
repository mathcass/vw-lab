"""Summarize feedback curves: macro-F1 at a few checkpoints per domain.

    uv run python -I -m vwjev.feedback_report runs/v6_generic_shuf [runs/v5_full_shuf ...]
"""

import json
import sys
from pathlib import Path

SHOW = [0, 25, 100, 400, 1600]


def main() -> None:
    for run in sys.argv[1:]:
        print(f"\n== {run}  (macro-F1 on held-out eval rows after N feedback events; last col = online acc over the whole stream)")
        print(f"{'domain':36s} {'start/mode':22s} " + " ".join(f"{('n=' + str(n)):>7s}" for n in SHOW) + "    final  online")
        for f in sorted((Path(run) / "feedback").glob("*.json")):
            res = json.loads(f.read_text())
            for key, curve in res["curves"].items():
                at = {p["n"]: p["macro_f1"] for p in curve}
                last = curve[-1]
                cells = " ".join(f"{at[n]:7.3f}" if n in at else f"{'':>7s}" for n in SHOW)
                print(f"{res['domain'][:36]:36s} {key:22s} {cells}   {last['macro_f1']:.3f}@{last['n']:<5d} {last['online_acc']:.3f}")


if __name__ == "__main__":
    main()
