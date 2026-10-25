"""Item 3: adding -q when reloading a model silently drops the saved -q's.

Train with `-q ab`, save, then reload with `-i model.vw -q cd` to add a new
interaction for continued training. The reloaded model no longer applies
`-q ab`: its predictions change even on examples that have no `c` or `d`
features at all. Restating the saved interaction (`-q ab -q cd`) restores
them exactly.

    uv run --project decisions python bug-repros/03_reload_with_q_drops_saved_interactions.py
"""

import random
import sys
import tempfile
from pathlib import Path

from vowpalwabbit import Workspace


def block(i, gold=None):
    costs = [0, 0] if gold is None else [int(j != gold) for j in range(2)]
    return [f"shared |a w{i % 7}"] + [f"{j + 1}:{c} |b opt{j}" for j, c in enumerate(costs)]


def scores(ws, lines):
    exs = ws.parse(lines)
    from vowpalwabbit import pyvw

    pyvw.pylibvw.vw.predict_multi(ws, exs)
    out = [round(e.get_partial_prediction(), 4) for e in exs[1:]]
    ws.finish_example(exs)
    return out


def main() -> int:
    rng = random.Random(0)
    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp) / "m.vw"
        ws = Workspace(f"--csoaa_ldf mc -q ab --quiet -f {model}")
        for _ in range(500):
            i = rng.randrange(7)
            ws.learn(block(i, gold=i % 2))
        ws.finish()

        variants = {
            "-i m.vw": f"-i {model}",
            "-i m.vw -q cd": f"-i {model} -q cd",
            "-i m.vw -q ab -q cd": f"-i {model} -q ab -q cd",
        }
        got = {}
        for name, args in variants.items():
            w = Workspace(args + " --quiet")
            got[name] = [scores(w, block(i)) for i in range(4)]
            w.finish()
            print(f"{name:22s} {got[name]}")

    changed = got["-i m.vw -q cd"] != got["-i m.vw"]
    restored = got["-i m.vw -q ab -q cd"] == got["-i m.vw"]
    print(f"\nadding -q cd changed predictions (no c/d features present): {changed}")
    print(f"restating -q ab restores them: {restored}")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
