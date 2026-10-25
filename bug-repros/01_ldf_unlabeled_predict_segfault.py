"""Bug 1: predicting on unlabeled multi-line (ldf) examples segfaults.

Each case runs in a child process so a crash doesn't stop the demo.

    uv run --project decisions python bug-repros/01_ldf_unlabeled_predict_segfault.py
"""

import subprocess
import sys

TRAIN = [
    ["shared |t great love", "1:0 |l positive", "2:1 |l negative"],
    ["shared |t awful hate", "1:1 |l positive", "2:0 |l negative"],
]
LABELED = ["shared |t great", "1:0 |l positive", "2:0 |l negative"]
UNLABELED = ["shared |t great", "|l positive", "|l negative"]
UNLABELED_NO_SHARED = ["|l positive great", "|l negative great"]

CASES = [
    # (description, vw args, example to predict on)
    ("csoaa_ldf mc, labeled (dummy costs)", "--csoaa_ldf mc", LABELED),
    ("csoaa_ldf mc, unlabeled", "--csoaa_ldf mc", UNLABELED),
    ("csoaa_ldf mc, unlabeled, no shared line", "--csoaa_ldf mc", UNLABELED_NO_SHARED),
    ("csoaa_ldf mc --probabilities, labeled", "--csoaa_ldf mc --probabilities", LABELED),
    ("csoaa_ldf mc --probabilities, unlabeled", "--csoaa_ldf mc --probabilities", UNLABELED),
    ("csoaa_ldf mc --csoaa_rank, labeled", "--csoaa_ldf mc --csoaa_rank", LABELED),
    ("csoaa_ldf mc --csoaa_rank, unlabeled", "--csoaa_ldf mc --csoaa_rank", UNLABELED),
    ("csoaa_ldf m (multiline), unlabeled", "--csoaa_ldf m", UNLABELED),
    ("wap_ldf mc, unlabeled", "--wap_ldf mc", UNLABELED),
]

CHILD = """
import sys, json
from vowpalwabbit import Workspace
args, train, ex = json.loads(sys.argv[1])
ws = Workspace(args + " -q tl --quiet")
for _ in range(20):
    for t in train:
        ws.learn(t)
print(repr(ws.predict(ex)))
ws.finish()
"""


def main() -> int:
    import json

    import vowpalwabbit

    print(f"vowpalwabbit {vowpalwabbit.__version__}, Python {sys.version.split()[0]}, {sys.platform}\n")
    crashed = 0
    for desc, args, ex in CASES:
        p = subprocess.run(
            [sys.executable, "-c", CHILD, json.dumps([args, TRAIN, ex])],
            capture_output=True,
            text=True,
        )
        if p.returncode == 0:
            result = f"ok      -> {p.stdout.strip()}"
        elif p.returncode < 0:
            crashed += 1
            result = f"CRASH   signal {-p.returncode} ({'SIGSEGV' if p.returncode == -11 else 'other'})"
        else:
            last = (p.stderr.strip().splitlines() or ["?"])[-1]
            result = f"error   exit {p.returncode}: {last[:120]}"
        print(f"{desc:45s} {result}")
    print(f"\n{crashed} case(s) crashed")
    return 1 if crashed else 0


if __name__ == "__main__":
    sys.exit(main())
