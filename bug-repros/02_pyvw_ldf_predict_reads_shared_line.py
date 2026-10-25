"""Bug 2: pyvw predict() on an ldf example with a shared line returns the
shared line's prediction slot, so per-option scores are shifted by one.

With --csoaa_ldf mc --csoaa_rank, predict() should return one score per
option. It returns [shared_line_score, option_1, option_2, ...] truncated to
the number of options: the first value is always 0.0 and the last option is
dropped. Reading each option's own partial prediction gives the right answer.

    uv run --project decisions python bug-repros/02_pyvw_ldf_predict_reads_shared_line.py
"""

import random
import sys

from vowpalwabbit import Workspace, pyvw

LAB = ["positive", "negative", "neutral"]
WORDS = {0: "great love excellent".split(), 1: "awful hate broken".split(), 2: "okay fine average".split()}


def block(text, gold=None):
    # dummy 0 costs when predicting (unlabeled ldf predict segfaults: see bug 1)
    costs = [0] * 3 if gold is None else [int(i != gold) for i in range(3)]
    return [f"shared |t {text}"] + [f"{i + 1}:{c} |l {d}" for i, (c, d) in enumerate(zip(costs, LAB))]


def main() -> int:
    rng = random.Random(0)
    ws = Workspace("--csoaa_ldf mc --csoaa_rank -q tl --quiet")
    for _ in range(600):
        g = rng.randrange(3)
        ws.learn(block(" ".join(rng.sample(WORDS[g], 2)), g))

    bad = 0
    for text, gold in [("great love", 0), ("awful hate", 1), ("okay fine", 2)]:
        from_predict = ws.predict(block(text))

        exs = ws.parse(block(text))
        pyvw.pylibvw.vw.predict_multi(ws, exs)
        per_line = [round(e.get_partial_prediction(), 3) for e in exs]
        ws.finish_example(exs)
        per_option = per_line[1:]

        best = min(range(3), key=per_option.__getitem__)
        print(f"text={text!r} gold={LAB[gold]!r}")
        print(f"  predict()                        -> {[round(x, 3) for x in from_predict]}")
        print(f"  partial_prediction, every line   -> {per_line}   (line 0 = shared)")
        print(f"  partial_prediction, options only -> {per_option}   best = {LAB[best]!r}")
        if [round(x, 3) for x in from_predict] != per_option:
            bad += 1
    print(f"\npredict() disagreed with the per-option scores on {bad}/3 examples")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
