# Vowpal Wabbit bug reproductions

Found while building the [`decisions/`](../decisions/FINDINGS.md) models. Each
item has a standalone demo; each exits non-zero while the bug is present.
[`../VW-NOTES.md`](../VW-NOTES.md) lists these with the other VW problems
this repo ran into.

Environment where these reproduce:

- `vowpalwabbit` 9.11.9 from PyPI (wheel; no `vw` CLI binary included)
- Python 3.12.13, Linux x86_64 (kernel 6.8)

Run from the repo root, with `decisions/`'s environment (VW 9.11.9):

```bash
uv run --project decisions python bug-repros/01_ldf_unlabeled_predict_segfault.py
uv run --project decisions python bug-repros/02_pyvw_ldf_predict_reads_shared_line.py
uv run --project decisions python bug-repros/03_reload_with_q_drops_saved_interactions.py
```

---

## 1. Predicting on unlabeled multi-line (ldf) examples segfaults

**Demo:** `01_ldf_unlabeled_predict_segfault.py`

**What happens:** after training a `--csoaa_ldf` (or `--wap_ldf`) model,
calling `predict()` on a multi-line example whose action lines carry no
cost labels kills the process with SIGSEGV. The same example with dummy
costs (`1:0 |l ...`) predicts fine.

**Expected:** unlabeled examples are the normal input at prediction time and
should return a prediction.

**Observed (9.11.9):**

| Case | Result |
|---|---|
| `--csoaa_ldf mc`, labeled (dummy costs) | ok |
| `--csoaa_ldf mc`, unlabeled | **SIGSEGV** |
| `--csoaa_ldf mc`, unlabeled, no shared line | **SIGSEGV** |
| `--csoaa_ldf mc --probabilities`, labeled | ok |
| `--csoaa_ldf mc --probabilities`, unlabeled | **SIGSEGV** |
| `--csoaa_ldf mc --csoaa_rank`, labeled | ok |
| `--csoaa_ldf mc --csoaa_rank`, unlabeled | **SIGSEGV** |
| `--csoaa_ldf m`, unlabeled | **SIGSEGV** |
| `--wap_ldf mc`, unlabeled | **SIGSEGV** |

The crash is inside the native `pylibvw.vw.predict_multi` call (Python's
faulthandler shows `pyvw.py` line 779 as the last Python frame). It also
happens after reloading a saved model with `-i`. Not yet checked: whether the
`vw` CLI crashes on the same input, which would separate the C++ core from
the Python binding.

**Minimal shape:**

```python
from vowpalwabbit import Workspace
ws = Workspace("--csoaa_ldf mc -q tl --quiet")
for _ in range(20):
    ws.learn(["shared |t great love", "1:0 |l positive", "2:1 |l negative"])
    ws.learn(["shared |t awful hate", "1:1 |l positive", "2:0 |l negative"])
ws.predict(["shared |t great", "|l positive", "|l negative"])  # SIGSEGV
```

**Workaround:** give every action line a dummy cost (`i:0`) when predicting.

---

## 2. `--csoaa_rank` scores are shifted by one when a shared line is present

**Demo:** `02_pyvw_ldf_predict_reads_shared_line.py`

**What happens:** with `--csoaa_ldf mc --csoaa_rank`, `predict()` should
return one score per action. With a `shared` line it instead returns the
scores of lines 0..K−1, counting the shared line as an action: the first
value is always `0.0` (the shared line) and the last action's score is
missing.

**Observed (9.11.9),** 3 actions, trained so the gold action scores ≈ −1:

| Text (gold) | `predict()` | Each action line's `get_partial_prediction()` |
|---|---|---|
| "great love" (positive) | `[0.0, -1.005, 1.008]` | `[-1.005, 1.008, 1.08]` |
| "awful hate" (negative) | `[0.0, 1.0, -1.0]` | `[1.0, -1.0, 1.0]` |
| "okay fine" (neutral) | `[0.0, 1.008, 1.0]` | `[1.008, 1.0, -1.001]` |

Taking the lowest score from `predict()` gives the wrong action in the third
row (it picks `positive`; the right answer is `neutral`).

**Where the bug likely is:** `pyvw.Workspace.predict` returns
`ec[0].get_prediction(...)`, and for multi-line examples VW stores the
prediction on the first example, which is the shared line. That part looks
intentional. The *content* is wrong: the stored scores match the per-line
partial predictions of lines 0..K−1, i.e. the shared example is counted as an
action. So this is most likely in VW core's csoaa_ldf rank output, not the
Python wrapper. Not confirmed against the `vw` CLI.

**Not affected (as far as checked):** `--probabilities` returned one value per
action that agreed with the per-action scores in our tests; `--cb_adf`
returned per-action costs (sorted by score, not by action).

**Workaround:** after `predict_multi`, read `get_partial_prediction()` from
each action example (skip the shared line). These come back in input order.

---

## 3. Adding `-q` when reloading a model silently drops its saved interactions

**Demo:** `03_reload_with_q_drops_saved_interactions.py`

**What happens:** a model trained with `-q ab` and reloaded with
`-i model.vw -q cd` (to add an interaction for continued training) no longer
applies `-q ab`. Its predictions change even on examples with no `c` or `d`
features, with no warning. Restating the saved interaction
(`-i model.vw -q ab -q cd`) restores the original predictions exactly.

**Observed (9.11.9),** trained so the gold option scores ≈ −1:

| Reload args | Scores for 2 examples |
|---|---|
| `-i m.vw` | `[-1.0, 1.0]`, `[1.0, -1.0]` |
| `-i m.vw -q cd` | `[-0.001, 0.044]`, `[-0.040, 0.005]` |
| `-i m.vw -q ab -q cd` | `[-1.0, 1.0]`, `[1.0, -1.0]` |

**Expected:** either the saved and new interactions are merged, or VW warns
that command-line `-q` replaces the saved ones. This may be intended
("command line overrides saved options"), but the silent loss of the
model's main interactions is easy to miss. It cost us a round of wrong
experiment results.

**Workaround:** restate every saved `-q` / `--interactions` when adding new
ones at load time.

---

## Related, not a bug (misleading message)

Loading a `--csoaa_ldf` model with `--cb_adf` / `--cb_explore_adf` fails with
`Model content is corrupted, weight vector index ... must be less than total
vector length`. The model isn't corrupted; the two reductions store different
state. A clearer error would help.
