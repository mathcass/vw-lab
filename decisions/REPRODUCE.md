# Reproducing every result

Every number in `FINDINGS.md` and the report page comes from the commands
below, run from this folder (`decisions/`). Each step's outputs go under `data/` or
`runs/`; both are git-ignored, about 3 GB in total.

## 0. Setup

- Python 3.12 via [uv](https://docs.astral.sh/uv/). `uv sync` installs the
  pinned versions from `uv.lock`, including `vowpalwabbit==9.11.9`.
- The OpenAI step needs `OPENAI_API_KEY` in a `.env` file
  (`OPENAI_API_KEY=sk-...`; git-ignored) and costs about $0.50.
- Memory: one VW training or feedback job uses about 2.3 GB, and GLiNER on
  CPU about 3 GB. Run heavy jobs one at a time. On a shared desktop, cap
  them so the job is what gets killed, e.g. prefix with
  `systemd-run --user --scope -p MemoryMax=4G`.
- The downloaded data is untrusted: every command runs Python with `-I`.

```bash
uv sync
```

## 1. Data

```bash
uv run python -I -m vwjev.download   # both datasets at the exact revisions used
uv run python -I -m vwjev.data       # -> data/processed/{bev,fast}.jsonl
uv run python -I -m vwjev.splits     # -> data/processed/splits/fold{0..3}.json
```

| Dataset | Hugging Face repo @ revision | Role |
|---|---|---|
| BEV Decision Mix | `avbiswas/bev-decision` @ `11ed9b5b` | train / held-out-domain dev / seen-domain test |
| fast-decisions (dev split) | `fastino/fast-decisions` @ `35afb79a` | lockbox: never trained or tuned on |

`splits.py` makes the domain-grouped 4-fold split. Each fold holds out about a
quarter of BEV's 28 domains:

- **dev:** BEV test rows of the held-out domains, capped at 1,500 per domain.
- **seen_test:** BEV test rows of the training domains, capped at 300 per domain.
- **train:** BEV train rows of the training domains.

BEV texts that share an 8-word sequence with the lockbox are dropped.

## 2. VW zero-shot models (all four folds)

Two models per fold. "Full" sees the text's words; "generic" sees only
features that don't depend on vocabulary.

```bash
for k in 0 1 2 3; do
  sfx=$([ $k = 0 ] && echo "" || echo "_f$k")
  uv run python -I -m vwjev.run --fold $k --name v5_full_shuf$sfx    --vw "-q tl -q ql -q sl -q tp -q qp -q qm"
  uv run python -I -m vwjev.run --fold $k --name v6_generic_shuf$sfx --vw "-q sl -q qm -q qp --ignore t"
done
```

Each run takes 1.5–2.5 minutes. It writes `runs/<name>/{model.vw,
feature_config.json, results.json, preds_{dev,seen_test,lockbox}.jsonl}` and
prints macro-F1 and accuracy, with the random and always-first-option
baselines. The other run names in `FINDINGS.md` (v0–v4) were earlier steps
before the data fixes. They are not needed for the final numbers.

## 3. VW learning from feedback

**BEV held-out domains (fold 0).** Each domain's BEV train rows are fed to the
model one at a time, up to 3,200; the domain's dev rows are scored at
checkpoints. Four curves per domain: pre-trained or from scratch, times full
corrections or right/wrong-only (bandit) feedback.

```bash
for d in "Spam detection" "Tool and workflow decisions" "Sentiment, emotion, and moderation" \
         "Rule-conditioned decisions" "Calibrated probability" "Response preference and quality" \
         "Sensory quality rating"; do
  uv run python -I -m vwjev.feedback --model runs/v6_generic_shuf --domain "$d"
done
uv run python -I -m vwjev.feedback_report    # summary table
```

Outputs: `runs/v6_generic_shuf/feedback/<domain>.json`.

**Lockbox.** For each of the 17 domains, the 100 texts are split 50/50 into a
test half and a learning pool, repeated 10 times. The model learns from 0,
5, 10, 20, 30 or 50 pool texts, and the test half is scored at each step.
It needs the OpenAI run (step 4) for the reference line; run step 4 first,
or drop `--openai`.

```bash
uv run python -I -m vwjev.lockbox_curve --model runs/v6_generic_shuf --openai luna_f0
```

Output: `runs/v6_generic_shuf/lockbox_curve.json`.

Both scripts reload the pre-trained model with an extra interaction.
`feedback.load_pretrained` restates the saved interactions while doing so,
because VW silently drops them otherwise (bug-repros item 3).

## 4. OpenAI Decisions API (fold 0)

```bash
uv run python -I -m vwjev.openai_decide --sets lockbox --limit 3 --workers 1 --name smoke --key-file .env --budget 0.01   # smoke test
uv run python -I -m vwjev.openai_decide --sets lockbox dev seen_test --name luna_f0 --key-file .env --budget 1.2
uv run python -I -m vwjev.openai_report --name luna_f0
```

- **Questions:** the same as VW's (same 4,000-character text cut, same option
  order). One request per text, carrying every question about that text.
- **Question types:** yes/no becomes `predicate`, choice becomes `choice`,
  and score becomes `score`. Multi-label questions are skipped, since the
  API has no multi-select.
- **Spend cap:** the script stops itself at `--budget` dollars.
- **Logging:** every raw response is saved to `runs/openai/luna_f0/`, so
  scoring can be redone without spending again. The key is never printed
  or saved, and anything key-shaped is removed from saved responses.

Our run: 11,300 requests, 4.8M billed tokens, $0.48, about 3 minutes at 8
parallel requests.

## 4b. Cold start: OpenAI first, then VW (fold 0)

Simulates a product with no labeled data: texts go to OpenAI at first, VW
learns from OpenAI's answers, and routing rules decide when VW answers
instead. It needs OpenAI answers for a longer stream of texts per domain:
up to 3,000 BEV training texts from each of the 7 held-out domains, on top
of the dev answers from step 4.

```bash
uv run python -I -m vwjev.openai_decide --sets stream --name luna_f0 --key-file .env --budget 0.95   # 19,426 requests, $0.77
uv run python -I -m vwjev.cascade prepare      # -> data/processed/cascade_f0.jsonl (one line per text)
./scripts/cascade_all.sh                        # all routing rules x 7 domains x 3 orders, about 45 minutes
uv run python -I -m vwjev.cascade_report       # tables + cost model -> runs/cascade/report.json
```

- **Routing rules:** always OpenAI; a fixed warm-up of N texts and then VW
  (with or without random audits); and a confidence gate. The gate lets VW
  answer once, on past OpenAI calls above its confidence threshold, VW
  agreed with OpenAI often enough. The rules are defined at the top of
  `src/vwjev/cascade.py`.
- **VW starts from scratch.** In step 3, scratch and pre-trained models
  were level after about 100 examples, and scratch is much lighter. Each
  simulation run takes 1–3 seconds and under 400 MB.
- **Cost-model assumptions** (engineering and server costs) are constants
  at the top of `src/vwjev/cascade_report.py`. Change them there and rerun
  the report.

## 5. GLiNER2.5-Decide (fold 0, not yet run)

GLiNER runs in its own environment with CPU PyTorch. Use a machine with
spare memory: it needs about 3 GB, and about 1–2.5 s per text on a 4-core
CPU.

```bash
uv run python -I -m vwjev.export_eval --fold 0   # exact eval questions + seeded 400-text BEV samples
                                                 # -> data/processed/eval_f0_{lockbox,dev,seen_test,dev_s400,seen_test_s400}.jsonl

uv venv .venv-gliner --python 3.12
uv pip install --python .venv-gliner/bin/python --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python .venv-gliner/bin/python "gliner2[local]"
# if the 1.9 GB weight download stalls, set HF_HUB_DISABLE_XET=1

for s in lockbox dev_s400 seen_test_s400; do
  .venv-gliner/bin/python -I bench/gliner_decide.py data/processed/eval_f0_$s.jsonl runs/gliner/f0/preds_$s.jsonl
done
```

Each question gets the same information OpenAI got: task name, instructions,
and option descriptions. The script resumes if interrupted. To run it on
another machine, copy `data/processed/eval_f0_*.jsonl`, `bench/`, and this
file, then copy `runs/gliner/f0/` back.

## 6. Report numbers

```bash
uv run python -I -m vwjev.final_report      # -> runs/final_report.json
uv run python -I -m vwjev.cascade_report    # cold-start section (after step 4b)
uv run python -I -m vwjev.build_report_page # -> report/can-vw-replace-jev.html (the report page)
uv run python -I -m vwjev.cost_estimate     # OpenAI cost estimates for every dataset
```

The report page's numbers all come from `runs/final_report.json` and
`runs/cascade/report.json`; only the prose lives in `report/template.html`.
The page is a local file; open it in a browser.

`final_report` puts every model on the same questions. When GLiNER results
exist in `runs/gliner/f0/`, it adds them on the rows GLiNER answered. It also
checks that OpenAI's lockbox-curve score matches the one `lockbox_curve`
recorded.

## Bug reproductions

```bash
cd .. && uv run --project decisions python bug-repros/01_ldf_unlabeled_predict_segfault.py
uv run --project decisions python bug-repros/02_pyvw_ldf_predict_reads_shared_line.py
uv run --project decisions python bug-repros/03_reload_with_q_drops_saved_interactions.py
```

They live at the repo root, in [`../bug-repros/`](../bug-repros/README.md); [`../VW-NOTES.md`](../VW-NOTES.md) collects the VW problems all three projects ran into.

## Checks: numbers you should get

| Step | Number |
|---|---|
| Fold 0, `v5_full_shuf` | seen_test macro-F1 0.600, dev 0.293, lockbox 0.284 (lockbox 0.290 on the questions OpenAI can answer) |
| Fold 0, `v6_generic_shuf` | seen_test 0.448, dev 0.299, lockbox 0.299 |
| OpenAI, fold 0 | seen_test 0.677, dev 0.640, lockbox 0.617 |
| Spam, pre-trained, corrections | 0.51 cold, 0.91 after 100 |
| Lockbox curve, pre-trained, corrections | 0.301 cold, 0.365 after 50 texts; OpenAI 0.603 on the same questions |
| Cold-start cascade, Gate 90% + 2% audits | later on: 65% of texts to OpenAI, macro-F1 0.629 vs 0.641 always-OpenAI (mean of 7 domains); spam 11% to OpenAI, 0.958 |

VW results are deterministic for a given machine and VW build (seeded
shuffles, single-threaded learning). OpenAI results can drift if the model
behind `gpt-6-luna` changes.
