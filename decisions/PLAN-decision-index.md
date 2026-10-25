# Plan: score VW on the Jev Decision Index (public suite)

Written 2026-10-22. This plan is meant to run on a separate, larger machine.
Step 1 needs no new data and can run anywhere.

## Goal

Estimate where VW would sit on the [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index),
using only the **public** part of the benchmark.

- **The Index:** a community leaderboard of open Jev rebuilds, with Jev
  1.14 itself on it. Not run by TypeSafe.
- **How it ranks:** by private tests only. 60% are private versions of
  the same skills as the public benchmarks; 40% are private tasks from
  new domains. Only the maintainers can run those.
- **What anyone can run:** the public index. 37 public benchmarks
  (~110k requests, plus ~30k for 7 added benchmarks), five areas, every
  benchmark chance-corrected (0 = random, 100 = perfect). It is shown on
  the board but does not count toward the ranking.

**Read the result as optimistic.** On the board, small models score far
higher on the public tests than on the private new-domain tests:

| Model | Public | Private, new domains |
|---|---|---|
| GLiClass v4 edge (0.4B) | 47.5 | 10.5 |
| Crivo-1B | 47.8 | 10.8 |

VW's public score will most likely overstate its ranking score in the
same way.

**Questions to answer:**

1. **Zero-shot:** where does VW (trained on BEV, never on the suite) land
   on the public index and on each of the five areas? Compare it with Jev
   and with other small, fast models.
2. **Learning from data:** if VW learns from some of a benchmark's own
   rows, how quickly does it close the gap, per area? This is outside the
   board's rules, so it is reported separately and labeled as such.
3. **Optional, needs approval:** OpenAI's Decisions API on the same
   scale.

## Reference points (board snapshot 2026-10-22, edition 0.4.1)

The public suite is unchanged since 0.3, so these public scores are
comparable with a local 0.3 run. Area columns are public area scores
(0–100, chance-corrected).

| Model | Public index | Knowledge | Language | Retrieval & Classif. | Tools | Arts | Median latency |
|---|---|---|---|---|---|---|---|
| Fastino GLiDE 1.2 (#1, 27B) | 72.0 | 60.6 | 75.5 | 72.2 | 88.0 | 62.6 | 93 ms |
| **Jev 1.14** (#41) | **60.7** | 57.1 | 61.2 | 63.5 | 73.8 | 39.2 | (HTTPS API) |
| GLiClass v4 edge (0.4B encoder) | 47.5 | 21.4 | 54.6 | 66.7 | 58.1 | 38.6 | 22 ms |
| Crivo-17M (encoder) | 22.2 | 1.2 | 21.0 | 40.1 | 36.5 | 17.0 | 6 ms |
| GLiNER2.5-Decide | 11.5 | 2.3 | 8.3 | 25.3 | 14.8 | 9.8 | 23 ms |
| Bekko System One v0 17M | 7.4 | 2.5 | 2.7 | 21.3 | 5.9 | 7.5 | 7 ms |

Area weights: Knowledge & Reasoning 25.8%, Language Understanding 25.8%,
Retrieval & Classification 20.0%, Tools & Automation 18.3%, Arts & Human
Taste 10%.

**For our question (routing-style decisions), Retrieval & Classification
and Tools matter most.** Knowledge & Reasoning is mostly exam questions,
math and chess, which no small model can do.

GLiNER2.5-Decide is already on the board, so the write-up (step 8) can use the
board's number instead of running GLiNER ourselves.

## Rules to follow (from the Index, plus ours)

**Index rules:**

- **No truncation.** Send every request whole. Our VW pipeline currently
  cuts text to 4,000 characters (`run.MAX_CHARS`) and reads only the
  first 256 tokens (`FeatureConfig.max_text_tokens`). The scored runs
  must use neither (see step 4).
- **No option filtering, no per-benchmark logic, no prompt tuning.** Map
  every request to VW with one fixed rule.
- **No gold labels inside the engine, and no retries of wrong answers.**
- **Unanswered = wrong.** VW has no length or option limit, so it should
  answer everything.

**Our rules:**

- **Never train on the suite.** It is evaluation data under its sources'
  terms. The only exception is the off-board learning curve (step 6),
  which throws every adapted model away.
- **Don't redistribute the suite.** Several sources forbid it. Keep it
  out of this repo, out of any artifact and out of git. Bring back
  scores, not rows.
- **No external actions.** Don't submit to the leaderboard, open PRs or
  issues, or publish anything unless asked. Reports stay local files.
- **No OpenAI spending without approval** (step 7). Never print or log
  the API key; pass it with `--key-file .env`.
- **Untrusted downloads.** Treat downloaded data as untrusted: keep it in
  its own directory and run Python that reads it with `python -I`. The
  kit's own code is run on purpose, in its own virtual environment.

## Machine and inputs

- **Machine:** Linux, at least **35 GB free disk** (about 7 GB of
  downloads, about 17 GB of build space, plus runs), 16 GB+ RAM and 4+
  cores. No GPU is needed: VW runs on CPU.
- **Accounts:** a Hugging Face login, with the
  [HLE](https://huggingface.co/datasets/cais/hle) terms accepted (that
  dataset is gated).
- **The kit:** `https://github.com/apolinario/decision-index`, pinned to
  commit `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d` (2026-10-20, edition
  0.4 merge). Install it in its own virtual environment.
- **Copied from this machine** (not in git: `data/` and `runs/` are
  ignored):

| Path | Size | Why |
|---|---|---|
| repo: `src/`, `pyproject.toml`, `uv.lock`, `FINDINGS.md`, `REPRODUCE.md` | small | code |
| `runs/v6_generic_shuf/{model.vw,feature_config.json,results.json}` | 12 MB | VW generic (the main entry) |
| `runs/v5_full_shuf/{model.vw,feature_config.json,results.json}` | 230 MB | VW full (word-pair model) |
| `data/processed/bev.jsonl` | 961 MB | overlap check, temperature fit, optional retrain |
| `.env` | — | only if step 7 is approved |

Both models are the fold-0 models: trained on 21 of the 28 BEV domains.

## Steps

### 1. Rough mapping from existing numbers (no new data; can be done here)

Put the numbers we already have on the Index's scale before running
anything:

- **Chance-correct them:** `skill = clip((acc − chance) / (1 − chance), 0, 1)`,
  with chance = the mean of 1/options. Apply it per domain, then average.
- **Which numbers:** VW full, VW generic and OpenAI, on BEV seen domains,
  BEV dev and the lockbox.
- **Loose mapping to the Index's two private parts:**
  - BEV seen domains ≈ "same skills".
  - BEV dev and the lockbox ≈ "new domains".
- **Plug into the Index's formula,** to show roughly where each model
  would land:
  `Full = 0.6 × (23.5 + 16.54 × (S − 24.27) / 16.35) + 0.4 × (23.5 + 16.54 × (O − 16.88) / 13.39)`.
  Here S and O are the two parts on the 0–100 scale. The constants are
  the stock-model means and SDs from the board's `data/v03.json`.
- **Caveat:** different benchmarks, so this is an analogy, not a
  measurement. Label it that way.
- **Output:** a short table in `FINDINGS.md`.

### 2. Build the public suite (edition 0.3)

```sh
git clone https://github.com/apolinario/decision-index.git && cd decision-index
git checkout 25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d
python -m venv .venv && . .venv/bin/activate
pip install -e ".[transformers,rebuild]"
export HF_HUB_DISABLE_XET=1
hf auth login

python -m decision_index suite rebuild --work work
python -m decision_index suite import \
    --rows work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz \
    --added-rows work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz \
    --gsm8k-rows work/artifacts/benchmark-suite/release-v3-rebuilt/gsm8k-rows.jsonl.gz
python -m decision_index suite verify
```

**Check:** the import must accept the files. It refuses any file whose
hash doesn't match. Expected uncompressed sha256 values:

- `b2b56d6f…`: base rows.
- `7429f3c9…`: the 7 added benchmarks.
- `75933489…`: the rebuilt GSM8K rows.

**Fallback:** if disk is short, `suite rebuild --only <ids>` builds
benchmarks one at a time. The biggest downloads are ESCI (2 GB), HoVer's
Wikipedia database (2.2 GB), BRIGHT (0.6 GB) and Habermas (0.4 GB). A
partial suite fails the hash check, so any index from it must be labeled
partial.

### 3. Check BEV for overlap with the suite

BEV was rebuilt from public datasets. Some of them may also be in the
suite: BANKING77, CLINC150, NLI sets, sentiment and stance sets,
HellaSwag and so on.

- **The check:** run the same 8-word-sequence check we used for the
  lockbox (`src/vwjev/splits.py`, `shingles`, `SHINGLE = 8`) between
  every BEV text and every suite `state`.
- **Report:** matching BEV texts per suite benchmark and per BEV domain.
- **If any overlap is found:**
  - The fold-0 models saw those texts in training. Report the affected
    benchmarks separately.
  - For the clean number, retrain VW generic with the overlapping BEV
    groups removed (same command as `runs/v6_generic_shuf`, see
    REPRODUCE.md) and use that model for the headline.
- **Output:** `runs/jdi/overlap.json` and a line in FINDINGS.

### 4. Write the VW engine for the kit

New file `src/vwjev/jdi_engine.py`, class `VWEngine(decision_index.engines.Engine)`.
It is used as `--engine vwjev.jdi_engine:VWEngine --option model_dir=runs/v6_generic_shuf`.

**Turning a request into our record format,** one fixed rule for every
benchmark:

- **Text:** `text` = the request's `state`. If it is a JSON object, use
  the kit's `engines.base.text()`, which serializes it compactly.
- **One record per question key `k`:**
  - `task = k`.
  - `instructions`: the question's instructions, serialized the same way
    if they are an object.
  - `options`: `[{"key": key, "desc": desc} for key, desc in criteria.items()]`,
    in the given order.
- **Question type:**
  - `"noul"` → `yesno`, with options `yes` / `no`, answered as
    `{"type": "noul", "noul": p_yes}`.
  - A `"choice"` whose keys are exactly {yes, no} → `yesno` too,
    answered as a choice.
  - Anything else → `choice`.

**Scoring:**

- **Per-option scores:** from `run.option_scores(ws, features.to_lines(rec, cfg))`.
  Lower is better; the answer is the argmin.
- **Probabilities:** `p_i = softmax(−s_i / T)`.
  - Fit T once on BEV fold-0 dev, minimizing log loss, and keep it fixed
    for every benchmark.
  - It matters because ToolRet and BRIGHT rank candidates by `p(yes)`
    across separate questions (nDCG@10).
  - Record T in `provenance`.
- **The kit's checks:** every option must get a probability, and they
  must sum to 1 within 0.01. The kit checks this on every answer.

**No truncation:**

- Don't apply `MAX_CHARS`, and set `max_text_tokens` to no limit. This
  needs a small change in `features.to_lines` so it accepts `None`.
- The models were trained with a 256-token cap, so on BEV dev compare
  capped vs uncapped prediction and report both numbers. Score the
  Index runs uncapped.

**Engine details:**

- **Never raise `Unsupported`.** VW has no length or option limit. Check
  memory and time on the longest ToolRet and BRIGHT rows.
- **Load the model once:** `Workspace(f"-i {model} --quiet")`. Restate
  the saved `-q` interactions if any are added (see
  `feedback.load_pretrained` and bug-repros item 3).
- **`provenance` records:** the model file's sha256, the VW version
  (9.11.9), this repo's state (no commit yet; record a hash of `src/`),
  T, and `max_text_tokens`.
- **`latency`:** describe what the timing covers (CPU, in-process).

**Smoke test:**

```sh
python -m decision_index suite sample --n 100 --out sample-100.jsonl.gz
python -m decision_index run --engine random --rows sample-100.jsonl.gz --out runs/jdi/random-sample
python -m decision_index run --engine vwjev.jdi_engine:VWEngine --option model_dir=<repo>/runs/v6_generic_shuf \
    --rows sample-100.jsonl.gz --out runs/jdi/vw-generic-sample
python -m decision_index score --results runs/jdi/vw-generic-sample/results.jsonl
```

**Checks:** no `error` rows; every answer passes the kit's checks; the
random engine's index is about 0.

### 5. Runs

**Main runs:** the full public 0.3 suite for each engine, one at a time.
Each scores itself at the end, and nothing is uploaded (no `--upload`):

```sh
python -m decision_index pipeline --engine random --out runs/jdi/random
python -m decision_index pipeline --engine vwjev.jdi_engine:VWEngine --option model_dir=<repo>/runs/v6_generic_shuf --out runs/jdi/vw-generic --compact
python -m decision_index pipeline --engine vwjev.jdi_engine:VWEngine --option model_dir=<repo>/runs/v5_full_shuf  --out runs/jdi/vw-full --compact
```

| Run | Model | Purpose |
|---|---|---|
| A. `vw-generic` | VW generic (fold 0, or the overlap-clean retrain from step 3) | Main entry: the fair zero-shot comparison |
| B. `vw-full` | VW full, with word-pair features | Expected to do worse out of domain; confirms the pattern |
| C. `random` | the kit's random engine | Sanity check; should score about 0 |
| D. Optional: `vw-generic-all` | VW generic retrained on all 28 BEV domains, minus overlap | The best zero-shot VW we could ship; needs a small change to `run.py` to train on every domain |

**From each run, keep:**

- `scores.json`, `index.json`, `benchmark-summary.json` and
  `environment.json`.
- Median and p95 time per request (CPU).
- With `--compact`, the results file has no request text; still don't
  copy it off the machine.

**Time estimate:** about 540k questions in the base suite, plus the added
benchmarks. At 1–3 ms per question in Python, that is roughly 15–45
minutes per run, plus extra time for the long retrieval texts.

### 6. Learning from data (off the board, labeled as such)

This shows how quickly VW learns each public benchmark from that
benchmark's own rows. It uses the same method as the lockbox learning
curve (`src/vwjev/lockbox_curve.py`).

**Method:**

- **Split:** per benchmark, shuffle the linked case groups (`_evaluation.group_id`)
  with a seed. Half is the learning pool, half the test set. Keep linked
  groups whole.
- **Learn:** starting from VW generic, learn from k ∈ {0, 25, 100, 400, 1600}
  pool groups, one at a time, with correction updates (as in
  `lockbox_curve.learn`). Cap k at the pool size.
- **Score** the test half with the kit's own scorer
  (`scoring.report.score` on those rows), then chance-correct it with
  that benchmark's chance level (`scoring.metrics.skill`, levels in
  `data/index-0.3.json`).
- **Repeat** with 3 seeds, and throw away every adapted model.

**Output:**

- Per-benchmark and per-area skill at each k, next to Jev's public area
  scores.
- An "index at k" curve, computed with the area weights.

**Expected:** the classification benchmarks (BANKING77, CLINC150,
PhishNChips, ESCI) learn fastest; knowledge and reasoning barely move.

**Label:** "not a board score: the Index forbids retraining."

### 7. Optional: OpenAI Decisions API on the same scale (ask before spending)

**The engine:** wrap `openai_decide.py`'s request building in an `Engine`.

- `choice` → choice; `noul` → predicate.
- **Probabilities:** the API returns a choice and a confidence, not a
  probability per option, but the kit needs one for every option.
  - Give the chosen option its confidence and spread the rest evenly
    across the other options.
  - Record this in `provenance` as a mechanical probability conversion.
    The Index allows that if it is recorded.
- **Refusals** count as wrong (status `abstained`).

**Cost:**

- First count tokens on the actual suite rows (`cost_estimate.py`,
  o200k, × 1.27 for the billed-to-estimate ratio we measured).
- Ballpark: Jev's own run used 292M input tokens, so about $30–40 at
  $0.10 per 1M.
- A stratified sample (`suite sample`) costs about $1–2. It gives an
  estimate, not an official score.
- Present both to the user and spend nothing until they choose.

### 8. Write-up (local files only)

- **`FINDINGS.md`,** new section "Jev Decision Index (public suite 0.3)":
  - The reference table above with VW rows added: public index, five
    areas, latency.
  - The overlap result.
  - The step-1 mapping.
  - The learning-from-data table.
  - A takeaway, plus the "public overstates small models" caveat.
- **`REPRODUCE.md`:** a new step with the exact commands, the kit
  commit, T, and the suite hashes.
- **Local report page:** one new section. Rebuild it with
  `vwjev.build_report_page`, and don't publish it.

## What to bring back

- `runs/jdi/<run>/{scores.json,index.json,benchmark-summary.json,environment.json}`
  for each run.
- `runs/jdi/overlap.json` and `runs/jdi/learning_curve.json`.
- `src/vwjev/jdi_engine.py` and any code changes, ideally as a patch.
- The overlap-clean model, if one was trained (about 12 MB).
- No suite rows, payloads or raw results files.

## Open questions and risks

- **Overlap with BEV** could inflate VW's numbers on some benchmarks.
  Step 3 settles it before anything is reported.
- **The truncation rule against our 256-token training cap:** uncapped
  prediction is the compliant one. Step 4 measures what it costs on BEV.
- **Retrieval needs probabilities that compare across questions.** A
  single temperature may rank poorly; if so, report it rather than
  tuning per benchmark.
- **The board changes often** (0.1 to 0.4.1 in a month). Pin the kit
  commit and quote reference numbers with their snapshot date.
- **The ranking score is out of reach** without submitting to the
  maintainers. Submitting needs the model on the Hub, inference code, a
  PR and their latency check (VW passes easily: under 1 ms per question).
  Only do it if the user asks.
