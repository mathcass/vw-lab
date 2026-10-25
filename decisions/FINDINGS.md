# Can Vowpal Wabbit replace Jev? — running findings

The question: can a VW model answer Jev-style decisions (yes/no, pick one,
score on a scale) well enough for low-stakes product use (e.g. routing a
query to an LLM tier)? Two bars:

1. **Zero-shot:** a pre-trained model reads a new schema and is usable as-is.
2. **Cold start + feedback:** usable-ish on day one, then improves quickly
   from product feedback (corrections, or just "that was right/wrong").

## Data

| Name | What it is | Use |
|---|---|---|
| BEV Decision Mix (`avbiswas/bev-decision`) | Third-party set of ~276k texts / 502k questions rebuilt from public datasets into Jev's CHOICE / NOUL / SCORE format. 28 domains. License listed as "unknown". | Train, dev, seen-domain test |
| fast-decisions (`fastino/fast-decisions`) | Fastino's published dev split of their GLiNER2.5-Decide benchmark: 17 domains × 100 texts, 2,900 questions. Synthetic, long, deliberately tricky. | **Lockbox** (see below): never trained or tuned on |

### What the lockbox is

"Lockbox" is our name for a test set that is locked away during
development. We never train on it, and we never use it to choose settings.
We only open it to measure a finished model. This is the fairest test of
the zero-shot bar, because nothing about it leaked into how VW was built.
BEV's dev split can't fully play that role: we looked at dev scores while
choosing features, so dev is slightly tuned-on.

Our lockbox is Fastino's fast-decisions set:

- **Who made it:** Fastino, the GLiNER2.5-Decide team, as a benchmark of their own.
  It is unrelated to BEV, so its writing style, schemas and label names are
  new to VW.
- **Size:**
  - 17 domains, 100 texts each, 1,700 texts in all.
  - 2,900 questions: 1,900 choice, 700 yes/no and 300 multi-label.
- **What it looks like:**
  - The domains are product-style tasks: email triage, ticket routing,
    banking intent, clinic requests, restaurant reviews, sports recaps and
    so on.
  - The texts are synthetic and fairly long, with a median of 750 characters.
  - Many options are bare labels with no description. For example, ticket
    routing picks a queue from `billing_disputes`, `refunds`, `shipping`,
    `returns`, `identity` and `payments`.
- **The 2,600 figure:** it counts the 2,900 questions minus the 300
  multi-label ones, which the OpenAI API can't answer.
- **No overlap with BEV:** 76 BEV texts that share an 8-word sequence with
  a lockbox text were removed.

The **lockbox learning curve** (below) is the one place where VW learns from
lockbox texts. Even there, each run learns from half of a domain's texts and
is scored on the other half. Every adapted model is then thrown away, so
the lockbox stays unused for building or tuning the main model.

Splits (`src/vwjev/splits.py`): the 28 BEV domains are grouped into 4 folds.
For fold k, **dev** = BEV test rows from the fold's domains (never trained on:
the zero-shot number), **train** = BEV train rows from the other domains,
**seen_test** = BEV test rows from training domains. No text crosses splits;
76 BEV texts sharing an 8-word sequence with the lockbox were dropped.

## Metric

Macro-F1, pooled per domain over `(question name, answer)` classes, then
averaged across domains (`src/vwjev/metrics.py`). Accuracy is reported
alongside. "random" = uniform random choice among the offered options.

## Model

VW `--csoaa_ldf mc`: one line per option, the model scores each option
against the shared text/question features, lowest score wins.

## Results (fold 0)

> **The v0–v4 table below predates two fixes** (option-order shuffling and the
> min-support macro-F1). It is kept for the record; see "Results after fixes".

Dev domains held out in fold 0: Calibrated probability, Response preference
and quality, Rule-conditioned decisions, Sensory quality rating, Sentiment /
emotion / moderation, Spam detection, Tool and workflow decisions.

| Run | Features | Dev (zero-shot) | Seen domains | Lockbox | Model |
|---|---|---|---|---|---|
| random | | 0.293 | 0.260 | 0.179 | |
| v0 | text × option words (`-q tl -q ql`) | 0.288 | 0.464 | 0.227 | 233 MB |
| v1 | v0 + match / position interactions | 0.285 | **0.486** | 0.189 | 235 MB |
| v2 | generic features only (no text words) | 0.289 | 0.401 | 0.233 | 5.5 MB |
| v3 | v2 + IDF-weighted and relative overlap | **0.307** | 0.402 | **0.264** | 6.3 MB |
| v4 | v3 + low-rank text × option (`--lrq tl16`) | 0.306 | 0.373 | 0.209 | — (6× slower) |

All numbers macro-F1. 318k training questions, one pass, ~2–3 min to train.

### What this says so far

- **Word-pair features don't transfer.** Text-word × option-word weights help
  on domains seen in training but add nothing zero-shot, and hurt on the
  lockbox. This matches the toy test: learned crosses memorize.
- **All zero-shot signal comes from generic features** (overlap between option
  words and the text, relative to the other options; position on a scale).
  These get the lockbox from 0.18 (random) to 0.26. That is not usable.

## Results after fixes (fold 0)

Options shuffled (except score scales); macro-F1 ignores classes with < 5
gold examples. Random = uniform random among offered options.

| Run | Features | Dev (zero-shot) | Seen domains | Lockbox | Model |
|---|---|---|---|---|---|
| random | | 0.320 | 0.354 | 0.184 | |
| v5 | text × option words + generic | 0.293 | **0.600** | 0.284 | 235 MB |
| v6 | generic only | 0.299 | 0.448 | **0.299** | 6.4 MB |

### All four folds (macro-F1, mean over folds)

| Model | Dev (zero-shot) | Seen domains | Lockbox |
|---|---|---|---|
| random | 0.326 | 0.360 | 0.184 |
| v5: text × option words + generic | 0.327 | **0.605** | 0.290 |
| v6: generic only | 0.283 | 0.450 | **0.297** |

Per fold, dev: v5 0.293 / 0.324 / 0.325 / 0.367, random 0.320 / 0.359 / 0.327 / 0.296.

Best zero-shot domains (v5): Support and intent routing 0.58 (50% acc on
~5 options), Tool and workflow 0.56 (60% acc), Engagement and ranking 0.52,
Reading comprehension 0.51. Worst: Sensory rating 0.11, Response preference
0.17, Answer verification 0.20.

### Verdict on headline 1 (zero-shot): no-go

Averaged over held-out domains, zero-shot VW is no better than random
macro-F1, and the lockbox (0.29–0.30 vs 0.18 random) is far from usable.
Only routing-style domains with label words that appear in the text get
well above random, and even those (0.56–0.58) are short of a product bar.
Every VW-native lever tried (interactions, IDF/relative overlap, low-rank
`--lrq`) left zero-shot within a few points of these numbers. Moving on to
headline 2: cold start + feedback.

## Cold start + feedback (fold 0 held-out domains)

`src/vwjev/feedback.py`: stream a held-out domain's BEV train rows, answer
each with the current model, update from feedback; score on that domain's
dev rows. Macro-F1, pre-trained generic model (v6), correction feedback,
with OpenAI's zero-shot score on the same dev rows:

| Domain | Cold | 25 | 100 | 400 | 3,200 | OpenAI, no training |
|---|---|---|---|---|---|---|
| Spam detection | 0.51 | 0.84 | **0.91** | **0.95** | **0.95** (stream ends at 1,784) | **0.91** |
| Tool and workflow decisions | 0.55 | 0.54 | 0.63 | 0.63 | 0.80 | **0.96** |
| Sentiment, emotion, moderation | 0.32 | 0.34 | 0.37 | 0.40 | 0.50 | **0.63** |
| Rule-conditioned decisions | 0.39 | 0.37 | 0.40 | 0.38 | 0.51 | **0.91** |
| Calibrated probability | 0.22 | 0.28 | 0.27 | 0.30 | 0.40 | **0.50** |
| Sensory quality rating | 0.14 | 0.14 | 0.13 | 0.21 | **0.32** (ends at 1,040) | 0.27 |
| Response preference and quality | 0.11 | 0.14 | 0.19 | 0.17 | 0.18 | **0.31** |

- Keyword-driven tasks with dataset-specific labeling become usable fast
  and can match or beat OpenAI (spam: level after 100 corrections, ahead
  after 400; sensory ratings: ahead after about 1,000).
- Judgment and reasoning tasks barely improve with feedback, and stay well
  below OpenAI zero-shot even after 3,200 corrections.
- Pre-training gives a slightly better start; a from-scratch model catches
  up within 25–100 events.
- Right/wrong-only (bandit) feedback matches corrections on yes/no tasks but
  lags on multi-option ones (tool routing 0.73 vs 0.80 at 3,200).

**Correction (2026-10-19):** the first version of these curves reloaded
the pre-trained model with `-i model.vw -q fl`, which silently drops the
model's saved interactions (bug-repros item 3). Cold starts were too low,
e.g. Spam 0.38 instead of 0.51. Rerun with `feedback.load_pretrained`; the
conclusions did not change.

## Lockbox learning curve (fast-decisions, 17 domains; see "What the lockbox is")

`src/vwjev/lockbox_curve.py`: per domain, 10 random 50/50 splits of the 100
texts into a test half and a learning pool; the model learns from 0–50 pool
texts one at a time. Settings were fixed in advance and every adapted model
is thrown away, so the lockbox is still unused for tuning. Macro-F1 pooled
over the 10 splits, multi-label questions excluded:

| Start / feedback | 0 | 5 | 10 | 20 | 30 | 50 texts |
|---|---|---|---|---|---|---|
| pre-trained, corrections | 0.301 | 0.266 | 0.283 | 0.318 | 0.332 | 0.365 |
| from scratch, corrections | 0.093 | 0.225 | 0.257 | 0.304 | 0.334 | **0.374** |
| pre-trained, right/wrong only | 0.301 | 0.276 | 0.284 | 0.275 | 0.268 | 0.270 |
| from scratch, right/wrong only | 0.093 | 0.202 | 0.211 | 0.221 | 0.225 | 0.234 |
| **OpenAI, no training (same questions)** | **0.603** | | | | | |

After 50 labeled texts, VW is closest on agent handoff (0.59 vs 0.62),
ticket routing (0.34 vs 0.37) and email triage (0.33 vs 0.43). Nowhere
does it reach OpenAI. Right/wrong-only feedback does not help at this scale.

## OpenAI Decisions API benchmark (fold 0)

`src/vwjev/openai_decide.py` and `openai_report.py`; `gpt-6-luna`, same
questions as VW (same text cut and option order), multi-label skipped.
Macro-F1 (accuracy):

| Set | For VW this is | VW full | VW generic | OpenAI | Random |
|---|---|---|---|---|---|
| seen_test (6,059 q) | in-domain | 0.600 (0.621) | 0.448 (0.547) | **0.677** (0.695) | 0.359 |
| dev (7,750 q) | out-of-domain | 0.293 (0.381) | 0.299 (0.395) | **0.640** (0.704) | 0.325 |
| lockbox (2,600 q) | out-of-domain | 0.290 (0.359) | 0.301 (0.387) | **0.617** (0.633) | 0.209 |

- OpenAI never trained on any of this; all three sets are out-of-domain for
  it. It beats VW even on the domains VW trained on (0.677 vs 0.600).
  Without in-domain data, VW is at random level.
- OpenAI refused 362 of 16,409 questions (2.2%); refusals count as wrong.
- **Confidence:** OpenAI's most confident 25% of answers are 99% accurate
  on dev, 90% on seen_test and 84% on the lockbox. VW's are 45–60% out of
  domain and 79% in-domain.
- **Cost:** 11,300 requests, 4.80M billed input tokens, **$0.48**. Billed
  tokens ran 1.27× the o200k estimate; a linear fit gives billed ≈ 1.04 ×
  estimate + 76 tokens per request. Rate limits seen: 5,000 requests and 2M
  tokens per minute. Server processing time per request: median 63 ms, p90
  94 ms, p99 425 ms (network time not included).

**Correction (2026-10-20): OpenAI's yes/no answers were mis-scored.**
Option order is shuffled per question, so "yes" is not always the first
option, but the scoring mapped OpenAI's "yes" to the first option. About
half of OpenAI's yes/no answers on BEV dev and seen_test were scored
backwards. Fixed in `openai_report.openai_preds`; nothing was re-sent.
OpenAI went from 0.557 to 0.640 on dev and from 0.596 to 0.677 on
seen_test (Spam: 0.52 to 0.91). Lockbox numbers are unchanged: its options
are not shuffled, so "yes" was always first. Earlier text saying VW "ties"
OpenAI in-domain was wrong.

GLiNER2.5-Decide: scripts ready (`bench/gliner_decide.py`, REPRODUCE.md
step 5), not yet run; it needs a machine with more free memory.

## Cold start: OpenAI first, then VW (fold 0)

`src/vwjev/cascade.py`, `cascade_report.py`. The product scenario: no
labeled data on day one.

**Takeaway.** Training VW on OpenAI's answers and letting it take the texts
it is sure about works as a way to cut cost, not to replace OpenAI. It is
worth building only for high-volume, keyword-driven tasks. Everywhere else,
keep paying OpenAI.

- **It barely hurts quality.** With a confidence gate, VW takes over about
  a third of the traffic on average, and macro-F1 drops only from 0.641 to
  0.629.
- **The task decides almost everything:**
  - Spam: VW takes 89% of texts after about 100 OpenAI answers, at equal or
    better quality.
  - Tool routing: VW takes 73%.
  - Judgment tasks (sentiment, rules, preferences): VW takes 0–16%, however
    long it learns.
- **How to run it:** a confidence gate that needs no true labels, plus 2–5%
  random audits. No fixed warm-up length works, even 1,600 texts.
- **The money is small unless volume is large.** OpenAI already costs
  only about $38 per million texts. The cascade pays for its upkeep above
  about $8,500 a month of decision calls; below that, the engineering
  costs more than it saves.
- **Not yet tested:** what happens when the incoming texts or the schema
  change mid-stream.

- **The stream:** texts arrive one at a time. A routing rule sends each one
  to OpenAI (paid; the user gets OpenAI's answer and VW learns from it) or
  to VW (free; the user gets VW's answer).
- **Background answers:** VW answers every text in the background, so each
  OpenAI call also shows whether VW would have agreed.
- **Scoring:** true labels are used only to score what users received.
- **Data:** the 7 BEV domains held out in fold 0, 1,300–6,063 texts each:
  up to 3,000 BEV training texts per domain (newly scored by OpenAI for
  $0.77) plus the dev texts from the benchmark. Each domain runs in 3
  random arrival orders.
- **VW model:** VW starts from scratch. In the feedback runs, scratch and
  pre-trained models were level after about 100 examples.

Routing rules:

- **Warm-up N:** OpenAI for the first N texts, then VW. Optionally a random
  R share still goes to OpenAI as audits, and VW keeps learning from those.
- **Gate A:** VW answers a text when its confidence clears a threshold. The
  threshold is picked from recent OpenAI calls: above it, VW agreed with
  OpenAI on at least A of the texts. Everything else goes to OpenAI, plus
  random audits. A product can run this rule without any true labels.

Mean over the 7 domains (macro-F1 of what users received; "later on" = the
last 1,000 texts of each stream):

| Rule | Share to OpenAI | Macro-F1 | Later on: share to OpenAI | Later on: macro-F1 |
|---|---|---|---|---|
| Always OpenAI | 100% | 0.638 | 100% | 0.641 |
| OpenAI for 200 texts, then VW | 7% | 0.428 | 0% | 0.407 |
| OpenAI for 400, then VW + 10% audits | 23% | 0.486 | 12% | 0.463 |
| OpenAI for 1,600, then VW + 10% audits | 57% | 0.545 | 28% | 0.489 |
| Gate 80% + 5% audits | 59% | 0.604 | 53% | 0.607 |
| Gate 90% + 2% audits | 70% | 0.622 | 65% | 0.629 |
| Gate 90% + 5% audits | 72% | 0.624 | 66% | 0.632 |
| Gate 90% + 20% audits | 77% | 0.629 | 71% | 0.634 |
| Gate 95% + 5% audits | 83% | 0.634 | 77% | 0.638 |

Per domain, with the "Gate 90% + 2% audits" rule. VW accuracy is at the end
of the stream, learning from OpenAI's answers vs from true labels:

| Domain | Texts | VW first answers at text # (median) | Later on: share moved to VW | Later on: macro-F1 served | Always OpenAI | VW accuracy: OpenAI / true labels |
|---|---|---|---|---|---|---|
| Spam | 2,209 | 102 | 89% | 0.958 | 0.915 | 0.94 / 0.98 |
| Tool routing | 4,212 | 602 | 73% | 0.887 | 0.987 | 0.92 / 0.93 |
| Sentiment | 3,606 | 509 | 5% | 0.654 | 0.653 | 0.67 / 0.76 |
| Rule-conditioned | 6,063 | 2321 | 10% | 0.871 | 0.880 | 0.55 / 0.58 |
| Calibrated prob. | 3,867 | 958 | 16% | 0.487 | 0.488 | 0.62 / 0.70 |
| Sensory rating | 1,300 | 207 | 53% | 0.219 | 0.233 | 0.33 / 0.32 |
| Response pref. | 3,634 | never | 0% | 0.331 | 0.331 | 0.36 / 0.50 |

**How many examples before switching to VW?** No fixed number works.

- **Fixed warm-ups lose too much.** After 1,600 OpenAI answers, going
  all-VW still drops later-on macro-F1 from 0.641 to 0.464.
- **The gate decides per domain:**
  - Spam: VW starts after about 100 texts and then takes 89% of the
    traffic. Its quality beats OpenAI's (0.958 vs 0.915), because VW
    trained on OpenAI's answers is right more often than OpenAI on spam.
  - Tool routing and sensory ratings: VW takes 50–75%.
  - Judgment tasks: VW takes 0–16%.
  - The gate needs no true labels to make these calls.
- **Agreement targets:** a 95% target protects quality better (tool
  routing 0.94 vs 0.89 at 90%) but moves less work. 80% moves more and
  loses more.

**How often to re-check with OpenAI?**

- **In a steady stream, 2–5% random audits are enough.** Going from 2% to
  20% audits raises the later-on OpenAI share from 65% to 71% and macro-F1 only
  from 0.629 to 0.634.
- **Audits are mostly for catching change:** new kinds of text, or an
  edited schema. This simulation does not include that.
- **Size audits by count, not share:** about 200 audits per period
  measure agreement to within ±4 points.

**Does the teacher hold VW back?** A little. Learning from true labels
instead would add up to 14 accuracy points by the end of the stream (response preference; about 1 point on tool routing). The rest
of the gap to OpenAI is VW's own limit.

### Cost of ownership (illustrative)

- **OpenAI price at these lengths:** about $38 per million texts
  (383 billed tokens per text, 1.46 questions per text).
- **Share moved to VW:** with "Gate 90% + 2% audits", 35% of texts move to VW
  later on, averaged over these 7 domains.
- **Assumptions:** all are constants at the top of `cascade_report.py`.
  - Building the cascade costs $20,000 once (about 4 engineer-weeks).
  - Upkeep costs $3,000 a month (about a tenth of an engineer).
  - VW runs on $40-a-month vCPUs, provisioned at 3× the measured
    0.55 ms per question.

| Scenario | Monthly decision spend | Texts / s | OpenAI bill after | VW vCPUs | Net saving / month | Pays back build in |
|---|---|---|---|---|---|---|
| Ramp median | $2,246 | 23 | $1,458 | 1 ($40) | −$2,252 | never |
| $100K / month | $100,000 | 1,009 | $64,904 | 3 ($120) | $31,976 | under a month |
| Ramp average | $140,842 | 1,420 | $91,412 | 4 ($160) | $46,270 | under a month |

- **Break-even:** monthly decision spend ≈ upkeep ÷ share moved to VW ≈
  **$8,500 a month**. Server cost is negligible: 1,000 texts/s needs a
  few vCPUs.
- **Where the scenarios come from:** the Ramp AI Index (April 2026) median
  ($2,246) and average ($140,842) are monthly business AI spend across all
  uses, mostly text generation. Only the share spent on decision-style
  calls counts here.
- **What this means:**
  - A company whose decision calls cost a few thousand a month will not
    recover the effort.
  - One spending tens of thousands a month on decision calls pays it back
    within a month.
- **The task mix matters as much as volume.** For a spam-like workload
  (89% moved), break-even is about $3,400 a month. For judgment-heavy
  workloads it never pays, because almost nothing moves.
- **Quality is not priced in.** In most domains offloading costs some
  macro-F1, as the per-domain table shows.

## OpenAI Decisions API cost estimate (for the final benchmark)

`src/vwjev/cost_estimate.py`, $0.10 per 1M input tokens, tokens counted with
o200k_base as a stand-in, one request per text:

| Set | Requests | Tokens | Cost |
|---|---|---|---|
| fast-decisions lockbox | 1,700 | 0.38M | $0.04 |
| one fold's eval sets | 9–11k | 3.5–4.8M | $0.35–0.48 |
| all 4 folds' eval sets | 25k | 10.8M | $1.08 |
| BEV test, all configs | 45k | 17.6M | $1.76 |
| BEV train + test | 276k | 110M | $11.03 |

Largest request ≈ 5.1k tokens. Rate limits (unpublished) will matter more
than cost.

## Planned

- GLiNER2.5-Decide on fold 0 (lockbox + 400-text BEV samples), on a larger
  machine.
- Cold start with drift: a new schema or a new kind of text arriving
  mid-stream, to test whether the gate and audits catch it.
- Multilingual check (e.g. MASSIVE) for the "monolingual vs multilingual"
  axis.

## Data and metric gotchas found

- **The BEV mix leaks answers through option order.** The right answer is
  listed first far more often than chance in several domains (Support and
  intent routing 47% vs 20% expected; Trap and adversarial reading 54% vs
  22%; Civic and safety 44% vs 22%). Options are now shuffled for training
  and BEV evaluation, except score scales, whose order carries meaning. The
  lockbox shows no such bias (17% vs 17%) and keeps its original order.
- **Pooled macro-F1 is unstable with one-off classes.** Some domains contain
  many single-row questions; a lucky guess scores F1 = 1 on each. An
  untrained model reached 0.863 macro-F1 at 44% accuracy in the Tool domain.
  Classes with fewer than 5 gold examples are now left out of the macro
  average (they still count toward accuracy).

## VW gotchas found (9.11.9, Python bindings)

- `--csoaa_ldf` / `--wap_ldf` **segfault** when predicting on multi-line
  examples with no labels (any variant tried, with or without a shared
  line). Workaround: give every option a dummy `i:0` cost when predicting.
- With `--csoaa_rank`, `predict()` returns scores shifted by one: the
  shared line is counted as an action (first value always 0.0, last option
  dropped). Likely VW core, not the Python wrapper. Read each option's
  `get_partial_prediction()` instead (input order, lower = better).
- Reloading a model with an extra `-q` (`-i model.vw -q fl`) **silently
  drops the model's saved interactions**. Restate them
  (`feedback.load_pretrained` does this).
- All three have standalone demos in [`../bug-repros/`](../bug-repros/README.md). [`../VW-NOTES.md`](../VW-NOTES.md) collects these with the ones `nlp/` and `word2vec/` found.
- `--probabilities` output is very flat (0.58 max on a perfectly separable
  toy), so confidence must be calibrated separately.
- A `--csoaa_ldf` model can't be loaded with `--cb_adf` / `--cb_explore_adf`
  ("model content is corrupted"). Bandit feedback is therefore turned into
  `csoaa_ldf` updates in Python.
- `--cb_adf` allows only one costed option per example.
- With all costs 0/1, `csoaa_ldf mc` trains as a classifier with ±1 targets;
  other costs switch it to plain regression, so keep feedback updates 0/1.
- Interactions can be added when reloading a model (`-i model.vw -q fl`);
  a fresh namespace starts at zero weight and learns from new data.
- System Python 3.10 has no `venv`/`ensurepip`; the project uses `uv`.
