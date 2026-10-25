# decisions: VW as a Jev-style decision model

Can a VW model answer Jev-style decisions (yes/no, pick one, score on a
scale) well enough for low-stakes product use, such as routing a query to an
LLM tier? Two bars: zero-shot on a new schema, and cold start plus learning
from product feedback.

**Short answer:** not zero-shot. VW with no in-domain data scores at random
level on held-out domains (macro-F1 0.29-0.30 against OpenAI's 0.62-0.64).
As a cost-cutter behind OpenAI it works for keyword-driven tasks: trained on
OpenAI's answers and gated by its own confidence, VW takes about a third of
the traffic for a 0.01 drop in macro-F1 (0.641 to 0.629). For spam it takes
89%, at better quality than OpenAI's.

| File | What it has |
|---|---|
| [`FINDINGS.md`](FINDINGS.md) | every result, the data, metrics and VW gotchas |
| [`REPRODUCE.md`](REPRODUCE.md) | the commands behind every number, with expected values |
| [`PLAN-decision-index.md`](PLAN-decision-index.md) | next: scoring VW on the public Jev Decision Index |
| `report/can-vw-replace-jev.html` | the report page, built from `runs/final_report.json` and `runs/cascade/report.json` |
| `src/vwjev/` | the code (`uv run python -I -m vwjev.<module>`) |
| `scripts/` | multi-step runs (cascade simulation, GLiNER, feedback reruns) |

This folder is its own uv project (Python 3.12, VW 9.11.9). Run everything
from here. Its data and runs stay in `data/` and `runs/` here (not
committed), except the two report summaries above. The OpenAI steps need
`OPENAI_API_KEY` in `.env` (git-ignored). The VW bugs found here have
standalone demos in [`../bug-repros/`](../bug-repros/README.md).
