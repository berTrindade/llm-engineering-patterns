# cost-and-routing-poc

Cost per call from token counts and a price table, and confidence-based routing between a
cheap and a strong model, over invented Fernway Travel support questions.

## Problem

Every call to a large model has a cost and a latency. A team that always calls the strong
model pays for accuracy it does not always need; a team that always calls the cheap model
gets wrong answers on the cases that needed more capability. This proof of concept measures
that trade-off directly: it puts accuracy, cost and latency for three strategies side by
side on the same 60 questions, so the choice between them is a number, not a guess.

Fernway Travel is an invented travel booking company used only for this repository. All
policy facts, fees and timelines are made up; see [data/README.md](data/README.md).

## Run it

```bash
cp .env.example .env   # optional, defaults already point at local Ollama
make setup
make run                # one question through all three strategies
make eval                # full eval, writes results/<date>.json and prints a table
make test
make lint
```

Requires a running OpenAI-compatible endpoint at `LLM_BASE_URL` (Ollama by default) with
`CHEAP_MODEL`, `STRONG_MODEL` and `EMBED_MODEL` pulled.

For a fast local check instead of the full 60-question eval:

```bash
EVAL_LIMIT=3 STRONG_MODEL=qwen3:8b make eval
```

This writes `results/smoke-<date>.json` and keeps the 35B strong model from loading.

## Result

Not yet run in full. Run `make eval`.

## How it works

- [src/cost_and_routing/llm.py](src/cost_and_routing/llm.py) calls
  `{LLM_BASE_URL}/chat/completions` directly with `httpx`. No provider SDK. Every call sends
  `"reasoning_effort": "none"` by default, which stops `qwen3` thinking; a `/no_think` prompt
  suffix alone was not enough (a 20-token JSON answer reported 542 completion tokens without
  it, verified 2026-09-15 against Ollama 0.34.0).
- [src/cost_and_routing/pricing.py](src/cost_and_routing/pricing.py) loads
  [prices.toml](prices.toml), USD per million input/output tokens, and multiplies by the
  recorded prompt and completion tokens per call. Local Ollama models have no real bill: the
  figures here show what the recorded token counts would cost at these illustrative prices.
  Point `LLM_BASE_URL` at a paid provider and edit `prices.toml` with its real prices to get
  real cost numbers.
- [src/cost_and_routing/scoring.py](src/cost_and_routing/scoring.py) normalises an answer
  (case, currency formatting, small number words) and checks that every `required_facts`
  string from [data/questions.jsonl](data/questions.jsonl) appears in it.
- [src/cost_and_routing/routing.py](src/cost_and_routing/routing.py) has the cheap model
  answer with a `json_schema` response of `{answer, confidence, needs_escalation}`, and
  escalates to the strong model when confidence is below a threshold or
  `needs_escalation` is true. The routed cost always includes the cheap attempt, plus the
  strong attempt when it escalates. A cheap response that fails to parse as valid JSON is
  treated as a genuine failure and forces escalation, never silently dropped.
- [evals/run.py](evals/run.py) runs `always_cheap`, `always_strong` and `routed` (swept over
  thresholds 0.5, 0.7, 0.9) across the question set, aggregates accuracy, escalation rate,
  total cost, cost per correct answer, mean latency and mean tokens per strategy, buckets the
  cheap model's self-reported confidence against its actual accuracy (calibration), and runs
  one extra pair of calls on the same question with `reasoning_effort` omitted versus `"none"`
  to show the thinking-token difference directly.

## Limits

- 60 invented questions across three difficulty tags (`simple_lookup`,
  `multi_step_calculation`, `policy_exception`); this is a small sample, not a statistically
  powered benchmark.
- Prices in `prices.toml` are illustrative, not real provider prices; local models have no
  real bill.
- Scoring is exact substring matching after normalisation, so an answer that gets the
  substance right but does not use the exact required phrase is marked wrong even if a human
  reader would accept it. In a smoke run every strategy answered "can I change my name 30
  hours before departure" correctly in substance (explaining the change is not allowed) but
  never used the dataset's required phrase "not permit", so all five strategy rows mark that
  case wrong. This is a dataset/scorer strictness limit, not a routing or cost bug.
- The smoke command (`EVAL_LIMIT=3 STRONG_MODEL=qwen3:8b`) sets the cheap and strong model to
  the same model so the 35B model is not loaded during development; its cost and escalation
  figures are not meaningful, only `make eval`'s full run with distinct models is.
