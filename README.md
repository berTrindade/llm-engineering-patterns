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

Full run on 2026-09-16: 60 questions, `qwen3:8b` as the cheap model and `qwen3.5:35b` as the
strong model, through local Ollama, with no failed calls. Raw output is in
[results/2026-09-16.json](results/2026-09-16.json). Costs use the illustrative prices in
`prices.toml`, so only the ratios mean anything.

| Strategy | Threshold | Correct | Escalated | Total cost (USD) | Cost per correct | Median latency |
|---|---|---|---|---|---|---|
| always_cheap | | 58/60 | | 0.0051 | 0.00009 | 0.9 s |
| always_strong | | 60/60 | | 0.0368 | 0.00061 | 1.8 s |
| routed | 0.5 | 57/60 | 0/60 | 0.0055 | 0.00010 | 1.6 s |
| routed | 0.7 | 58/60 | 0/60 | 0.0055 | 0.00009 | 1.6 s |
| routed | 0.9 | 57/60 | 0/60 | 0.0054 | 0.00010 | 1.6 s |

- **Routing never routed.** The cheap model reported a confidence of 0.95 or higher on all 180
  routed calls and never set `needs_escalation`, so not one question reached the strong
  model at any threshold. The routed rows are the cheap model answering again in a different
  output format, and the gaps between them (57 to 58 correct) are run-to-run variation.
- **The confidence is not a probability.** On 26 of those calls the cheap model returned
  `100.0`, reading the field as a percentage. The schema asks for a number without bounds,
  and nothing in the output told the router the model was unsure, including on the answers
  it got wrong. Self-reported confidence from this model is not a usable routing signal.
- **The strong model was right on every question, at about 7 times the cost.** The cheap
  model missed two multi-step questions: it priced an extra-legroom seat at £12 instead of
  £25, and gave 28 February as the deadline for a refund window that starts on 1 March.
- **Routed answers were slower and slightly more expensive than always_cheap** with no
  accuracy gain: the structured `{answer, confidence, needs_escalation}` response costs
  about 20 extra tokens per call. Three routed answers to the Basic checked-bag question
  were just `0`, which the scorer cannot accept (see Limits).
- **Thinking tokens are most of the bill when left on.** On the same question, `qwen3:8b`
  used 408 completion tokens with `reasoning_effort` omitted and 41 with it set to
  `"none"`, which at these prices more than doubles the cost of that call.

What would make routing useful here is a signal the model cannot talk itself out of, for
example a verifier or a check on the answer, rather than its own confidence. This
repository does not implement one.

## How it works

- [src/cost_and_routing/llm.py](src/cost_and_routing/llm.py) calls
  `{LLM_BASE_URL}/chat/completions` directly with `httpx`. No provider SDK. Every call sends
  `"reasoning_effort": "none"` by default, which stops `qwen3` thinking; a `/no_think` prompt
  suffix alone was not enough (verified 2026-09-15 against Ollama 0.34.0). The
  `thinking_tokens_check` block in
  [results/2026-09-16.json](results/2026-09-16.json) records the measured difference on one
  question: 408 completion tokens with `reasoning_effort` omitted, 41 with it set to
  `"none"`.
- [src/cost_and_routing/pricing.py](src/cost_and_routing/pricing.py) loads
  [prices.toml](prices.toml), USD per million input/output tokens, and multiplies by the
  recorded prompt and completion tokens per call. Local Ollama models have no real bill: the
  figures here show what the recorded token counts would cost at these illustrative prices.
  Point `LLM_BASE_URL` at a paid provider and edit `prices.toml` with its real prices to get
  real cost numbers.
- [src/cost_and_routing/scoring.py](src/cost_and_routing/scoring.py) normalises an answer
  (case, currency formatting, small number words) and checks that every `required_facts`
  entry from [data/questions.jsonl](data/questions.jsonl) appears in it. A required fact can
  list alternative wordings separated by `|`, satisfied when any one of them appears, so the
  score tests whether the fact was stated, not which words the model chose to state it.
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
- Scoring is substring matching after normalisation. A required fact can list alternative
  wordings separated by `|` (for example `"not permit|cannot|can't|not allowed|not possible"`),
  which is satisfied when any alternative appears, so a correctly-worded paraphrase is not
  marked wrong just for using different phrasing. The alternatives are deliberately specific
  (never a bare "yes" or "no") to avoid the opposite failure: a required fact that is too loose
  and accepts a wrong answer. What this cannot catch is an answer that contains the right fact
  alongside a wrong claim elsewhere in the same answer, since a match only checks that the fact
  is present, not that the rest of the answer is consistent with it.
- Substring matching also cannot express an exact short answer. The Basic checked-bag
  question requires wording such as "no checked bags" or "0 checked bags", so a reply that
  is only "0" scores as wrong even though it is correct; accepting a bare "0" would also
  accept any answer that happens to contain a zero.
- The smoke command (`EVAL_LIMIT=3 STRONG_MODEL=qwen3:8b`) sets the cheap and strong model to
  the same model so the 35B model is not loaded during development; its cost and escalation
  figures are not meaningful, only `make eval`'s full run with distinct models is.
