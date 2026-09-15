# context-window-poc

## Problem

Long support ticket threads outgrow what fits in a model's context window, or what a
team is willing to pay for on every turn. When older turns have to be dropped or
compressed, facts a customer gave early in the conversation (their account ID, plan,
the original error code, their time zone) can get lost before the ticket closes. This
repository measures how three common history-trimming strategies affect recall of
those early facts at three token budgets, using invented Ledgerline (a fictional
small-business invoicing product) support tickets.

## Run it

```bash
make setup   # uv sync
make run     # one thread through every strategy and budget, prints a table
make eval    # full eval over all 20 threads, writes results/<date>.json
make test    # pytest, no network or model calls
make lint    # ruff check
make calibrate   # fits the token estimator against real API usage, writes data/calibration.json
```

Needs a local Ollama running the models in `.env.example` (`qwen3:8b` by default). Copy
`.env.example` to `.env` to override the endpoint or model names.

## Result

Not yet run in full. Run `make eval`.

A smoke run (`EVAL_LIMIT=3`, one budget, `CHEAP_MODEL` only) is in
`results/smoke-<date>.json`, to confirm the pipeline runs end to end. It is not a
result to draw conclusions from.

## How it works

- **Dataset**: `src/context_window_poc/generate_threads.py` deterministically generates
  20 invented ticket threads (40-80 turns each, seeded, no LLM involved). Each thread
  plants four facts (account ID, plan name, error code, time zone) in its first two
  exchanges, fills the rest with unrelated invented support conversation, and closes
  with a customer question that needs those four facts back.
- **Token counting**: there is no local tokenizer for qwen models, so
  [`tokens.py`](src/context_window_poc/tokens.py) estimates tokens as
  `characters / chars_per_token` and separately records the real `prompt_tokens` the
  API reports for every call. `make calibrate` fits `chars_per_token` from real usage on
  a handful of prompts and writes it to `data/calibration.json`; the eval table reports
  the estimator's error against the real count for every case.
- **System prompt**: a fixed Ledgerline support-agent prompt is always sent and always
  counted against the budget; every strategy is required to keep it.
- **Strategies** ([`strategies.py`](src/context_window_poc/strategies.py)), each a pure
  function of `(system_prompt, turns, budget_tokens, estimator)`:
  - `last_n`: keeps a fixed window of the most recent 12 turns (a common naive
    default), trimmed further only if that window would still exceed the budget. It
    does not use any extra budget headroom beyond its fixed window.
  - `token_budget`: greedily drops the oldest turns until the estimate fits the budget,
    so it adapts to however much budget is available.
  - `summarise_older`: keeps the same adaptive recent window as `token_budget`, but
    replaces everything older with one summary produced by `CHEAP_MODEL`. The summary
    call's own tokens count as the strategy's extra cost, and summaries are cached per
    thread and budget within a run so the same summary is not paid for twice.
  - `full_history`: the no-trimming baseline, for comparison.
- **Budgets**: 2000, 4000 and 8000 tokens, plus the full-history baseline.
- **Scoring**: [`scoring.py`](src/context_window_poc/scoring.py) normalises text
  (lowercase, punctuation stripped, whitespace collapsed) and scores fact recall as the
  share of the four expected fact values present in the final answer.
- **Eval harness**: [`evals/run.py`](evals/run.py) runs every thread through every
  strategy and budget, scores each case, and prints strategy, budget, mean fact recall,
  mean real prompt tokens, extra tokens spent on summaries, mean latency and mean
  estimator error. Every failed case (bad output, timeout, exception) is recorded with
  its reason and counted, never dropped silently.

## Limits

- The estimator is a single global ratio (characters per token). It has no ceiling
  guarantee against the real tokenizer: it will be systematically off for text that is
  unusually dense in short tokens (numbers, IDs, code) or unusually sparse (long English
  prose), and the calibration in `data/calibration.json` reflects only the handful of
  prompts it was fitted on, not the full dataset.
- 20 threads is a small sample; recall differences of a few percentage points between
  strategies are not statistically meaningful.
- `summarise_older`'s recall depends partly on `CHEAP_MODEL`'s own summarisation
  quality, so a low score there measures the summariser as much as the strategy.
- `last_n`'s fixed window (12 turns) was chosen once, not tuned; a different window
  would trade differently against the token budgets tested here.
- The dataset and the fact-recall scorer are both invented for this repository; results
  do not generalise to real support tickets, which are messier and less consistently
  structured.
