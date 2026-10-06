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
EVAL_RESUME=1 make eval   # continue an interrupted run from its results file
make test    # pytest, no network or model calls
make lint    # ruff check
make calibrate   # fits the token estimator against real API usage, writes data/calibration.json
```

Needs a local Ollama running the models in `.env.example` (`qwen3:8b` by default). Copy
`.env.example` to `.env` to override the endpoint or model names.

## Result

Full run on 2026-09-16: 20 threads, 200 cases, `qwen3:8b` through local Ollama, with no
failed cases and no truncation flags. Raw output is in
[results/2026-09-16.json](results/2026-09-16.json).

| Strategy | Budget | Fact recall | Prompt tokens | Summary tokens | Estimator error |
|---|---|---|---|---|---|
| full_history | none | 1.00 | 15,594 | 0 | 0.5% |
| last_n | 2000 | 0.01 | 1,591 | 0 | 3.0% |
| last_n | 4000 | 0.03 | 3,588 | 0 | 1.3% |
| last_n | 8000 | 0.03 | 5,075 | 0 | 1.2% |
| token_budget | 2000 | 0.01 | 1,591 | 0 | 3.0% |
| token_budget | 4000 | 0.04 | 3,588 | 0 | 1.3% |
| token_budget | 8000 | 0.03 | 7,568 | 0 | 0.9% |
| summarise_older | 2000 | 0.53 | 1,440 | 14,055 | 11.3% |
| summarise_older | 4000 | 0.65 | 3,543 | 12,072 | 4.6% |
| summarise_older | 8000 | 0.68 | 7,490 | 8,107 | 2.0% |

Every column is a mean over that row's 20 cases. Prompt tokens are the real counts Ollama
reported; estimator error is the mean absolute gap between the estimate and that count.

- **Dropping the oldest turns loses the facts.** `last_n` and `token_budget` recall almost
  none of the four early facts at any budget, even 8,000 tokens, because the facts sit in
  the first two exchanges and those are the first turns to go.
- **Summarising keeps them only some of the time.** `summarise_older` recalls 0.53 to 0.68
  of the facts, but per thread the outcome is close to all or nothing: at the 2000 budget,
  8 threads recalled none of the facts and 8 recalled all four. The mean hides whether the
  summary happened to carry the facts.
- **The summary is not free.** At the 2000 budget the summary call used 14,055 tokens on
  average, about as many as sending the full history once (15,594). Each thread here is
  answered once, so there is nothing to spread that cost over. A live conversation would
  reuse the summary across later turns; this repository does not measure that.
- **The estimator held up on real prompts.** Full-history prompts measured 14,412 to 16,959
  real tokens with a mean estimate error of 0.5%, even though the longest threads (about
  40,000 characters) run slightly past the 35,000 characters the fit was calibrated on.
  Error is highest (11.3%) on the smallest summarised prompts, where the fixed per-message
  overhead dominates.
- **Some gaps are noise.** With 20 threads and near-binary outcomes per thread, the
  difference between `summarise_older` at 4000 and at 8000 (0.65 against 0.68) means
  nothing.

Latency is left out of the table. The machine slept for part of the run, so wall-clock
latency is not comparable across cases: `last_n` and `token_budget` at the 2000 budget
send identical prompts yet averaged 5.5 s and 3.1 s. Per-case values are in the results
file.

## How it works

- **Dataset**: `src/context_window_poc/generate_threads.py` deterministically generates
  20 invented ticket threads (41 to 43 turns each, seeded, no LLM involved). Each
  thread plants four facts (account ID, plan name, error code, time zone) in its first
  two exchanges, fills the rest with unrelated invented support conversation padded
  with pasted log excerpts and invoice line items (the way a real agent pastes logs
  while investigating), and closes with a customer question that needs those four
  facts back. Full history lands at roughly 34,000-40,000 characters per thread, which
  the full run measured at 14,412 to 16,959 real prompt tokens. The local Ollama server
  reports `context_length` 40960 for `qwen3:8b`, well above that, so these prompts are
  not truncated. That
  puts every thread well above the 8,000 token budget, so `token_budget` and
  `summarise_older` actually have to trim or summarise something; a generator test
  asserts every thread clears the largest budget by a wide margin.
- **Token counting**: there is no local tokenizer for qwen models, so
  [`tokens.py`](src/context_window_poc/tokens.py) estimates tokens with a small linear
  model fitted on real usage: `tokens ~= a*chars + b*n_messages + c`. The
  `b*n_messages` term exists because every message in a chat payload carries a fixed
  chat-template overhead independent of its content, which a single chars-per-token
  ratio (fitted only on short, few-message prompts) folds into the ratio and gets wrong
  for prompts with a different message count. `make calibrate` fits `a`, `b` and `c` by
  ordinary least squares on about 12 real calls spanning roughly 200 to 35,000
  characters and a range of message counts, reports mean absolute percentage error on
  a further 4 held-out prompts, and writes the fit to `data/calibration.json`. The eval
  table reports the estimator's error against the real count for every case, and flags
  any case where real `prompt_tokens` comes back below half the estimate, as a guard
  against the server silently truncating the prompt rather than the estimator simply
  being off.
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
  mean real prompt tokens, extra tokens spent on summaries, mean latency, mean
  estimator error and a count of possible truncations. Every failed case (bad output,
  timeout, exception) is recorded with its reason and counted, never dropped silently.

## Limits

- The estimator is a 3-parameter linear model (`a*chars + b*n_messages + c`), not a
  real tokenizer. It has no ceiling guarantee: it is fitted on English support-ticket
  prose plus pasted logs and invoice lines, so it will be systematically off for text
  with a very different character mix (dense numeric IDs, non-English text, code), and
  the fit in `data/calibration.json` only covers the ~16 prompts it was trained and
  tested on, not the full dataset. The held-out error percentage reported by
  `make calibrate` is the honest ceiling on how far to trust it.
- The fitted `b` (tokens per message) came out negative (`-14.49`) on the calibration
  run committed here, which is not a physically meaningful "each extra message removes
  tokens" effect. It is because the calibration samples were built as growing prefixes
  of real threads, so character count and message count are highly correlated in that
  sample, and least squares can trade the two off against each other rather than
  identify each one separately. Held-out error stayed at 0.9%, so the fit still
  predicts well within the range it was calibrated on, but `a` and `b` individually
  should not be read as "chars cost this many tokens, messages cost that many"; only
  the combined prediction is trustworthy, and only inside the ~200-35,000 character
  range it was fitted on.
- The truncation guard (real `prompt_tokens` well below the estimate) only catches
  truncation large enough to look like estimator error; a small silent truncation could
  still hide inside normal estimator noise.
- 20 threads is a small sample; recall differences of a few percentage points between
  strategies are not statistically meaningful.
- `summarise_older`'s recall depends partly on `CHEAP_MODEL`'s own summarisation
  quality, so a low score there measures the summariser as much as the strategy.
- `last_n`'s fixed window (12 turns) was chosen once, not tuned; a different window
  would trade differently against the token budgets tested here.
- The dataset and the fact-recall scorer are both invented for this repository; results
  do not generalise to real support tickets, which are messier and less consistently
  structured.
