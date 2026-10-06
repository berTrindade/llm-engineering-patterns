# evals-and-judge-poc

## Problem

Anyone shipping an LLM feature needs a way to say whether it is any good, and that answer has to
be checked, not assumed. This POC builds a small evaluation harness for an invented online shop's
customer support bot, "Northwind Home": a golden set of replies, deterministic scorers that need
no model, an LLM judge with a rubric and structured output, and a check of how well that judge
actually agrees with a human, plus whether the order the two candidate replies are shown in
changes its verdict.

## Run it

```bash
make setup   # uv sync
make run     # one example end to end
make eval    # full eval, writes results/<date>.json and prints a table
make label   # terminal CLI to add your own pass/fail labels, resumable
make test    # pytest, no network, no model
make lint    # ruff check
```

Needs local Ollama serving an OpenAI-compatible API at `http://localhost:11434/v1` with the
models named in `.env.example` pulled. Copy `.env.example` to `.env` to override any of it.

## Result

Full run on 2026-09-16: 60 replies and 20 position-bias pairs, judged by `qwen3.5:35b`
through local Ollama, with no failed judge calls. Raw output is in
[results/2026-09-16.json](results/2026-09-16.json).

**No human labels yet.** `make label` has not been run, so judge agreement with a human, the
number this repository is built to report, is not available. Everything below compares
against `designed_label`, the label the generator assigned when it injected (or did not
inject) a flaw.

Judge agreement with designed labels: **54/60 (90%)**.

| Flaw type | Judge correct | Deterministic scorer that targets it |
|---|---|---|
| none (good reply) | 29/30 | all three pass all 30 |
| rude_tone | 6/6 | none |
| forbidden_promise | 6/6 | `no_forbidden_promise`: 6/6 |
| wrong_policy_fact | 6/6 | none |
| ignores_question | 6/6 | `order_number_echoed` also catches 6/6 |
| missing_order_number | **1/6** | `order_number_echoed`: 6/6 |

- **The judge's blind spot is the one thing a regex does perfectly.** On five of the six
  replies with the order number removed, the judge passed the reply and its reason said the
  reply "repeats the correct order number"; one reason quoted the number from the
  customer's message as if the reply contained it. `order_number_echoed` caught all six. Neither check alone covers the set: the judge
  catches tone, policy and relevance flaws no scorer can see, and the scorer catches the
  omission the judge reads past.
- **The earlier run hid this.** The first full run scored 6/6 on this flaw, but the
  generator then left "about order your order" in those replies, an easy cue. Fixing the
  wording (see the git history) removed the cue and exposed the miss.
- **One false fail.** The judge failed good reply case-016, calling its "order over GBP 50"
  wording a policy error. The reply asserts the order total without the customer stating
  it, which is arguable, but the same template passed on the other good return replies.
- **Position bias was low.** Across 20 pairs, the judge picked the designed better reply in
  both orders 19 times; one pair (pair-001, forbidden promise) flipped. The first-shown reply
  won 52.5% of the 40 trials, close to the 50% an unbiased judge would give.
- **`length_within_bounds` flagged nothing.** Every reply, flawed or not, fell inside its
  bounds, so on this data it adds no signal.

The scorer agreement figures printed by `make eval` (70%, 60%, 50%) are each scorer against
all 60 designed labels, so they mostly count the flaws a scorer was never meant to detect.
Judge mean latency was 2.5 s per call (median 2.2 s; the first call, which loads the model,
took 14 s), with 786 prompt and 73 completion tokens on average.

## How it works

- `data/policy.md`: the invented Northwind Home policy (returns, delivery windows, damaged
  items, personalised items) that both the replies and the judge are grounded in.
- `src/evals_judge/dataset.py`: a seeded, pure-Python generator, no LLM, that writes
  `data/replies.jsonl`, 60 cases. Half are good replies; half each carry exactly one deliberately
  injected flaw (wrong policy fact, missing order number, forbidden promise, rude tone, ignores
  the question), recorded as `designed_label`.
- `src/evals_judge/scorers.py`: three deterministic scorers (order number echoed, no forbidden
  promise, reply length in bounds). Each only catches its own kind of flaw, so its agreement with
  the full designed-label set is reported honestly rather than inflated.
- `src/evals_judge/judge.py`: an LLM judge with a rubric (`evals/rubric_prompt.md`) that returns
  structured output (`verdict`, `failed_criteria`, `reason`) via `response_format` json_schema.
- `src/evals_judge/labels.py`: `make label` walks you through every reply (customer message,
  reply, policy) and asks pass or fail plus an optional note, saving each answer immediately to
  `data/labels.human.jsonl` so it resumes where it left off.
- `src/evals_judge/kappa.py`: Cohen's kappa, implemented from the standard formula and checked
  against a textbook contingency table.
- `src/evals_judge/position_bias.py`: 20 pairs of replies to the same message, one better by
  design; the judge picks A or B in both orders, and the run reports the flip rate and the share
  of wins that go to whichever reply is shown first.
- `evals/run.py`: runs all of the above, prints the results table, and writes
  `results/<date>.json` (or `results/smoke-<date>.json` when `EVAL_LIMIT` is set).

Everything talks to the model through `src/evals_judge/llm.py`, a plain httpx POST to
`{LLM_BASE_URL}/chat/completions` with `"reasoning_effort": "none"` so qwen3 models skip their
thinking pass. A model call that fails, times out, or returns invalid JSON is recorded per case
with a reason; nothing is silently dropped.

## Limits

- The dataset is templated, not written by real customers or a real support team; it is useful
  for measuring the evaluation method, not for judging real Northwind Home replies (Northwind
  Home is invented).
- The headline number is judge agreement with human labels. Until `make label` has been run,
  there are no human labels and the table says so plainly instead of guessing.
- The judge and the position-bias pairs share the same rubric and policy text, so a mistake in
  the rubric affects both; the deterministic scorers are the only fully independent check.
- Position bias is only measured with one judge model at a time; it does not compare how flip
  rate changes across models.
