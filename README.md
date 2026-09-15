# structured-outputs-poc

## Problem

Supplier invoices often arrive as messy emailed text rather than structured data: forwarded
threads, inconsistent date formats, currency shown as a symbol or a code, line items as a
table, a bullet list or a sentence. This repository compares three ways of turning that text
into a validated `Invoice` record with a local LLM:

- **prompt_only**: instructions in the prompt asking for JSON, parsed out of free text.
- **json_mode**: `response_format={"type": "json_object"}`.
- **json_schema**: `response_format={"type": "json_schema", ...}` built from
  `Invoice.model_json_schema()`.

Each approach gets one repair retry: if the first reply fails to parse or fails Pydantic
validation, the model is shown its own validation errors and asked to try again.

## Run it

```bash
make setup   # uv sync
make run     # one example end to end, json_schema approach
make eval    # full eval over all 50 cases, writes results/<date>.json
make test    # pytest, no network, no model
make lint    # ruff check
```

Requires a local OpenAI-compatible server (Ollama by default) at `LLM_BASE_URL`. Copy
`.env.example` to `.env` to override the defaults.

## Result

Not yet run in full. Run `make eval`.

A smoke run (`EVAL_LIMIT=3`, `CHEAP_MODEL` only) is committed at
`results/smoke-<date>.json` to prove the pipeline runs end to end; it is not the headline
result.

## How it works

- `data/cases.jsonl`: 50 invented invoice cases, each a ground-truth `Invoice` and a messy
  email rendering of it. Generated deterministically by
  `src/structured_outputs/generator.py`, no LLM involved. See `data/README.md`.
- `src/structured_outputs/schema.py`: the `Invoice` and `LineItem` Pydantic models.
- `src/structured_outputs/llm.py`: a small httpx client for
  `POST {LLM_BASE_URL}/chat/completions`, always sending `"reasoning_effort": "none"` so
  qwen3 does not spend tokens thinking.
- `src/structured_outputs/extract.py`: the three approaches, each one call plus one optional
  repair call.
- `src/structured_outputs/scoring.py`: per-field scoring (money to 2dp, dates as dates, line
  items as a multiset) and per-approach aggregation.
- `evals/run.py`: loads the cases, runs all three approaches, prints a table and writes
  `results/<date>.json`. Set `EVAL_LIMIT` to run only the first N cases.

## Limits

- All data is invented for this project; no real supplier or invoice is represented.
- Every case uses the same model for all three approaches, so this measures the technique,
  not model choice.
- The repair retry is capped at one attempt; a model that fails twice is recorded as a
  failing case, not retried further.
- Line item matching treats two items with the same description, quantity and unit price as
  identical even if the source order differed; it does not check the line items are in the
  same order as the email.
