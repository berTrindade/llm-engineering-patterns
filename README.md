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

Full run on 2026-09-16: 50 cases per approach, `qwen3:8b` through local Ollama, with no
failed calls. Raw output is in [results/2026-09-16.json](results/2026-09-16.json).

| Approach | Valid first try | Valid after repair | Field accuracy | Prompt tokens | Completion tokens |
|---|---|---|---|---|---|
| prompt_only | 49/50 | 50/50 | 98.4% | 282 | 220 |
| json_mode | 49/50 | 50/50 | 97.8% | 280 | 220 |
| json_schema | 47/50 | 50/50 | 97.6% | 314 | 229 |

Field accuracy is over 500 field checks per approach (10 fields per case). Token columns
are means per case and include the repair call when there was one.

- **On this model and data, the three approaches are close to indistinguishable.** Every
  case produced a valid `Invoice` after at most one repair, and field accuracy differs by
  4 checks out of 500 (8, 11 and 12 wrong fields). With 50 cases that gap is noise.
- **Constrained decoding did not reduce invalid first replies here.** `json_schema` needed
  the repair retry three times, the other two once each. The results file does not keep the
  per-case replies, so the cause of those three is not recorded.
- **The repair retry is what makes all three reliable.** Without it, validity would have
  been 94% to 98%; with one retry it is 100%.
- **The extra prompt tokens for `json_schema` probably come from its repairs.** A repair
  call resends the original prompt, the first reply and the validation errors, so each one
  costs several hundred prompt tokens; two extra repairs across 50 cases would explain most
  of the 34-token gap. This is an inference: the results file does not split first-call and
  repair tokens. `prompt_only` and `json_mode`, each with one repair, are within 3 tokens of
  each other.

Mean latency was 7.3 s for `prompt_only` and 6.1 s for the other two. The approaches ran one
after another, `prompt_only` first, so that gap is not attributed to the approach.

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
