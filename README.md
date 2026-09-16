# tool-calling-poc

## Problem

Can a hand-written agent loop, with no agent framework and no provider SDK, reliably turn a
scheduling request into the right calls to a set of tools, recover when it gets a tool call
wrong, and correctly decline requests that cannot be satisfied? This proof of concept answers
that with a meeting scheduler for an invented office, Harbour Street Studio: 5 rooms, 12 people
across 5 time zones, and a small calendar. The loop, the tools, and the domain rules (room
capacity, double booking, working hours) are all written from scratch in this repository.

## Run it

```
make setup   # uv sync
make run     # one scheduling request through the loop, end to end
make eval    # scripted eval, prints a results table, writes results/<date>.json
make test    # deterministic unit tests, no model, no network
make lint    # ruff check
```

Requires a local Ollama (or any OpenAI-compatible `/v1/chat/completions` endpoint) with
`qwen3:8b` pulled. Copy `.env.example` to `.env` to point at a different server or model.

## Result

Not yet run in full. Run `make eval`.

A smoke run (`EVAL_LIMIT=3`, `CHEAP_MODEL` only, 3 of the 30 tasks) is in
`results/smoke-2026-09-15.json`, to confirm the harness runs end to end. It is not the headline
result.

## How it works

- `data/office.json` seeds an in-memory `OfficeState`: 5 rooms with capacity and equipment,
  12 people with working hours and IANA time zones, and 6 existing meetings. `now` is a fixed
  timestamp in the seed file, not the real clock, so every eval run sees the same "today".
- `src/tool_calling_poc/state.py` holds the domain rules: a room can't be over capacity or
  double-booked, a person can't be double-booked or booked outside their working hours (checked
  in their own time zone via `zoneinfo`), and every id has to exist.
- `src/tool_calling_poc/tools.py` wraps the state's methods as five tools (`find_free_slots`,
  `list_rooms`, `book_meeting`, `cancel_meeting`, `list_meetings`), each with a pydantic argument
  model. The JSON schema sent to the model comes straight from `model_json_schema()`.
- `src/tool_calling_poc/llm.py` is a small `httpx` client that POSTs to
  `{LLM_BASE_URL}/chat/completions` and parses the OpenAI-style response (content, tool_calls,
  usage). It always sends `"reasoning_effort": "none"`, which is what actually stops qwen3
  thinking; a `/no_think` prompt suffix alone was not enough when this was checked against Ollama
  0.34.0 on 2026-09-15.
- `src/tool_calling_poc/agent.py` is the loop itself: send the messages and tool specs, run
  whatever tools come back, validate each call's arguments with pydantic, and feed the error text
  back to the model as the tool result when validation fails or the tool raises a domain error, so
  the model gets a chance to correct itself. It stops on a plain assistant message or after 8
  iterations, whichever comes first, and logs every step (tool, arguments, result or error,
  tokens, latency) to a trace.
- `evals/tasks.jsonl` has 30 scripted requests: 10 simple bookings, cancellations and listings,
  12 multi-step requests (find a slot across several people and time zones, then book it), and
  8 that are impossible under the rules (over capacity, no shared working hours, a clash with an
  existing meeting) where the correct behaviour is to explain and book nothing. Each task defines
  its expected end state as a check on the calendar, not an expected transcript, and every task
  runs against a fresh clone of the seed state.
- `evals/run.py` runs every task, classifies any failure (`wrong_end_state`,
  `invalid_arguments_unrecovered`, `iteration_cap`, `booked_when_impossible`, `model_error`),
  prints the results table and writes `results/<date>.json`.

## Limits

- The 8 "impossible" tasks are hand-picked to be unambiguous (over capacity, zero time zone
  overlap, an exact clash with an existing meeting). A request with a plausible alternative the
  model should propose instead is out of scope here.
- `find_free_slots` walks a fixed 15-minute grid; it does not do anything smarter than checking
  attendee overlap and working hours at each grid point.
- The failure classifier in `evals/run.py` is a set of heuristics on the trace (iteration cap,
  whether any tool call was recovered from, task category), not a certainty about why the model
  did what it did.
- Only a 3-task smoke run against `CHEAP_MODEL` has been run so far. The full 30-task eval is
  implemented but has not been run.
