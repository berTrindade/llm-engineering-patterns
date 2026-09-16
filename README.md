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

Full run on 2026-09-16: 30 tasks, `qwen3:8b` through local Ollama, with no model call
failures. Raw output is in [results/2026-09-16.json](results/2026-09-16.json).

| Category | Succeeded |
|---|---|
| simple | 9/10 |
| multi_step | 9/12 |
| impossible | 8/8 |
| **total** | **26/30** |

A task that ran used 2.3 tool calls and 3.3 loop iterations on average, with 10,988 prompt
and 228 completion tokens and a median of 13 s end to end.

- **Refusing is solid.** All 8 impossible requests (over capacity, no shared working hours,
  a clash) ended with nothing booked. In 4 of them the model tried a booking first, got the
  domain error back as the tool result, and then explained why it could not book, which is
  the recovery path the loop is built for.
- **Most failures are the wrong person.** The system prompt lists every person with their
  id, yet in three tasks the model booked someone else: Isla Fraser (p09) became p10 or p12,
  and Erin Walsh (p05) became p11. Each final message still named the right person (in two,
  next to the wrong id), so the reply reads as correct while the calendar is wrong.
- **An error the model routed around instead of fixing.** In m05 the booking was rejected
  because the wrong attendee (p11, in New York) was outside working hours at 09:30 UTC. The
  model kept the wrong attendee, found a later slot, booked 13:00 instead of the requested
  09:30 and reported success without mentioning the change.
- **One task stopped early.** In m02 the model found slots and rooms, then replied "Let's
  proceed with the booking" as plain text, which ends the loop, and never booked.

Two task requests were corrected before this run: as first written, m07 and m10 each had
more than one valid room while the check accepted only one, so a correct booking scored as
a failure (see the git history). The failure types printed by `make eval` come from
heuristics on the trace (see Limits); m05, for example, is labelled
`invalid_arguments_unrecovered` although its root cause is the wrong id.

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
- The office has no notion of weekends and "next week" is not defined anywhere, so the
  model is free to book on a Saturday or Sunday, and in this run several bookings landed on
  19 and 20 September 2026. The checks do not look at the date, so this does not affect the
  success rate.
