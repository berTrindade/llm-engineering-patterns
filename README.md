# human-in-the-loop-poc

Proof of concept: an agent proposes an action on an expense claim or refund request, and the
proposal always pauses for a human decision before anything happens. The pause is stored in
SQLite, so it survives a crash, and every decision is written to an append-only audit log.

## Problem

An agent that reads a policy and proposes approve/reject/refund decisions is useful, but nobody
should let it act unsupervised on money. The standard fix, a human approval step, is easy to get
wrong in a way that only shows up in production: the process is killed between "the agent
proposed something" and "a person decided", or between "a person approved it" and "the action
was actually applied", and the request is either lost or applied twice. This repository builds
the smallest version of that approval workflow that survives both failure points, using nothing
but SQLite and the standard library, and proves it with a test that kills the process mid-flow
and restarts it.

Domain: expense claims and customer refund requests for Tallow & Finch, an invented homeware
retailer. The policy in [data/policy.md](data/policy.md) and every request in
[data/](data/README.md) are invented for this proof of concept.

## Run it

Requires [uv](https://docs.astral.sh/uv/) and a local [Ollama](https://ollama.com) with
`qwen3:8b` pulled.

```bash
make setup   # uv sync
make run     # submit sample requests, propose, list what needs a decision
make test    # unit tests and the crash/resume test, deterministic, no model
make eval    # full proposal accuracy assessment against evals/requests.jsonl
```

`make run` stops at `pending`, since it does not approve or reject anything: that decision is the
point of the exercise. Continue the workflow by hand:

```bash
uv run python -m hitl approve exp-001 --by "Priya Shah"
uv run python -m hitl reject exp-002 --by "Priya Shah" --reason "no receipt over £25"
uv run python -m hitl resume
uv run python -m hitl audit exp-001
```

## Result

Not yet run in full. Run `make eval`.

The crash and resume test (`uv run pytest tests/test_crash_resume.py`) needs no model and runs in
CI: it submits 5 requests, kills `python -m hitl process` with SIGKILL right after the whole
batch reaches `AWAITING_APPROVAL`, approves 3 and rejects 2 directly against the database, kills
`python -m hitl resume` with SIGKILL right before it executes anything, then restarts and runs
`resume` twice. It asserts no request is lost, nothing is executed twice, and every request's
audit log is complete and ordered.

## How it works

- **State machine** (`src/hitl/db.py`): every request moves through
  `RECEIVED -> PROPOSED -> AWAITING_APPROVAL -> APPROVED|REJECTED -> EXECUTED|FAILED`. Each move
  is one SQLite transaction that updates the request row and appends one row to `audit_log`. A
  disallowed move (e.g. skipping straight to `EXECUTED`) raises `IllegalTransitionError`.
- **Exactly-once execution**: `execute_request()` inserts a `ledger` row and flips the request to
  `EXECUTED` in a single transaction. `ledger.request_id` has a `UNIQUE` constraint, so if the
  process crashes mid-execution or `resume` runs twice, the second attempt hits the constraint
  and is treated as `"already_executed"` rather than writing a second ledger row.
  `journal_mode=WAL` means a committed transaction survives a SIGKILL of the process that wrote
  it.
- **The agent** (`src/hitl/agent.py`): reads the request and `data/policy.md`, calls
  `CHEAP_MODEL` with a JSON-schema `response_format` built from the `Proposal` pydantic model,
  and returns `{action, amount, policy_clauses, rationale}`. If the reply does not validate
  against the schema, the request is moved to `FAILED` with the validation error as the reason.
  The agent never guesses.
- **CLI** (`python -m hitl ...`): `submit`, `process` (propose and stop at
  `AWAITING_APPROVAL`), `pending`, `approve --by NAME`, `reject --by NAME --reason TEXT`,
  `resume` (executes every `APPROVED` request not yet executed), `audit <id>`.
- **LLM client** (`src/hitl/llm.py`): a plain `httpx` POST to
  `{LLM_BASE_URL}/chat/completions` with `reasoning_effort: "none"`. Verified locally on
  2026-09-15 against Ollama 0.34.0 that this stops `qwen3` "thinking"; a `/no_think` prompt
  suffix alone was not enough (a 20-token JSON answer reported 542 completion tokens without this
  field). An `LLM_FIXTURE` env var swaps in a recorded response instead of calling the network,
  which is what makes the crash test and unit tests deterministic and model-free.

## Limits

- The state machine, the schema, and the policy are all invented for this proof of concept and
  have not been reviewed by anyone who runs a real approvals process.
- `resume` retries every `APPROVED` request every time it runs; there is no backoff or retry
  limit for a request that keeps failing at execution.
- The crash test kills the process at two specific points chosen to match the two failure modes
  named in the design spec. It does not fuzz arbitrary kill points, and it does not simulate a
  crash mid-`transition()` (SQLite's own transaction atomicity is what protects that path, not
  application code, so there is nothing extra to test there).
- The proposal accuracy figure in `results/` only covers the accuracy of the agent's proposed
  `action` against a policy-derived expected answer; it says nothing about how good the
  `rationale` text is.
