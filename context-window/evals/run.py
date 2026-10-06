"""Eval harness: strategy x budget x thread, scored for fact recall.

Reads env vars EVAL_LIMIT (max threads, default all), EVAL_BUDGETS (comma-separated
token budgets, default "2000,4000,8000") and RESULTS_FILE (default
results/<YYYY-MM-DD>.json). Only CHEAP_MODEL is used, for both the final answer and
summarise_older's summaries.
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from context_window_poc.llm import chat_completion
from context_window_poc.scoring import fact_recall
from context_window_poc.strategies import (
    full_history,
    last_n,
    summarise_older,
    token_budget,
)
from context_window_poc.tokens import CalibrationFit, estimate_messages_tokens, load_calibration

SYSTEM_PROMPT = (
    "You are a Ledgerline support agent. Ledgerline is small-business invoicing "
    "software. Answer the customer's question directly and concisely, using only "
    "facts you can see in this conversation. If you are not sure of a fact, say so "
    "rather than guessing."
)

BUDGETED_STRATEGIES = ("last_n", "token_budget", "summarise_older")

# Full-history prompts here run to roughly 14,000-17,000 estimated tokens; prefill
# takes longer than the client default allows for, and Ollama serves one request at
# a time.
CHAT_CALL_TIMEOUT_S = 300.0

# If real prompt_tokens comes back below this fraction of the estimate, the server
# may have silently truncated the prompt rather than the estimator being off. qwen3:8b
# on this deployment has context_length 40960 (reported by the local Ollama server),
# well above our largest full-history prompt, so this should not trigger; it is
# recorded as a guard, not assumed away.
TRUNCATION_GUARD_FACTOR = 0.5


@dataclass
class CaseResult:
    thread_id: str
    strategy: str
    budget: int | None
    fact_recall: float | None
    prompt_tokens_real: int | None
    extra_tokens: int
    latency_ms: float | None
    estimator_error_pct: float | None
    possible_truncation: bool | None
    failed: bool
    reason: str | None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def aggregate(results: list[CaseResult]) -> list[dict]:
    groups: dict[tuple[str, int | None], list[CaseResult]] = {}
    for r in results:
        groups.setdefault((r.strategy, r.budget), []).append(r)

    rows = []
    for (strategy, budget), group in groups.items():
        recalls = [r.fact_recall if not r.failed else 0.0 for r in group]
        successes = [r for r in group if not r.failed]
        rows.append(
            {
                "strategy": strategy,
                "budget": budget,
                "n": len(group),
                "failures": sum(1 for r in group if r.failed),
                "possible_truncations": sum(1 for r in group if r.possible_truncation),
                "fact_recall": _mean(recalls),
                "prompt_tokens_real": _mean([r.prompt_tokens_real for r in successes]),
                "extra_tokens": _mean([r.extra_tokens for r in group]),
                "latency_ms": _mean([r.latency_ms for r in successes]),
                "estimator_error_pct": _mean([r.estimator_error_pct for r in successes]),
            }
        )
    rows.sort(key=lambda r: (r["strategy"], r["budget"] if r["budget"] is not None else -1))
    return rows


def format_table(rows: list[dict]) -> str:
    headers = [
        "strategy",
        "budget",
        "fact_recall",
        "prompt_tokens",
        "extra_tokens",
        "latency_ms",
        "estimator_error_pct",
        "failures",
        "possible_truncations",
    ]
    lines = [" | ".join(headers)]
    for row in rows:
        lines.append(
            " | ".join(
                [
                    row["strategy"],
                    str(row["budget"]),
                    f"{row['fact_recall']:.2f}" if row["fact_recall"] is not None else "-",
                    f"{row['prompt_tokens_real']:.0f}" if row["prompt_tokens_real"] is not None else "-",
                    f"{row['extra_tokens']:.0f}" if row["extra_tokens"] is not None else "-",
                    f"{row['latency_ms']:.0f}" if row["latency_ms"] is not None else "-",
                    f"{row['estimator_error_pct']:.1f}" if row["estimator_error_pct"] is not None else "-",
                    str(row["failures"]),
                    str(row["possible_truncations"]),
                ]
            )
        )
    return "\n".join(lines)


def _run_case(
    thread: dict, strategy: str, budget: int | None, model: str, fit: CalibrationFit, cache: dict
) -> CaseResult:
    turns = thread["turns"]

    def estimator(messages: list[dict]) -> int:
        return estimate_messages_tokens(messages, fit)

    if strategy == "full_history":
        messages = full_history(SYSTEM_PROMPT, turns)
        extra_tokens = 0
    elif strategy == "last_n":
        messages = last_n(SYSTEM_PROMPT, turns, budget, estimator)
        extra_tokens = 0
    elif strategy == "token_budget":
        messages = token_budget(SYSTEM_PROMPT, turns, budget, estimator)
        extra_tokens = 0
    elif strategy == "summarise_older":

        def summarise_fn(older_text: str) -> tuple[str, int]:
            response = chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": "Summarise this support conversation in 2-3 sentences, "
                        "keeping any account IDs, plan names, error codes or time zones "
                        f"mentioned:\n\n{older_text}",
                    }
                ],
                model=model,
                timeout=CHAT_CALL_TIMEOUT_S,
            )
            return response.content, response.prompt_tokens + response.completion_tokens

        cache_key = f"{thread['thread_id']}:{budget}"
        messages = summarise_older(SYSTEM_PROMPT, turns, budget, estimator, summarise_fn, cache, cache_key)
        extra_tokens = cache.get(cache_key, {}).get("tokens", 0)
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    estimated = estimate_messages_tokens(messages, fit)

    try:
        response = chat_completion(messages=messages, model=model, timeout=CHAT_CALL_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        return CaseResult(
            thread_id=thread["thread_id"],
            strategy=strategy,
            budget=budget,
            fact_recall=None,
            prompt_tokens_real=None,
            extra_tokens=extra_tokens,
            latency_ms=None,
            estimator_error_pct=None,
            possible_truncation=None,
            failed=True,
            reason=str(exc),
        )

    recall = fact_recall(response.content, thread["expected_facts"])
    error_pct = (
        abs(estimated - response.prompt_tokens) / response.prompt_tokens * 100
        if response.prompt_tokens
        else None
    )
    possible_truncation = (
        response.prompt_tokens < estimated * TRUNCATION_GUARD_FACTOR if estimated else False
    )

    return CaseResult(
        thread_id=thread["thread_id"],
        strategy=strategy,
        budget=budget,
        fact_recall=recall,
        prompt_tokens_real=response.prompt_tokens,
        extra_tokens=extra_tokens,
        latency_ms=response.latency_ms,
        estimator_error_pct=error_pct,
        possible_truncation=possible_truncation,
        failed=False,
        reason=None,
    )


def load_completed(out_path: Path, resume: bool) -> tuple[list[CaseResult], set[str]]:
    """Cases already recorded in out_path, and the thread ids they cover.

    Returns nothing unless resume is set, so a plain run always starts clean rather than
    quietly building on an older results file.
    """
    if not (resume and out_path.exists()):
        return [], set()
    prior = json.loads(out_path.read_text())
    cases = [CaseResult(**case) for case in prior.get("cases", [])]
    return cases, {case.thread_id for case in cases}


def main() -> None:
    model = os.environ.get("CHEAP_MODEL", "qwen3:8b")
    limit = os.environ.get("EVAL_LIMIT")
    budgets = [int(b) for b in os.environ.get("EVAL_BUDGETS", "2000,4000,8000").split(",")]
    results_file = os.environ.get("RESULTS_FILE", f"results/{datetime.now(UTC).date().isoformat()}.json")

    fit = load_calibration()

    thread_paths = sorted(Path("data/threads").glob("*.json"))
    if limit:
        thread_paths = thread_paths[: int(limit)]
    threads = [json.loads(p.read_text()) for p in thread_paths]

    cache: dict = {}

    out_path = Path(results_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # EVAL_RESUME=1 keeps the cases already in results_file and runs only the threads
    # missing from it, so an interrupted run does not repeat finished work. Off by
    # default: a plain `make eval` should never quietly reuse an older run.
    results, done_ids = load_completed(out_path, os.environ.get("EVAL_RESUME") == "1")
    if done_ids:
        print(f"resuming: {len(done_ids)} threads already complete", flush=True)

    def write_results() -> None:
        out_path.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now(UTC).date().isoformat(),
                    "model": model,
                    "budgets": budgets,
                    "n_threads": len({r.thread_id for r in results}),
                    "estimator_fit": {"a": fit.a, "b": fit.b, "c": fit.c},
                    "rows": aggregate(results),
                    "cases": [asdict(r) for r in results],
                },
                indent=2,
            )
            + "\n"
        )

    for i, thread in enumerate(threads, 1):
        if thread["thread_id"] in done_ids:
            continue
        results.append(_run_case(thread, "full_history", None, model, fit, cache))
        for strategy in BUDGETED_STRATEGIES:
            for budget in budgets:
                results.append(_run_case(thread, strategy, budget, model, fit, cache))
        # Written after every thread, so a run that stalls or is killed keeps the cases
        # it already finished instead of discarding all of them.
        write_results()
        print(f"[{i}/{len(threads)}] {thread['thread_id']} done, {len(results)} cases", flush=True)

    rows = aggregate(results)
    table = format_table(rows)
    print(table)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
