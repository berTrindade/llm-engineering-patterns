from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from evals.checks import check_end_state
from tool_calling_poc.agent import run_agent_loop
from tool_calling_poc.config import load_config
from tool_calling_poc.llm import chat_completion
from tool_calling_poc.state import OfficeState

TASKS_PATH = Path(__file__).resolve().parent / "tasks.jsonl"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "office.json"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_tasks(path: str | Path) -> list[dict]:
    tasks = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


@dataclass
class TaskOutcome:
    task_id: str
    category: str
    success: bool
    failure_type: str | None
    reason: str | None
    tool_call_count: int
    iterations: int
    recovered_error_count: int
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    final_message: str | None
    stopped_reason: str | None
    trace: list[dict]


def classify_failure(
    stopped_reason: str, recovered_error_count: int, category: str, end_state_ok: bool
) -> str | None:
    if end_state_ok:
        return None
    if stopped_reason == "iteration_cap":
        return "iteration_cap"
    if category == "impossible":
        return "booked_when_impossible"
    if recovered_error_count > 0:
        return "invalid_arguments_unrecovered"
    return "wrong_end_state"


def _trace_to_dicts(trace: list) -> list[dict]:
    return [
        {
            "tool_name": step.tool_name,
            "arguments": step.arguments,
            "result_or_error": step.result_or_error,
            "is_error": step.is_error,
        }
        for step in trace
    ]


def run_task(
    task: dict, seed_state: OfficeState, chat_fn, model: str, max_iterations: int = 8
) -> TaskOutcome:
    state = seed_state.clone()
    original_ids = set(state.meetings)

    try:
        result = run_agent_loop(
            state, task["request"], chat_fn, model=model, max_iterations=max_iterations
        )
    except Exception as exc:
        return TaskOutcome(
            task_id=task["id"],
            category=task["category"],
            success=False,
            failure_type="model_error",
            reason=f"{type(exc).__name__}: {exc}",
            tool_call_count=0,
            iterations=0,
            recovered_error_count=0,
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=0.0,
            final_message=None,
            stopped_reason=None,
            trace=[],
        )

    ok, reason = check_end_state(state, original_ids, task["checks"])
    failure_type = classify_failure(
        result.stopped_reason, result.recovered_error_count, task["category"], ok
    )

    return TaskOutcome(
        task_id=task["id"],
        category=task["category"],
        success=ok,
        failure_type=failure_type,
        reason=reason,
        tool_call_count=result.tool_call_count,
        iterations=result.iterations,
        recovered_error_count=result.recovered_error_count,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        latency_ms=result.latency_ms,
        final_message=result.final_message,
        stopped_reason=result.stopped_reason,
        trace=_trace_to_dicts(result.trace),
    )


def build_report(outcomes: list[TaskOutcome]) -> dict:
    n = len(outcomes)
    success_count = sum(1 for o in outcomes if o.success)

    by_category: dict[str, dict] = {}
    for category in sorted({o.category for o in outcomes}):
        cat_outcomes = [o for o in outcomes if o.category == category]
        cat_success = sum(1 for o in cat_outcomes if o.success)
        by_category[category] = {
            "total": len(cat_outcomes),
            "success": cat_success,
            "success_rate": cat_success / len(cat_outcomes) if cat_outcomes else 0.0,
        }

    failures_by_type = Counter(o.failure_type for o in outcomes if o.failure_type)

    return {
        "task_count": n,
        "success_count": success_count,
        "success_rate": success_count / n if n else 0.0,
        "by_category": by_category,
        "mean_tool_calls": sum(o.tool_call_count for o in outcomes) / n if n else 0.0,
        "mean_iterations": sum(o.iterations for o in outcomes) / n if n else 0.0,
        "recovered_error_count": sum(o.recovered_error_count for o in outcomes),
        "failures_by_type": dict(failures_by_type),
        "mean_latency_ms": sum(o.latency_ms for o in outcomes) / n if n else 0.0,
        "mean_prompt_tokens": sum(o.prompt_tokens for o in outcomes) / n if n else 0.0,
        "mean_completion_tokens": sum(o.completion_tokens for o in outcomes) / n if n else 0.0,
        "tasks": [asdict(o) for o in outcomes],
    }


def print_table(report: dict) -> None:
    print(
        f"Task success rate: {report['success_count']}/{report['task_count']} "
        f"({report['success_rate']:.0%})"
    )
    print()
    print("Success by category:")
    for category, stats in report["by_category"].items():
        print(f"  {category:12s} {stats['success']}/{stats['total']} ({stats['success_rate']:.0%})")
    print()
    print(f"Mean tool calls per task: {report['mean_tool_calls']:.2f}")
    print(f"Mean iterations per task: {report['mean_iterations']:.2f}")
    print(f"Recovered-from-error count: {report['recovered_error_count']}")
    print()
    print("Failures by type:")
    if report["failures_by_type"]:
        for failure_type, count in sorted(report["failures_by_type"].items()):
            print(f"  {failure_type}: {count}")
    else:
        print("  none")
    print()
    print(f"Mean latency per task: {report['mean_latency_ms']:.0f} ms")
    print(f"Mean prompt tokens per task: {report['mean_prompt_tokens']:.0f}")
    print(f"Mean completion tokens per task: {report['mean_completion_tokens']:.0f}")


def main() -> None:
    config = load_config()
    tasks = load_tasks(TASKS_PATH)

    limit = os.environ.get("EVAL_LIMIT")
    is_smoke = limit is not None
    if is_smoke:
        tasks = tasks[: int(limit)]

    seed_state = OfficeState.from_json(DATA_PATH)
    model = config.cheap_model

    def chat_fn(**kwargs):
        return chat_completion(base_url=config.llm_base_url, api_key=config.llm_api_key, **kwargs)

    RESULTS_DIR.mkdir(exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    filename = f"smoke-{date_str}.json" if is_smoke else f"{date_str}.json"
    out_path = RESULTS_DIR / filename

    outcomes: list[TaskOutcome] = []
    for task in tasks:
        outcomes.append(run_task(task, seed_state, chat_fn, model))
        # Written after every task, so a run that stalls or is killed keeps what it finished.
        out_path.write_text(json.dumps({**build_report(outcomes), "complete": False}, indent=2))

    report = {**build_report(outcomes), "complete": True}
    print_table(report)

    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
