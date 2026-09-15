"""Runs the cost-and-routing eval: three strategies over the question set, a threshold
sweep for routing, a confidence-calibration check, and a thinking-tokens comparison.

Usage:
    uv run python -m evals.run            full eval, writes results/<date>.json
    uv run python -m evals.run --demo     one question through all three strategies
    EVAL_LIMIT=3 uv run python -m evals.run   smoke run, writes results/smoke-<date>.json
"""

import json
import os
import sys
from datetime import date

from cost_and_routing.llm import chat_completion
from cost_and_routing.pricing import cost_usd, load_prices
from cost_and_routing.routing import SYSTEM_PROMPT
from cost_and_routing.strategies import CaseResult, run_always_cheap, run_always_strong, run_routed
from evals.cases import load_questions

THRESHOLDS = [0.5, 0.7, 0.9]
CALIBRATION_THRESHOLD = 0.7


def aggregate(results: list[CaseResult]) -> dict:
    groups: dict[str, list[CaseResult]] = {}
    for r in results:
        key = f"{r.strategy}|{r.threshold}"
        groups.setdefault(key, []).append(r)

    stats = {}
    for key, group in groups.items():
        n = len(group)
        correct_n = sum(1 for r in group if r.correct)
        total_cost = sum(r.cost_usd for r in group)
        escalated_n = sum(1 for r in group if r.escalated)
        stats[key] = {
            "strategy": group[0].strategy,
            "threshold": group[0].threshold,
            "accuracy": correct_n / n if n else 0.0,
            "escalation_rate": escalated_n / n if n else 0.0,
            "total_cost_usd": total_cost,
            "cost_per_correct": (total_cost / correct_n) if correct_n else None,
            "mean_latency_ms": sum(r.latency_ms for r in group) / n if n else 0.0,
            "mean_tokens": sum(r.prompt_tokens + r.completion_tokens for r in group) / n
            if n
            else 0.0,
        }
    return stats


_BUCKETS = [
    ("0.0-0.5", 0.0, 0.5),
    ("0.5-0.7", 0.5, 0.7),
    ("0.7-0.9", 0.7, 0.9),
    ("0.9-1.0", 0.9, 1.01),  # inclusive of 1.0
]


def calibration_table(cheap_results: list[CaseResult]) -> dict[str, float | None]:
    buckets: dict[str, list[CaseResult]] = {name: [] for name, _, _ in _BUCKETS}
    for r in cheap_results:
        if r.confidence is None:
            continue
        for name, lo, hi in _BUCKETS:
            if lo <= r.confidence < hi:
                buckets[name].append(r)
                break

    return {
        name: (sum(1 for r in group if r.correct) / len(group) if group else None)
        for name, group in buckets.items()
    }


def thinking_tokens_check(question: str, model: str) -> dict:
    with_thinking = chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        model=model,
        reasoning_effort=None,
    )
    without_thinking = chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        model=model,
        reasoning_effort="none",
    )
    prices = load_prices()
    cheap_price = prices["cheap"]
    cost_with = cost_usd(with_thinking.prompt_tokens, with_thinking.completion_tokens, cheap_price)
    cost_without = cost_usd(
        without_thinking.prompt_tokens, without_thinking.completion_tokens, cheap_price
    )
    return {
        "question": question,
        "completion_tokens_reasoning_effort_omitted": with_thinking.completion_tokens,
        "completion_tokens_reasoning_effort_none": without_thinking.completion_tokens,
        "cost_usd_reasoning_effort_omitted": cost_with,
        "cost_usd_reasoning_effort_none": cost_without,
        "cost_difference_usd": cost_with - cost_without,
    }


def format_table(aggregates: dict) -> str:
    header = (
        f"{'strategy':<14}{'threshold':<11}{'accuracy':<10}{'escalation':<12}"
        f"{'cost_usd':<11}{'cost/correct':<14}{'latency_ms':<12}{'tokens':<8}"
    )
    lines = [header, "-" * len(header)]
    for stats in aggregates.values():
        threshold = "-" if stats["threshold"] is None else f"{stats['threshold']:.1f}"
        cost_per_correct = (
            "-" if stats["cost_per_correct"] is None else f"{stats['cost_per_correct']:.5f}"
        )
        lines.append(
            f"{stats['strategy']:<14}{threshold:<11}{stats['accuracy']:<10.2f}"
            f"{stats['escalation_rate']:<12.2f}{stats['total_cost_usd']:<11.5f}"
            f"{cost_per_correct:<14}{stats['mean_latency_ms']:<12.0f}{stats['mean_tokens']:<8.0f}"
        )
    return "\n".join(lines)


def _prices_by_model(cheap_model: str, strong_model: str) -> dict:
    role_prices = load_prices()
    return {cheap_model: role_prices["cheap"], strong_model: role_prices["strong"]}


def _run_full(limit: int | None) -> dict:
    cheap_model = os.environ.get("CHEAP_MODEL", "qwen3:8b")
    strong_model = os.environ.get("STRONG_MODEL", "qwen3.5:35b")
    prices = _prices_by_model(cheap_model, strong_model)
    questions = load_questions(limit=limit)

    all_results: list[CaseResult] = []
    all_results += run_always_cheap(questions, cheap_model, prices)
    all_results += run_always_strong(questions, strong_model, prices)

    routed_by_threshold: dict[float, list[CaseResult]] = {}
    for threshold in THRESHOLDS:
        routed_results = run_routed(questions, cheap_model, strong_model, threshold, prices)
        routed_by_threshold[threshold] = routed_results
        all_results += routed_results

    aggregates = aggregate(all_results)
    calibration = calibration_table(routed_by_threshold[CALIBRATION_THRESHOLD])
    thinking_check = thinking_tokens_check(questions[0].question, cheap_model)

    return {
        "date": date.today().isoformat(),
        "cheap_model": cheap_model,
        "strong_model": strong_model,
        "question_count": len(questions),
        "aggregates": aggregates,
        "calibration": calibration,
        "thinking_tokens_check": thinking_check,
        "cases": [vars(r) for r in all_results],
    }


def _run_demo() -> None:
    cheap_model = os.environ.get("CHEAP_MODEL", "qwen3:8b")
    strong_model = os.environ.get("STRONG_MODEL", "qwen3.5:35b")
    prices = _prices_by_model(cheap_model, strong_model)
    question = load_questions(limit=1)[0]

    print(f"Question: {question.question}\n")
    for name, runner in [
        ("always_cheap", lambda: run_always_cheap([question], cheap_model, prices)),
        ("always_strong", lambda: run_always_strong([question], strong_model, prices)),
        ("routed", lambda: run_routed([question], cheap_model, strong_model, 0.7, prices)),
    ]:
        result = runner()[0]
        print(f"[{name}] correct={result.correct} cost=${result.cost_usd:.5f}")


def main() -> None:
    if "--demo" in sys.argv:
        _run_demo()
        return

    limit_str = os.environ.get("EVAL_LIMIT")
    limit = int(limit_str) if limit_str else None

    report = _run_full(limit)
    print(format_table(report["aggregates"]))
    print("\nConfidence calibration (cheap model, accuracy by confidence bucket):")
    for bucket, acc in report["calibration"].items():
        print(f"  {bucket}: {'-' if acc is None else f'{acc:.2f}'}")
    print("\nThinking tokens check:")
    print(json.dumps(report["thinking_tokens_check"], indent=2))

    os.makedirs("results", exist_ok=True)
    filename = f"smoke-{report['date']}.json" if limit else f"{report['date']}.json"
    out_path = os.path.join("results", filename)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
