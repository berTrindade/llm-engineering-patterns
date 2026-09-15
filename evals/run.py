"""Runs the three extraction approaches over the invoice cases, scores them, prints a table
and writes results/<date>.json (or results/smoke-<date>.json when EVAL_LIMIT is set).
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from structured_outputs.extract import extract_json_mode, extract_json_schema, extract_prompt_only
from structured_outputs.llm import call_llm
from structured_outputs.schema import Invoice
from structured_outputs.scoring import aggregate

REPO_ROOT = Path(__file__).resolve().parents[1]

APPROACHES = (
    ("prompt_only", extract_prompt_only),
    ("json_mode", extract_json_mode),
    ("json_schema", extract_json_schema),
)

TABLE_HEADERS = (
    "approach",
    "valid first try",
    "valid after repair",
    "field accuracy",
    "mean latency ms",
    "mean prompt tokens",
    "mean completion tokens",
)


def load_cases(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def run_pipeline(
    cases: list[dict], model: str, limit: int | None = None, call_fn=call_llm
) -> dict:
    if limit is not None:
        cases = cases[:limit]

    ground_truths = {case["case_id"]: Invoice.model_validate(case["invoice"]) for case in cases}

    results = []
    for _approach_name, extract_fn in APPROACHES:
        for case in cases:
            results.append(
                extract_fn(case["case_id"], case["email_text"], model, call_fn=call_fn)
            )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": model,
        "n_cases": len(cases),
        "approaches": aggregate(results, ground_truths),
    }


def print_table(results: dict) -> None:
    print(" | ".join(TABLE_HEADERS))
    for approach, stats in results["approaches"].items():
        print(
            f"{approach} | "
            f"{stats['valid_first_try_rate']:.2f} | "
            f"{stats['valid_after_repair_rate']:.2f} | "
            f"{stats['field_accuracy']:.2f} | "
            f"{stats['mean_latency_ms']:.0f} | "
            f"{stats['mean_prompt_tokens']:.0f} | "
            f"{stats['mean_completion_tokens']:.0f}"
        )
    for approach, stats in results["approaches"].items():
        if stats["failing_case_ids"]:
            print(f"\nfailing cases for {approach}:")
            for failure in stats["failing_case_ids"]:
                print(f"  {failure['case_id']}: {failure['reason']}")


def main() -> None:
    limit_env = os.environ.get("EVAL_LIMIT")
    limit = int(limit_env) if limit_env else None
    model = os.environ.get("CHEAP_MODEL", "qwen3:8b")

    cases = load_cases(REPO_ROOT / "data" / "cases.jsonl")
    results = run_pipeline(cases, model=model, limit=limit)
    print_table(results)

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    filename = f"smoke-{date_str}.json" if limit is not None else f"{date_str}.json"
    (results_dir / filename).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {results_dir / filename}")


if __name__ == "__main__":
    main()
