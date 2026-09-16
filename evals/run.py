"""Runs the three extraction approaches over the invoice cases, scores them, prints a table
and writes results/<date>.json (or results/smoke-<date>.json when EVAL_LIMIT is set).
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from structured_outputs.extract import (
    ExtractionResult,
    extract_json_mode,
    extract_json_schema,
    extract_prompt_only,
)
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
    "failures",
)


def load_cases(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def run_pipeline(
    cases: list[dict],
    model: str,
    limit: int | None = None,
    call_fn=call_llm,
    out_path: Path | None = None,
) -> dict:
    if limit is not None:
        cases = cases[:limit]

    ground_truths = {case["case_id"]: Invoice.model_validate(case["invoice"]) for case in cases}

    def report(complete: bool) -> dict:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "model": model,
            "n_cases": len(cases),
            "complete": complete,
            "approaches": aggregate(results, ground_truths),
        }

    results = []
    for approach_name, extract_fn in APPROACHES:
        for case in cases:
            try:
                result = extract_fn(case["case_id"], case["email_text"], model, call_fn=call_fn)
            except Exception as exc:  # noqa: BLE001 - recorded as a failed case, never swallowed
                result = ExtractionResult(
                    case_id=case["case_id"],
                    approach=approach_name,
                    valid_first_try=False,
                    valid_after_repair=False,
                    invoice=None,
                    error=f"{type(exc).__name__}: {exc}",
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=0.0,
                    failed=True,
                )
            results.append(result)
            # Written after every case, so a run that dies or is killed keeps what it finished.
            if out_path is not None:
                out_path.write_text(json.dumps(report(complete=False), indent=2))

    return report(complete=True)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def print_table(results: dict) -> None:
    print(" | ".join(TABLE_HEADERS))
    for approach, stats in results["approaches"].items():
        print(
            f"{approach} | "
            f"{stats['valid_first_try_rate']:.2f} | "
            f"{stats['valid_after_repair_rate']:.2f} | "
            f"{stats['field_accuracy']:.2f} | "
            f"{_fmt(stats['mean_latency_ms'])} | "
            f"{_fmt(stats['mean_prompt_tokens'])} | "
            f"{_fmt(stats['mean_completion_tokens'])} | "
            f"{stats['failures']}"
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

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    filename = f"smoke-{date_str}.json" if limit is not None else f"{date_str}.json"

    cases = load_cases(REPO_ROOT / "data" / "cases.jsonl")
    results = run_pipeline(cases, model=model, limit=limit, out_path=results_dir / filename)
    print_table(results)

    (results_dir / filename).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {results_dir / filename}")


if __name__ == "__main__":
    main()
