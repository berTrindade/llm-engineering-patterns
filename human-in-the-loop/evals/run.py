"""Proposal assessment: run the agent on evals/requests.jsonl and score it against the
policy-derived expected action for each case.

`EVAL_LIMIT` (int env var) caps how many cases run and switches the results filename to
`results/smoke-<date>.json`, so it can smoke-test against a shared Ollama instance instead of
running all 20 cases.
"""

from __future__ import annotations

import json
import os
import statistics
from datetime import date
from pathlib import Path

from hitl import agent

ROOT = Path(__file__).parent.parent
DEFAULT_CASES_PATH = ROOT / "evals" / "requests.jsonl"
DEFAULT_POLICY_PATH = ROOT / "data" / "policy.md"
RESULTS_DIR = ROOT / "results"


def _load_cases(cases_path: Path) -> list[dict]:
    with open(cases_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _assess_case(case: dict, policy_text: str, model: str) -> dict:
    expected = case["expected_action"]
    request = {k: v for k, v in case.items() if k != "expected_action"}
    try:
        result = agent.propose(request, policy_text, model)
    except Exception as exc:  # noqa: BLE001 - recorded as a failed case, never swallowed
        return {
            "id": case["id"],
            "expected_action": expected,
            "actual_action": None,
            "correct": False,
            "invalid": False,
            "failed": True,
            "error": f"{type(exc).__name__}: {exc}",
            "latency_ms": None,
            "total_tokens": None,
        }

    if not result.ok:
        return {
            "id": case["id"],
            "expected_action": expected,
            "actual_action": None,
            "correct": False,
            "invalid": True,
            "failed": False,
            "error": result.error,
            "latency_ms": result.latency_ms,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        }

    actual = result.proposal.action
    return {
        "id": case["id"],
        "expected_action": expected,
        "actual_action": actual,
        "correct": actual == expected,
        "invalid": False,
        "failed": False,
        "error": None,
        "latency_ms": result.latency_ms,
        "total_tokens": result.prompt_tokens + result.completion_tokens,
    }


def _summarise(per_case: list[dict]) -> dict:
    n = len(per_case)
    invalid = [c for c in per_case if c["invalid"]]
    valid = [c for c in per_case if not c["invalid"]]
    correct = [c for c in valid if c["correct"]]
    # Failed cases count against accuracy, but have no latency or tokens to average.
    measured = [c for c in per_case if not c["failed"]]

    return {
        "cases_run": n,
        "failures": n - len(measured),
        "proposal_accuracy": (len(correct) / n) if n else 0.0,
        "invalid_output_rate": (len(invalid) / n) if n else 0.0,
        "mean_latency_ms": (
            statistics.mean(c["latency_ms"] for c in measured) if measured else None
        ),
        "mean_total_tokens": (
            statistics.mean(c["total_tokens"] for c in measured) if measured else None
        ),
    }


def run_assessment(
    cases_path: str | Path = DEFAULT_CASES_PATH,
    limit: int | None = None,
    out_path: Path | None = None,
) -> dict:
    cases = _load_cases(Path(cases_path))
    if limit is not None:
        cases = cases[:limit]

    policy_text = DEFAULT_POLICY_PATH.read_text()
    model = os.environ.get("CHEAP_MODEL", "qwen3:8b")

    per_case = []
    for case in cases:
        per_case.append(_assess_case(case, policy_text, model))
        # Written after every case, so a run that dies or is killed keeps what it finished.
        if out_path is not None:
            partial = {"summary": _summarise(per_case), "cases": per_case, "complete": False}
            out_path.write_text(json.dumps(partial, indent=2))

    return {"summary": _summarise(per_case), "cases": per_case, "complete": True}


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def _print_table(summary: dict) -> None:
    print(f"{'metric':<24}{'value'}")
    print(f"{'cases_run':<24}{summary['cases_run']}")
    print(f"{'failures':<24}{summary['failures']}")
    print(f"{'proposal_accuracy':<24}{summary['proposal_accuracy']:.2f}")
    print(f"{'invalid_output_rate':<24}{summary['invalid_output_rate']:.2f}")
    print(f"{'mean_latency_ms':<24}{_fmt(summary['mean_latency_ms'])}")
    print(f"{'mean_total_tokens':<24}{_fmt(summary['mean_total_tokens'])}")


def main() -> int:
    limit_env = os.environ.get("EVAL_LIMIT")
    limit = int(limit_env) if limit_env else None

    RESULTS_DIR.mkdir(exist_ok=True)
    today = date.today().isoformat()
    filename = f"smoke-{today}.json" if limit is not None else f"{today}.json"
    out_path = RESULTS_DIR / filename

    results = run_assessment(limit=limit, out_path=out_path)
    _print_table(results["summary"])

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
