"""Proposal assessment: run the agent on evals/requests.jsonl and score it against the
policy-derived expected action for each case.

`EVAL_LIMIT` (int env var) caps how many cases run and switches the results filename to
`results/smoke-<date>.json`, use it to smoke-test against a shared Ollama instance instead of
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


def run_assessment(cases_path: str | Path = DEFAULT_CASES_PATH, limit: int | None = None) -> dict:
    cases = _load_cases(Path(cases_path))
    if limit is not None:
        cases = cases[:limit]

    policy_text = DEFAULT_POLICY_PATH.read_text()
    model = os.environ.get("CHEAP_MODEL", "qwen3:8b")

    per_case = []
    for case in cases:
        expected = case["expected_action"]
        request = {k: v for k, v in case.items() if k != "expected_action"}
        result = agent.propose(request, policy_text, model)

        if not result.ok:
            per_case.append({
                "id": case["id"],
                "expected_action": expected,
                "actual_action": None,
                "correct": False,
                "invalid": True,
                "error": result.error,
                "latency_ms": result.latency_ms,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            })
            continue

        actual = result.proposal.action
        per_case.append({
            "id": case["id"],
            "expected_action": expected,
            "actual_action": actual,
            "correct": actual == expected,
            "invalid": False,
            "error": None,
            "latency_ms": result.latency_ms,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        })

    n = len(per_case)
    invalid = [c for c in per_case if c["invalid"]]
    valid = [c for c in per_case if not c["invalid"]]
    correct = [c for c in valid if c["correct"]]

    summary = {
        "cases_run": n,
        "proposal_accuracy": (len(correct) / n) if n else 0.0,
        "invalid_output_rate": (len(invalid) / n) if n else 0.0,
        "mean_latency_ms": statistics.mean(c["latency_ms"] for c in per_case) if per_case else 0.0,
        "mean_total_tokens": statistics.mean(c["total_tokens"] for c in per_case) if per_case else 0.0,
    }

    return {"summary": summary, "cases": per_case}


def _print_table(summary: dict) -> None:
    print(f"{'metric':<24}{'value'}")
    print(f"{'cases_run':<24}{summary['cases_run']}")
    print(f"{'proposal_accuracy':<24}{summary['proposal_accuracy']:.2f}")
    print(f"{'invalid_output_rate':<24}{summary['invalid_output_rate']:.2f}")
    print(f"{'mean_latency_ms':<24}{summary['mean_latency_ms']:.1f}")
    print(f"{'mean_total_tokens':<24}{summary['mean_total_tokens']:.1f}")


def main() -> int:
    limit_env = os.environ.get("EVAL_LIMIT")
    limit = int(limit_env) if limit_env else None

    results = run_assessment(limit=limit)
    _print_table(results["summary"])

    RESULTS_DIR.mkdir(exist_ok=True)
    today = date.today().isoformat()
    filename = f"smoke-{today}.json" if limit is not None else f"{today}.json"
    out_path = RESULTS_DIR / filename
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
