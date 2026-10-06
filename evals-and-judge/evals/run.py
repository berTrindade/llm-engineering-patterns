"""Wires dataset, scorers, judge, position bias and human labels into one eval run."""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import statistics
from pathlib import Path

from evals_judge.dataset import DATA_PATH, load_dataset
from evals_judge.judge import JudgeVerdict, judge_case
from evals_judge.kappa import cohens_kappa
from evals_judge.labels import LABELS_PATH, load_labels
from evals_judge.position_bias import generate_pairs, run_position_bias
from evals_judge.scorers import SCORERS, ScorerResult

POLICY_PATH = DATA_PATH.parent / "policy.md"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_summary(
    cases: list,
    scorer_results: dict[str, list[ScorerResult]],
    judge_results: list[JudgeVerdict],
    human_labels: dict[str, dict],
    position_bias: dict,
) -> dict:
    n = len(cases)
    designed_labels = {c.id: c.designed_label for c in cases}

    per_scorer_agreement = {}
    for name, results in scorer_results.items():
        agree = sum(
            1
            for case, result in zip(cases, results, strict=True)
            if ("pass" if result.passed else "fail") == case.designed_label
        )
        per_scorer_agreement[name] = agree / n if n else 0.0

    judge_by_id = {v.case_id: v for v in judge_results}
    valid_judge = [v for v in judge_results if v.error is None]
    judge_errors = len(judge_results) - len(valid_judge)

    judge_vs_designed_hits = sum(
        1 for v in valid_judge if v.verdict == designed_labels.get(v.case_id)
    )
    judge_vs_designed_agreement = judge_vs_designed_hits / len(valid_judge) if valid_judge else 0.0

    human_ids = [
        cid
        for cid in human_labels
        if cid in judge_by_id and judge_by_id[cid].error is None
    ]
    if human_ids:
        judge_labels = [judge_by_id[cid].verdict for cid in human_ids]
        human_label_values = [human_labels[cid]["label"] for cid in human_ids]
        human_kappa = cohens_kappa(judge_labels, human_label_values)
        human_status = (
            f"Human labels: {len(human_labels)} of {n}, kappa vs judge = {human_kappa:.3f} "
            f"(n={len(human_ids)})"
        )
    else:
        human_kappa = None
        human_status = f"Human labels: {len(human_labels)} of {n}, run `make label`"

    mean_latency = statistics.mean(v.latency_ms for v in valid_judge) if valid_judge else 0.0
    mean_prompt_tokens = (
        statistics.mean(v.prompt_tokens for v in valid_judge) if valid_judge else 0.0
    )
    mean_completion_tokens = (
        statistics.mean(v.completion_tokens for v in valid_judge) if valid_judge else 0.0
    )

    return {
        "n_cases": n,
        "per_scorer_agreement": per_scorer_agreement,
        "judge_vs_designed_agreement": judge_vs_designed_agreement,
        "judge_errors": judge_errors,
        "judge_failures": sum(1 for v in judge_results if v.failed),
        "human_labelled_count": len(human_labels),
        "human_kappa": human_kappa,
        "human_status": human_status,
        "position_bias_flip_rate": position_bias.get("flip_rate", 0.0),
        "position_bias_first_position_win_share": position_bias.get(
            "first_position_win_share", 0.0
        ),
        "position_bias_failures": position_bias.get("failures", 0),
        "judge_mean_latency_ms": mean_latency,
        "judge_mean_prompt_tokens": mean_prompt_tokens,
        "judge_mean_completion_tokens": mean_completion_tokens,
    }


def format_table(summary: dict) -> str:
    lines = [f"Cases: {summary['n_cases']}", ""]
    lines.append("Deterministic scorer agreement with designed labels:")
    for name, agreement in summary["per_scorer_agreement"].items():
        lines.append(f"  {name:<24} {agreement:.1%}")
    lines.append("")
    lines.append(
        f"Judge vs designed labels agreement: {summary['judge_vs_designed_agreement']:.1%}"
        f" (judge errors: {summary['judge_errors']},"
        f" of which call failures: {summary['judge_failures']})"
    )
    lines.append("")
    lines.append(summary["human_status"])
    lines.append("")
    lines.append(f"Position bias flip rate: {summary['position_bias_flip_rate']:.1%}")
    lines.append(
        f"First-position win share: {summary['position_bias_first_position_win_share']:.1%}"
    )
    lines.append(f"Position bias call failures: {summary['position_bias_failures']}")
    lines.append("")
    lines.append(f"Judge mean latency: {summary['judge_mean_latency_ms']:.0f} ms")
    lines.append(f"Judge mean prompt tokens: {summary['judge_mean_prompt_tokens']:.1f}")
    lines.append(f"Judge mean completion tokens: {summary['judge_mean_completion_tokens']:.1f}")
    return "\n".join(lines)


def main() -> None:
    cases = load_dataset()
    policy_text = POLICY_PATH.read_text()

    limit = os.environ.get("EVAL_LIMIT")
    is_smoke = limit is not None
    if is_smoke:
        cases = cases[: int(limit)]

    judge_model = os.environ.get("JUDGE_MODEL") or os.environ.get("STRONG_MODEL", "qwen3.5:35b")

    scorer_results = {scorer.__name__: [scorer(c) for c in cases] for scorer in SCORERS}
    human_labels = load_labels(LABELS_PATH)

    RESULTS_DIR.mkdir(exist_ok=True)
    date_str = datetime.datetime.now(tz=datetime.UTC).date().isoformat()
    filename = f"smoke-{date_str}.json" if is_smoke else f"{date_str}.json"
    out_path = RESULTS_DIR / filename

    judge_results: list[JudgeVerdict] = []

    # Rewritten after every judge call and every position-bias pair, so a run that dies or
    # is killed keeps what it finished. complete stays false until the last write.
    def write_results(position_bias: dict, complete: bool = False) -> dict:
        summary = build_summary(cases, scorer_results, judge_results, human_labels, position_bias)
        output = {
            "summary": summary,
            "judge_model": judge_model,
            "n_cases": len(cases),
            "smoke": is_smoke,
            "complete": complete,
            "position_bias": position_bias,
            "judge_results": [dataclasses.asdict(v) for v in judge_results],
        }
        out_path.write_text(json.dumps(output, indent=2))
        return summary

    for case in cases:
        judge_results.append(judge_case(case, policy_text, judge_model))
        write_results({"records": []})

    pairs = generate_pairs(n=20)
    if is_smoke:
        pairs = pairs[: int(limit)]
    position_bias_result = run_position_bias(
        pairs,
        policy_text,
        judge_model,
        on_record=lambda records: write_results({"records": records}),
    )

    summary = write_results(position_bias_result, complete=True)
    print(format_table(summary))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
