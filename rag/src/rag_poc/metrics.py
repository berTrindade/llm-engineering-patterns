"""Score a list of eval cases into the headline metrics for results/*.json."""

from pydantic import BaseModel


class CaseResult(BaseModel):
    id: str
    answerable: bool
    hit_at_5: bool
    citation_correct_share: float | None  # answerable only
    refused: bool
    correct: bool  # answerable: not falsely refused; unanswerable: correctly refused
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    failed: bool
    failure_reason: str | None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def compute_metrics(results: list[CaseResult]) -> dict:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]

    hit_rate_at_5 = _mean([1.0 if r.hit_at_5 else 0.0 for r in answerable])
    citation_correctness = _mean(
        [r.citation_correct_share for r in answerable if r.citation_correct_share is not None]
    )
    correct_refusal_rate = _mean([1.0 if r.correct else 0.0 for r in unanswerable])
    false_refusal_rate = _mean([1.0 if r.refused and not r.correct else 0.0 for r in answerable])
    mean_latency_ms = _mean([r.latency_ms for r in results])
    mean_tokens = _mean([r.prompt_tokens + r.completion_tokens for r in results])
    failing_ids = [{"id": r.id, "reason": r.failure_reason} for r in results if r.failed]

    return {
        "hit_rate_at_5": hit_rate_at_5,
        "citation_correctness": citation_correctness,
        "correct_refusal_rate": correct_refusal_rate,
        "false_refusal_rate": false_refusal_rate,
        "mean_latency_ms": mean_latency_ms,
        "mean_tokens": mean_tokens,
        "failing_ids": failing_ids,
    }
