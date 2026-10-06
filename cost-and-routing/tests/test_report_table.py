from cost_and_routing.strategies import CaseResult, run_always_cheap, run_routed
from evals.cases import Question
from evals.run import aggregate, calibration_table, format_table


def _case(
    strategy="always_cheap",
    threshold=None,
    correct=True,
    escalated=False,
    cost_usd=0.01,
    latency_ms=100.0,
    prompt_tokens=50,
    completion_tokens=10,
    confidence=None,
) -> CaseResult:
    return CaseResult(
        question_id="q",
        strategy=strategy,
        threshold=threshold,
        correct=correct,
        escalated=escalated,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        confidence=confidence,
        failure_reason=None,
    )


def test_aggregate_computes_accuracy_and_cost() -> None:
    results = [
        _case(correct=True, cost_usd=0.01, latency_ms=100, prompt_tokens=50, completion_tokens=10),
        _case(correct=False, cost_usd=0.02, latency_ms=200, prompt_tokens=60, completion_tokens=20),
    ]

    stats = aggregate(results)["always_cheap|None"]

    assert stats["accuracy"] == 0.5
    assert stats["total_cost_usd"] == 0.03
    assert stats["mean_latency_ms"] == 150.0
    assert stats["mean_tokens"] == (60 + 80) / 2
    assert stats["cost_per_correct"] == 0.03 / 1  # total cost over the 1 correct case


def test_aggregate_escalation_rate_for_routed() -> None:
    results = [
        _case(strategy="routed", threshold=0.7, escalated=True),
        _case(strategy="routed", threshold=0.7, escalated=False),
    ]

    stats = aggregate(results)["routed|0.7"]

    assert stats["escalation_rate"] == 0.5


def test_aggregate_cost_per_correct_guards_zero_correct() -> None:
    results = [_case(correct=False, cost_usd=0.05)]

    stats = aggregate(results)["always_cheap|None"]

    assert stats["cost_per_correct"] is None


def test_calibration_table_buckets_by_confidence() -> None:
    results = [
        _case(confidence=0.2, correct=False),
        _case(confidence=0.6, correct=True),
        _case(confidence=0.6, correct=False),
        _case(confidence=0.95, correct=True),
    ]

    buckets = calibration_table(results)

    assert buckets["0.5-0.7"] == 0.5
    assert buckets["0.9-1.0"] == 1.0


def test_model_error_is_recorded_as_failed_case_not_raised(monkeypatch) -> None:
    def broken(**kwargs):
        raise RuntimeError("connection reset")

    monkeypatch.setattr("cost_and_routing.strategies.chat_completion", broken)
    monkeypatch.setattr("cost_and_routing.routing.chat_completion", broken)
    question = Question(
        id="q1", question="Can I cancel?", required_facts=["24 hours"], difficulty="easy"
    )

    cheap = run_always_cheap([question], "cheap-model", {})
    routed = run_routed([question], "cheap-model", "strong-model", 0.7, {})

    for result in (*cheap, *routed):
        assert result.failed is True
        assert result.correct is False
        assert result.failure_reason == "RuntimeError: connection reset"

    stats = aggregate(cheap + [_case(correct=True)])["always_cheap|None"]
    assert stats["failures"] == 1
    assert stats["accuracy"] == 0.5
    assert stats["mean_latency_ms"] == 100.0
    assert "always_cheap" in format_table({"always_cheap|None": stats})
