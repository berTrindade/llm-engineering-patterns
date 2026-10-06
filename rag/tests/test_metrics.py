from rag_poc.metrics import CaseResult, compute_metrics


def _result(**overrides) -> CaseResult:
    defaults = {
        "id": "q1",
        "answerable": True,
        "hit_at_5": True,
        "citation_correct_share": 1.0,
        "refused": False,
        "correct": True,
        "latency_ms": 100.0,
        "prompt_tokens": 50,
        "completion_tokens": 10,
        "failed": False,
        "failure_reason": None,
    }
    defaults.update(overrides)
    return CaseResult(**defaults)


def test_hit_rate_at_5_over_answerable_cases():
    results = [
        _result(id="a", answerable=True, hit_at_5=True),
        _result(id="b", answerable=True, hit_at_5=False),
        _result(id="c", answerable=True, hit_at_5=True),
        _result(id="d", answerable=False, hit_at_5=False, citation_correct_share=None),
    ]
    metrics = compute_metrics(results)
    assert metrics["hit_rate_at_5"] == 2 / 3


def test_citation_correctness_averages_answerable_only():
    results = [
        _result(id="a", answerable=True, citation_correct_share=1.0),
        _result(id="b", answerable=True, citation_correct_share=0.5),
        _result(id="c", answerable=False, citation_correct_share=None),
    ]
    metrics = compute_metrics(results)
    assert metrics["citation_correctness"] == 0.75


def test_refusal_rates():
    results = [
        _result(id="a", answerable=False, refused=True, correct=True, citation_correct_share=None, hit_at_5=False),
        _result(id="b", answerable=False, refused=False, correct=False, citation_correct_share=None, hit_at_5=False),
        _result(id="c", answerable=True, refused=True, correct=False),
        _result(id="d", answerable=True, refused=False, correct=True),
    ]
    metrics = compute_metrics(results)
    assert metrics["correct_refusal_rate"] == 1 / 2
    assert metrics["false_refusal_rate"] == 1 / 2


def test_mean_latency_and_tokens():
    results = [
        _result(id="a", latency_ms=100.0, prompt_tokens=40, completion_tokens=10),
        _result(id="b", latency_ms=200.0, prompt_tokens=60, completion_tokens=20),
    ]
    metrics = compute_metrics(results)
    assert metrics["mean_latency_ms"] == 150.0
    assert metrics["mean_tokens"] == 65.0  # (50 + 80) / 2


def test_failing_ids_lists_id_and_reason():
    results = [
        _result(id="a", failed=True, failure_reason="timeout"),
        _result(id="b", failed=False),
    ]
    metrics = compute_metrics(results)
    assert metrics["failing_ids"] == [{"id": "a", "reason": "timeout"}]
