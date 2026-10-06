import json

from evals.run import CaseResult, load_completed


def _case(thread_id: str) -> dict:
    return {
        "thread_id": thread_id,
        "strategy": "full_history",
        "budget": None,
        "fact_recall": 1.0,
        "prompt_tokens_real": 15000,
        "extra_tokens": 0,
        "latency_ms": 100.0,
        "estimator_error_pct": 0.5,
        "possible_truncation": False,
        "failed": False,
        "reason": None,
    }


def _write(path, cases):
    path.write_text(json.dumps({"cases": cases}))


def test_returns_nothing_when_resume_is_off(tmp_path):
    path = tmp_path / "results.json"
    _write(path, [_case("ticket-0001")])
    assert load_completed(path, resume=False) == ([], set())


def test_returns_nothing_when_file_is_absent(tmp_path):
    assert load_completed(tmp_path / "missing.json", resume=True) == ([], set())


def test_reads_prior_cases_and_thread_ids(tmp_path):
    path = tmp_path / "results.json"
    _write(path, [_case("ticket-0001"), _case("ticket-0002")])

    cases, done_ids = load_completed(path, resume=True)

    assert done_ids == {"ticket-0001", "ticket-0002"}
    assert [c.thread_id for c in cases] == ["ticket-0001", "ticket-0002"]
    assert all(isinstance(c, CaseResult) for c in cases)


def test_handles_a_results_file_with_no_cases(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"rows": []}))
    assert load_completed(path, resume=True) == ([], set())
