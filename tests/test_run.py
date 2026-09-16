import httpx

from evals.run import build_summary, format_table
from evals_judge.dataset import ReplyCase
from evals_judge.judge import JudgeVerdict, judge_case
from evals_judge.scorers import ScorerResult


def _cases():
    return [
        ReplyCase(
            id="case-001",
            customer_message="m1",
            order_number="NW-100001",
            reply="r1",
            designed_label="pass",
            flaw_type=None,
        ),
        ReplyCase(
            id="case-002",
            customer_message="m2",
            order_number="NW-100002",
            reply="r2",
            designed_label="fail",
            flaw_type="rude_tone",
        ),
    ]


def _scorer_results():
    return {
        "order_number_echoed": [
            ScorerResult("order_number_echoed", True, "ok"),
            ScorerResult("order_number_echoed", False, "missing"),
        ]
    }


def _judge_results():
    return [
        JudgeVerdict(
            case_id="case-001",
            verdict="pass",
            failed_criteria=[],
            reason="fine",
            model="m",
            latency_ms=100.0,
            prompt_tokens=50,
            completion_tokens=10,
        ),
        JudgeVerdict(
            case_id="case-002",
            verdict="fail",
            failed_criteria=["tone"],
            reason="rude",
            model="m",
            latency_ms=120.0,
            prompt_tokens=55,
            completion_tokens=12,
        ),
    ]


def _position_bias():
    return {"flip_rate": 0.1, "first_position_win_share": 0.55, "records": []}


def test_build_summary_scorer_agreement_and_judge_vs_designed():
    summary = build_summary(_cases(), _scorer_results(), _judge_results(), {}, _position_bias())
    assert summary["per_scorer_agreement"]["order_number_echoed"] == 1.0
    assert summary["judge_vs_designed_agreement"] == 1.0
    assert summary["n_cases"] == 2


def test_build_summary_reports_missing_human_labels():
    summary = build_summary(_cases(), _scorer_results(), _judge_results(), {}, _position_bias())
    assert summary["human_kappa"] is None
    assert "Human labels: 0 of 2, run `make label`" in summary["human_status"]


def test_build_summary_computes_kappa_when_human_labels_present():
    human_labels = {"case-001": {"label": "pass", "note": ""}}
    summary = build_summary(
        _cases(), _scorer_results(), _judge_results(), human_labels, _position_bias()
    )
    assert summary["human_kappa"] is not None
    assert "kappa" in summary["human_status"]


def test_format_table_renders_without_raising():
    summary = build_summary(_cases(), _scorer_results(), _judge_results(), {}, _position_bias())
    table = format_table(summary)
    assert "order_number_echoed" in table
    assert "Human labels" in table


def test_judge_call_failure_is_recorded_and_counted(monkeypatch):
    def timed_out(*args, **kwargs):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr("evals_judge.judge.chat", timed_out)
    failed = judge_case(_cases()[1], "policy", "fake-model")

    assert failed.failed is True
    assert failed.verdict is None
    assert failed.error == "ReadTimeout: timed out"

    summary = build_summary(
        _cases(), _scorer_results(), [_judge_results()[0], failed], {}, _position_bias()
    )
    assert summary["judge_failures"] == 1
    assert summary["judge_errors"] == 1
    assert summary["judge_vs_designed_agreement"] == 1.0
    assert "call failures: 1" in format_table(summary)
