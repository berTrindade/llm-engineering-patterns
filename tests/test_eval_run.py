import json
from pathlib import Path

import httpx

from evals.run import run_assessment
from hitl import agent

FIXTURES = Path(__file__).parent / "fixtures"


def test_model_error_is_recorded_as_failed_case_not_raised(tmp_path, monkeypatch):
    cases_path = tmp_path / "requests.jsonl"
    cases_path.write_text(
        "\n".join(
            json.dumps({"id": case_id, "amount": 42.5, "expected_action": "approve"})
            for case_id in ("c1", "c2")
        )
    )
    monkeypatch.setenv("LLM_FIXTURE", str(FIXTURES / "proposal_approve.json"))
    real_propose = agent.propose

    def times_out_on_c1(request, policy_text, model):
        if request["id"] == "c1":
            raise httpx.ReadTimeout("timed out")
        return real_propose(request, policy_text, model)

    monkeypatch.setattr(agent, "propose", times_out_on_c1)
    out_path = tmp_path / "results.json"

    results = run_assessment(cases_path, out_path=out_path)

    failed, passed = results["cases"]
    assert failed["failed"] is True
    assert failed["correct"] is False
    assert failed["error"] == "ReadTimeout: timed out"
    assert passed["failed"] is False
    assert passed["correct"] is True
    assert results["summary"]["failures"] == 1
    assert results["summary"]["proposal_accuracy"] == 0.5
    assert results["summary"]["mean_total_tokens"] == 248
    assert json.loads(out_path.read_text())["summary"]["failures"] == 1
