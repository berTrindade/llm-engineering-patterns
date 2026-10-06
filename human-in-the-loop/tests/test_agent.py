from pathlib import Path

from hitl import agent

FIXTURES = Path(__file__).parent / "fixtures"

REQUEST = {
    "id": "r1",
    "type": "expense",
    "category": "travel",
    "amount": 42.5,
    "receipt_provided": True,
    "description": "train ticket to client site",
}
POLICY_TEXT = "travel limit £500 per claim; receipts required over £25."


def test_propose_valid_output(monkeypatch):
    monkeypatch.setenv("LLM_FIXTURE", str(FIXTURES / "proposal_approve.json"))

    result = agent.propose(REQUEST, POLICY_TEXT, model="qwen3:8b")

    assert result.ok is True
    assert result.error is None
    assert result.proposal.action == "approve"
    assert result.proposal.amount == 42.5
    assert result.prompt_tokens == 210
    assert result.model == "qwen3:8b"


def test_propose_invalid_output_is_not_raised(monkeypatch):
    monkeypatch.setenv("LLM_FIXTURE", str(FIXTURES / "proposal_invalid.json"))

    result = agent.propose(REQUEST, POLICY_TEXT, model="qwen3:8b")

    assert result.ok is False
    assert result.proposal is None
    assert result.error
