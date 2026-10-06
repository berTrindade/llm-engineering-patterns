from evals_judge import judge
from evals_judge.dataset import ReplyCase
from evals_judge.llm import LLMResponse

POLICY = "30-day returns. No refunds on personalised items."


def _case():
    return ReplyCase(
        id="case-001",
        customer_message="Where is order NW-111111?",
        order_number="NW-111111",
        reply="Order NW-111111 is on its way, 3 to 5 business days.",
        designed_label="pass",
        flaw_type=None,
    )


def test_build_prompt_includes_policy_message_and_reply():
    messages = judge.build_prompt(_case(), POLICY)
    joined = " ".join(m["content"] for m in messages)
    assert "30-day returns" in joined
    assert "NW-111111" in joined
    assert "3 to 5 business days" in joined


def test_judge_case_parses_valid_verdict(monkeypatch):
    def fake_chat(messages, model, response_format=None, **kwargs):
        return LLMResponse(
            content='{"verdict": "pass", "failed_criteria": [], "reason": "looks fine"}',
            prompt_tokens=50,
            completion_tokens=10,
            latency_ms=120.0,
            model=model,
        )

    monkeypatch.setattr(judge, "chat", fake_chat)
    result = judge.judge_case(_case(), POLICY, model="qwen3:8b")
    assert result.verdict == "pass"
    assert result.failed_criteria == []
    assert result.reason == "looks fine"
    assert result.error is None
    assert result.prompt_tokens == 50
    assert result.completion_tokens == 10


def test_judge_case_records_error_on_malformed_json(monkeypatch):
    def fake_chat(messages, model, response_format=None, **kwargs):
        return LLMResponse(
            content="not json",
            prompt_tokens=5,
            completion_tokens=5,
            latency_ms=10.0,
            model=model,
        )

    monkeypatch.setattr(judge, "chat", fake_chat)
    result = judge.judge_case(_case(), POLICY, model="qwen3:8b")
    assert result.verdict is None
    assert result.error is not None


def test_judge_case_records_error_on_exception(monkeypatch):
    def raising_chat(*args, **kwargs):
        raise TimeoutError("boom")

    monkeypatch.setattr(judge, "chat", raising_chat)
    result = judge.judge_case(_case(), POLICY, model="qwen3:8b")
    assert result.verdict is None
    assert "boom" in result.error


def test_judge_pair_parses_winner(monkeypatch):
    def fake_chat(messages, model, response_format=None, **kwargs):
        return LLMResponse(
            content='{"winner": "A"}',
            prompt_tokens=20,
            completion_tokens=3,
            latency_ms=50.0,
            model=model,
        )

    monkeypatch.setattr(judge, "chat", fake_chat)
    result = judge.judge_pair(
        "pair-1", "Where is my order?", "Reply A text", "Reply B text", POLICY, model="qwen3:8b"
    )
    assert result.winner == "A"
    assert result.error is None


def test_judge_pair_records_error_on_malformed_json(monkeypatch):
    def fake_chat(messages, model, response_format=None, **kwargs):
        return LLMResponse(
            content="nope", prompt_tokens=1, completion_tokens=1, latency_ms=1.0, model=model
        )

    monkeypatch.setattr(judge, "chat", fake_chat)
    result = judge.judge_pair(
        "pair-1", "Where is my order?", "Reply A text", "Reply B text", POLICY, model="qwen3:8b"
    )
    assert result.winner is None
    assert result.error is not None
