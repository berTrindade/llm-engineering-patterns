"""The model can only answer correctly if it sees the policy. These tests assert the
policy text reaches every call path: always_cheap, always_strong, the routed cheap
attempt, and the routed escalation to the strong model.
"""

import json
from pathlib import Path

from cost_and_routing import routing, strategies
from cost_and_routing.llm import LLMResponse
from cost_and_routing.pricing import ModelPrice
from evals.cases import Question

FIXTURES = Path(__file__).parent / "fixtures"

# A fact that only appears in the policy document, not in any hand-written prompt text.
POLICY_MARKER = "Flex fares cancelled more than 14 days before departure have the £25 fee waived."

PRICES = {
    "cheap-test": ModelPrice(input_per_million=0.10, output_per_million=0.30),
    "strong-test": ModelPrice(input_per_million=0.60, output_per_million=2.00),
}

QUESTION = Question(
    id="q1",
    question="What is the Flex cancellation fee?",
    required_facts=["£25"],
    difficulty="simple_lookup",
)


def _load_fixture(name: str) -> LLMResponse:
    data = json.loads((FIXTURES / name).read_text())
    return LLMResponse(**data)


def _system_content(messages: list[dict]) -> str:
    return next(m["content"] for m in messages if m["role"] == "system")


def test_always_cheap_prompt_includes_policy(monkeypatch) -> None:
    captured = []

    def fake_chat_completion(messages, model, **kwargs):
        captured.append(messages)
        return _load_fixture("cheap_high_confidence.json")

    monkeypatch.setattr(strategies, "chat_completion", fake_chat_completion)

    strategies.run_always_cheap([QUESTION], "cheap-test", PRICES)

    assert POLICY_MARKER in _system_content(captured[0])


def test_always_strong_prompt_includes_policy(monkeypatch) -> None:
    captured = []

    def fake_chat_completion(messages, model, **kwargs):
        captured.append(messages)
        return _load_fixture("strong_answer.json")

    monkeypatch.setattr(strategies, "chat_completion", fake_chat_completion)

    strategies.run_always_strong([QUESTION], "strong-test", PRICES)

    assert POLICY_MARKER in _system_content(captured[0])


def test_routed_cheap_attempt_prompt_includes_policy(monkeypatch) -> None:
    captured = []

    def fake_chat_completion(messages, model, **kwargs):
        captured.append(messages)
        return _load_fixture("cheap_high_confidence.json")

    monkeypatch.setattr(routing, "chat_completion", fake_chat_completion)

    routing.answer_cheap(QUESTION.question, "cheap-test")

    assert POLICY_MARKER in _system_content(captured[0])


def test_routed_escalation_prompt_includes_policy(monkeypatch) -> None:
    captured = []

    def fake_chat_completion(messages, model, **kwargs):
        captured.append((model, messages))
        if model == "cheap-test":
            return _load_fixture("cheap_low_confidence.json")
        return _load_fixture("strong_answer.json")

    monkeypatch.setattr(routing, "chat_completion", fake_chat_completion)

    routing.route(QUESTION.question, "cheap-test", "strong-test", threshold=0.7, prices=PRICES)

    strong_call_messages = next(msgs for model, msgs in captured if model == "strong-test")
    assert POLICY_MARKER in _system_content(strong_call_messages)
