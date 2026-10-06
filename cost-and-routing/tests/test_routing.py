import json
from pathlib import Path

import pytest

from cost_and_routing import routing
from cost_and_routing.llm import LLMResponse
from cost_and_routing.pricing import ModelPrice

FIXTURES = Path(__file__).parent / "fixtures"

PRICES = {
    "cheap-test": ModelPrice(input_per_million=0.10, output_per_million=0.30),
    "strong-test": ModelPrice(input_per_million=0.60, output_per_million=2.00),
}


def _load_fixture(name: str) -> LLMResponse:
    data = json.loads((FIXTURES / name).read_text())
    return LLMResponse(**data)


def _patch_chat(monkeypatch, cheap_fixture: str, strong_fixture: str | None = None):
    def fake_chat_completion(messages, model, **kwargs):
        if model == "cheap-test":
            return _load_fixture(cheap_fixture)
        return _load_fixture(strong_fixture)

    monkeypatch.setattr(routing, "chat_completion", fake_chat_completion)


def test_route_no_escalation_above_threshold(monkeypatch) -> None:
    _patch_chat(monkeypatch, "cheap_high_confidence.json")

    result = routing.route(
        "What is the Standard change fee?",
        cheap_model="cheap-test",
        strong_model="strong-test",
        threshold=0.7,
        prices=PRICES,
    )

    assert result.escalated is False
    assert result.strong_attempt is None
    assert "45" in result.final_answer
    assert result.total_cost_usd > 0


def test_route_escalates_below_threshold(monkeypatch) -> None:
    _patch_chat(monkeypatch, "cheap_low_confidence.json", "strong_answer.json")

    result = routing.route(
        "How long does a refund take?",
        cheap_model="cheap-test",
        strong_model="strong-test",
        threshold=0.7,
        prices=PRICES,
    )

    assert result.escalated is True
    assert result.strong_attempt is not None
    assert "7 business days" in result.final_answer
    # cost must include both the cheap attempt and the strong attempt
    cheap_cost = result.cheap_attempt.llm_response
    strong_cost = result.strong_attempt.llm_response
    assert result.total_cost_usd >= 0
    assert cheap_cost.model == "cheap-test"
    assert strong_cost.model == "strong-test"


def test_route_needs_escalation_overrides_high_confidence(monkeypatch) -> None:
    _patch_chat(monkeypatch, "cheap_needs_escalation.json", "strong_answer.json")

    result = routing.route(
        "What happens if Fernway cancels my flight?",
        cheap_model="cheap-test",
        strong_model="strong-test",
        threshold=0.5,
        prices=PRICES,
    )

    assert result.escalated is True
    assert result.cheap_attempt.confidence == pytest.approx(0.95)


def test_answer_cheap_invalid_json_marks_needs_escalation(monkeypatch) -> None:
    def fake_chat_completion(messages, model, **kwargs):
        return _load_fixture("cheap_invalid_json.json")

    monkeypatch.setattr(routing, "chat_completion", fake_chat_completion)

    attempt = routing.answer_cheap("Any question", "cheap-test")

    assert attempt.needs_escalation is True
    assert attempt.confidence is None


def test_route_escalates_on_cheap_invalid_json(monkeypatch) -> None:
    _patch_chat(monkeypatch, "cheap_invalid_json.json", "strong_answer.json")

    result = routing.route(
        "How long does a refund take?",
        cheap_model="cheap-test",
        strong_model="strong-test",
        threshold=0.7,
        prices=PRICES,
    )

    assert result.escalated is True
    assert result.strong_attempt is not None
