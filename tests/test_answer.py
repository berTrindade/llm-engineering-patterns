import json

import pytest
from pydantic import ValidationError

from rag_poc.answer import REFUSAL_ANSWER, AnswerResult, answer_question
from rag_poc.chunk import Chunk


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(id=cid, text=text, source_url="https://example.test", page_title="P", heading_path="P > H")


def test_answer_question_refuses_below_threshold_without_calling_model(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("chat_completion should not be called below the threshold")

    monkeypatch.setattr("rag_poc.answer.chat_completion", boom)

    retrieved = [(_chunk("a", "unrelated text"), 0.1)]

    result, llm_response = answer_question(
        question="What is the meaning of life?",
        retrieved=retrieved,
        cheap_model="qwen3:8b",
        similarity_threshold=0.3,
    )

    assert result.refused is True
    assert result.answer == REFUSAL_ANSWER
    assert result.citations == []
    assert llm_response is None


def test_answer_question_calls_model_above_threshold_and_parses_result(monkeypatch):
    fixture = {
        "answer": "Landlords must fix the boiler.",
        "citations": ["a"],
        "refused": False,
    }

    class FakeResponse:
        content = json.dumps(fixture)
        prompt_tokens = 100
        completion_tokens = 20
        latency_ms = 5.0
        model = "qwen3:8b"

    def fake_chat_completion(**kwargs):
        return FakeResponse()

    monkeypatch.setattr("rag_poc.answer.chat_completion", fake_chat_completion)

    retrieved = [(_chunk("a", "Landlords must fix the boiler within a reasonable time."), 0.8)]

    result, llm_response = answer_question(
        question="Who fixes the boiler?",
        retrieved=retrieved,
        cheap_model="qwen3:8b",
        similarity_threshold=0.3,
    )

    assert isinstance(result, AnswerResult)
    assert result.answer == "Landlords must fix the boiler."
    assert result.citations == ["a"]
    assert result.refused is False
    assert llm_response.prompt_tokens == 100


def test_answer_question_raises_on_malformed_json(monkeypatch):
    class FakeResponse:
        content = "not json"
        prompt_tokens = 1
        completion_tokens = 1
        latency_ms = 1.0
        model = "qwen3:8b"

    monkeypatch.setattr("rag_poc.answer.chat_completion", lambda **kwargs: FakeResponse())

    retrieved = [(_chunk("a", "Landlords must fix the boiler."), 0.8)]

    with pytest.raises((json.JSONDecodeError, ValidationError)):
        answer_question(
            question="Who fixes the boiler?",
            retrieved=retrieved,
            cheap_model="qwen3:8b",
            similarity_threshold=0.3,
        )
