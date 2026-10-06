import json

import httpx
import pytest

from rag_poc.llm import LLMResponse, chat_completion, embed


def test_chat_completion_sends_reasoning_effort_none_by_default(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://fake/v1")
    monkeypatch.setenv("LLM_API_KEY", "")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "qwen3:8b",
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            },
        )

    monkeypatch.setattr(
        "rag_poc.llm._client", lambda: httpx.Client(transport=httpx.MockTransport(handler))
    )

    result = chat_completion(messages=[{"role": "user", "content": "hi"}], model="qwen3:8b")

    assert seen["url"] == "http://fake/v1/chat/completions"
    assert seen["body"]["reasoning_effort"] == "none"
    assert seen["body"]["model"] == "qwen3:8b"
    assert isinstance(result, LLMResponse)
    assert result.content == "hello"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 3
    assert result.model == "qwen3:8b"
    assert result.latency_ms >= 0


def test_chat_completion_sends_response_format_when_given(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://fake/v1")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "qwen3:8b",
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    monkeypatch.setattr(
        "rag_poc.llm._client", lambda: httpx.Client(transport=httpx.MockTransport(handler))
    )

    schema = {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}
    chat_completion(messages=[{"role": "user", "content": "hi"}], model="qwen3:8b", response_format=schema)

    assert seen["body"]["response_format"] == schema


def test_chat_completion_can_disable_reasoning_effort(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://fake/v1")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "qwen3:8b",
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    monkeypatch.setattr(
        "rag_poc.llm._client", lambda: httpx.Client(transport=httpx.MockTransport(handler))
    )

    chat_completion(messages=[{"role": "user", "content": "hi"}], model="qwen3:8b", reasoning_effort=None)

    assert "reasoning_effort" not in seen["body"]


def test_embed_posts_to_embeddings_endpoint(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://fake/v1")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]},
        )

    monkeypatch.setattr(
        "rag_poc.llm._client", lambda: httpx.Client(transport=httpx.MockTransport(handler))
    )

    vectors = embed(texts=["a", "b"], model="nomic-embed-text")

    assert seen["url"] == "http://fake/v1/embeddings"
    assert seen["body"]["input"] == ["a", "b"]
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_chat_completion_raises_on_http_error(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://fake/v1")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    monkeypatch.setattr(
        "rag_poc.llm._client", lambda: httpx.Client(transport=httpx.MockTransport(handler))
    )

    with pytest.raises(httpx.HTTPStatusError):
        chat_completion(messages=[{"role": "user", "content": "hi"}], model="qwen3:8b")
