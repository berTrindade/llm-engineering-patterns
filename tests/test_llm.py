import json

import httpx

from context_window_poc.llm import LLMResponse, chat_completion


def _mock_client(captured: dict) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "model": "qwen3:8b",
                "choices": [{"message": {"content": "hello there"}}],
                "usage": {"prompt_tokens": 42, "completion_tokens": 3},
            },
        )

    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport, base_url="http://localhost:11434/v1")


def test_chat_completion_returns_populated_response():
    captured: dict = {}
    client = _mock_client(captured)

    result = chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        model="qwen3:8b",
        base_url="http://localhost:11434/v1",
        client=client,
    )

    assert isinstance(result, LLMResponse)
    assert result.content == "hello there"
    assert result.prompt_tokens == 42
    assert result.completion_tokens == 3
    assert result.model == "qwen3:8b"
    assert result.latency_ms >= 0
    assert captured["url"].endswith("/chat/completions")


def test_chat_completion_sends_reasoning_effort_none_by_default():
    captured: dict = {}
    client = _mock_client(captured)

    chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        model="qwen3:8b",
        base_url="http://localhost:11434/v1",
        client=client,
    )

    assert captured["body"]["reasoning_effort"] == "none"


def test_chat_completion_reasoning_effort_can_be_disabled():
    captured: dict = {}
    client = _mock_client(captured)

    chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        model="qwen3:8b",
        base_url="http://localhost:11434/v1",
        client=client,
        reasoning_effort=None,
    )

    assert "reasoning_effort" not in captured["body"]
