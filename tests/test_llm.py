import json

import httpx
import pytest

from evals_judge.llm import chat


def _canned_response(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    assert body["reasoning_effort"] == "none"
    assert body["model"] == "qwen3:8b"
    return httpx.Response(
        200,
        json={
            "model": "qwen3:8b",
            "choices": [{"message": {"content": '{"answer": "ok"}'}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5},
        },
    )


def test_chat_sends_reasoning_effort_and_parses_response():
    transport = httpx.MockTransport(_canned_response)
    result = chat(
        messages=[{"role": "user", "content": "hi"}],
        model="qwen3:8b",
        base_url="http://fake/v1",
        api_key="",
        transport=transport,
    )
    assert result.content == '{"answer": "ok"}'
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 5
    assert result.model == "qwen3:8b"
    assert result.latency_ms >= 0


def _error_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, json={"error": "boom"})


def test_chat_raises_on_non_200():
    transport = httpx.MockTransport(_error_response)
    with pytest.raises(httpx.HTTPStatusError):
        chat(
            messages=[{"role": "user", "content": "hi"}],
            model="qwen3:8b",
            base_url="http://fake/v1",
            api_key="",
            transport=transport,
        )


def test_chat_passes_response_format(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "qwen3:8b",
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    transport = httpx.MockTransport(handler)
    schema = {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}
    chat(
        messages=[{"role": "user", "content": "hi"}],
        model="qwen3:8b",
        response_format=schema,
        base_url="http://fake/v1",
        api_key="",
        transport=transport,
    )
    assert captured["body"]["response_format"] == schema
