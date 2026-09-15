import json

import httpx

from structured_outputs.llm import LLMResponse, call_llm


def _client_with_response(captured: dict, body: dict) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        captured["request_json"] = json.loads(request.content)
        captured["headers"] = request.headers
        return httpx.Response(200, json=body)

    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport, base_url="http://localhost:11434/v1")


def _canned_body(content: str = '{"ok": true}') -> dict:
    return {
        "model": "qwen3:8b",
        "choices": [{"message": {"role": "assistant", "content": content, "tool_calls": None}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4},
    }


def test_reasoning_effort_none_always_sent():
    captured = {}
    client = _client_with_response(captured, _canned_body())
    call_llm(messages=[{"role": "user", "content": "hi"}], model="qwen3:8b", client=client)
    assert captured["request_json"]["reasoning_effort"] == "none"


def test_response_format_is_forwarded():
    captured = {}
    client = _client_with_response(captured, _canned_body())
    fmt = {"type": "json_object"}
    call_llm(
        messages=[{"role": "user", "content": "hi"}],
        model="qwen3:8b",
        response_format=fmt,
        client=client,
    )
    assert captured["request_json"]["response_format"] == fmt


def test_returns_parsed_llm_response():
    captured = {}
    client = _client_with_response(captured, _canned_body('{"hello": "world"}'))
    result = call_llm(messages=[{"role": "user", "content": "hi"}], model="qwen3:8b", client=client)
    assert isinstance(result, LLMResponse)
    assert result.content == '{"hello": "world"}'
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 4
    assert result.model == "qwen3:8b"
    assert result.latency_ms >= 0
    assert result.tool_calls is None


def test_tools_forwarded_and_tool_calls_returned():
    captured = {}
    body = _canned_body("")
    body["choices"][0]["message"]["tool_calls"] = [
        {"id": "1", "type": "function", "function": {"name": "find_slots", "arguments": "{}"}}
    ]
    client = _client_with_response(captured, body)
    tools = [{"type": "function", "function": {"name": "find_slots"}}]
    result = call_llm(
        messages=[{"role": "user", "content": "hi"}],
        model="qwen3:8b",
        tools=tools,
        client=client,
    )
    assert captured["request_json"]["tools"] == tools
    assert result.tool_calls[0]["function"]["name"] == "find_slots"
