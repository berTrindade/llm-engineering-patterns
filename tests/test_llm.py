import json

import httpx

from tool_calling_poc.llm import chat_completion


def _client_with(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_chat_completion_parses_content_and_usage_and_sends_reasoning_effort_none():
    captured_body = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "qwen3:8b",
                "choices": [{"message": {"role": "assistant", "content": "Booked it."}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 8},
            },
        )

    response = chat_completion(
        messages=[{"role": "user", "content": "book a room"}],
        tools=None,
        model="qwen3:8b",
        base_url="http://localhost:11434/v1",
        client=_client_with(handler),
    )

    assert response.content == "Booked it."
    assert response.tool_calls == []
    assert response.prompt_tokens == 120
    assert response.completion_tokens == 8
    assert response.model == "qwen3:8b"
    assert response.latency_ms >= 0
    assert captured_body["reasoning_effort"] == "none"


def test_chat_completion_parses_tool_calls():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "qwen3:8b",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "list_rooms",
                                        "arguments": '{"min_capacity": 4}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 200, "completion_tokens": 15},
            },
        )

    response = chat_completion(
        messages=[{"role": "user", "content": "find a room"}],
        tools=[{"type": "function", "function": {"name": "list_rooms"}}],
        model="qwen3:8b",
        base_url="http://localhost:11434/v1",
        client=_client_with(handler),
    )

    assert response.content is None
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "list_rooms"
    assert call.arguments == '{"min_capacity": 4}'
