from __future__ import annotations

import time
from dataclasses import dataclass

import httpx


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall]
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str


def chat_completion(
    messages: list[dict],
    tools: list[dict] | None,
    model: str,
    base_url: str,
    api_key: str = "",
    reasoning_effort: str | None = "none",
    timeout: float = 60.0,
    client: httpx.Client | None = None,
) -> LLMResponse:
    body: dict = {"model": model, "messages": messages}
    if tools:
        body["tools"] = tools
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    own_client = client is None
    http_client = client or httpx.Client(timeout=timeout)
    try:
        start = time.monotonic()
        response = http_client.post(f"{base_url}/chat/completions", json=body, headers=headers)
        latency_ms = (time.monotonic() - start) * 1000
        response.raise_for_status()
        data = response.json()
    finally:
        if own_client:
            http_client.close()

    message = data["choices"][0]["message"]
    tool_calls = [
        ToolCall(
            id=call["id"],
            name=call["function"]["name"],
            arguments=call["function"]["arguments"],
        )
        for call in message.get("tool_calls") or []
    ]
    usage = data.get("usage") or {}

    return LLMResponse(
        content=message.get("content"),
        tool_calls=tool_calls,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        latency_ms=latency_ms,
        model=data.get("model", model),
    )
