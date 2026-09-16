"""Thin httpx wrapper over an OpenAI-compatible /v1/chat/completions endpoint.

reasoning_effort is always sent as "none": verified on 2026-09-15 against Ollama 0.34.0 that
this stops qwen3 thinking tokens; a `/no_think` prompt suffix alone was not enough.
"""

import os
import time
from dataclasses import dataclass

import httpx


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[dict] | None
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str


def call_llm(
    messages: list[dict],
    model: str,
    response_format: dict | None = None,
    tools: list[dict] | None = None,
    timeout: float = 60.0,
    client: httpx.Client | None = None,
) -> LLMResponse:
    base_url = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
    api_key = os.environ.get("LLM_API_KEY", "")

    payload: dict = {
        "model": model,
        "messages": messages,
        "reasoning_effort": "none",
    }
    if response_format is not None:
        payload["response_format"] = response_format
    if tools is not None:
        payload["tools"] = tools

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    owns_client = client is None
    http_client = client or httpx.Client(base_url=base_url, timeout=timeout)
    try:
        start = time.monotonic()
        response = http_client.post("/chat/completions", json=payload, headers=headers)
        latency_ms = (time.monotonic() - start) * 1000
        response.raise_for_status()
        data = response.json()
    finally:
        if owns_client:
            http_client.close()

    message = data["choices"][0]["message"]
    usage = data.get("usage", {})

    return LLMResponse(
        content=message.get("content") or "",
        tool_calls=message.get("tool_calls"),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        latency_ms=latency_ms,
        model=data.get("model", model),
    )
