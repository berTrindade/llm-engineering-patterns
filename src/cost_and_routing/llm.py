"""A small client for an OpenAI-compatible /v1/chat/completions endpoint.

No provider SDK, no framework: the request body is visible here in full. Thinking is
disabled by default by sending "reasoning_effort": "none". Verified on 2026-09-15
against Ollama 0.34.0 that this stops qwen3 thinking; a /no_think prompt suffix alone
was not enough. The thinking_tokens_check block in results/smoke-2026-09-16.json records
the measured difference on one question: 332 completion tokens with reasoning_effort
omitted, 58 with it set to "none".
"""

import os
import time
from dataclasses import dataclass

import httpx


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str


def chat_completion(
    messages: list[dict],
    model: str,
    *,
    response_format: dict | None = None,
    reasoning_effort: str | None = "none",
    timeout: float = 60.0,
) -> LLMResponse:
    base_url = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
    api_key = os.environ.get("LLM_API_KEY", "")

    body: dict = {"model": model, "messages": messages}
    if response_format is not None:
        body["response_format"] = response_format
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    start = time.monotonic()
    response = httpx.post(
        f"{base_url}/chat/completions",
        json=body,
        headers=headers,
        timeout=timeout,
    )
    latency_ms = (time.monotonic() - start) * 1000
    response.raise_for_status()
    data = response.json()

    usage = data.get("usage", {})
    return LLMResponse(
        content=data["choices"][0]["message"]["content"],
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        latency_ms=latency_ms,
        model=model,
    )
