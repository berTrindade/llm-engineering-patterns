"""Thin client for an OpenAI-compatible /chat/completions endpoint (Ollama by default)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
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
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 60.0,
    reasoning_effort: str | None = "none",
    extra_body: dict | None = None,
    client: httpx.Client | None = None,
) -> LLMResponse:
    """POST to {base_url}/chat/completions and return content plus usage and timing.

    reasoning_effort="none" stops qwen3 from emitting hidden thinking tokens (verified
    2026-09-15 against Ollama 0.34.0; a /no_think prompt suffix alone is not enough).
    Pass reasoning_effort=None to omit the field entirely.
    """
    base_url = base_url or os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
    api_key = api_key if api_key is not None else os.environ.get("LLM_API_KEY", "")

    body: dict = {"model": model, "messages": messages}
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    if extra_body:
        body.update(extra_body)

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    owns_client = client is None
    client = client or httpx.Client(base_url=base_url, timeout=timeout)
    try:
        start = time.monotonic()
        response = client.post("/chat/completions", json=body, headers=headers)
        latency_ms = (time.monotonic() - start) * 1000
        response.raise_for_status()
        data = response.json()
    finally:
        if owns_client:
            client.close()

    usage = data.get("usage", {})
    return LLMResponse(
        content=data["choices"][0]["message"]["content"],
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        latency_ms=latency_ms,
        model=data.get("model", model),
    )
