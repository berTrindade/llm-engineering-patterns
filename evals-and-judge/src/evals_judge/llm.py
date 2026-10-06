"""Minimal OpenAI-compatible chat client. No provider SDK, no framework."""

from __future__ import annotations

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


def chat(
    messages: list[dict],
    model: str,
    response_format: dict | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 60.0,
    transport: httpx.BaseTransport | None = None,
) -> LLMResponse:
    """POST to {base_url}/chat/completions and return the parsed response.

    Sends "reasoning_effort": "none" so qwen3 models skip their thinking pass;
    verified on 2026-09-15 against Ollama 0.34.0 that this, unlike a /no_think
    prompt suffix alone, actually stops the extra completion tokens.
    """
    base_url = base_url if base_url is not None else os.environ.get(
        "LLM_BASE_URL", "http://localhost:11434/v1"
    )
    api_key = api_key if api_key is not None else os.environ.get("LLM_API_KEY", "")

    body = {"model": model, "messages": messages, "reasoning_effort": "none"}
    if response_format is not None:
        body["response_format"] = response_format

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    start = time.perf_counter()
    with httpx.Client(transport=transport, timeout=timeout) as client:
        response = client.post(f"{base_url}/chat/completions", json=body, headers=headers)
    latency_ms = (time.perf_counter() - start) * 1000

    response.raise_for_status()
    data = response.json()
    usage = data.get("usage", {})

    return LLMResponse(
        content=data["choices"][0]["message"]["content"],
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        latency_ms=latency_ms,
        model=data.get("model", model),
    )
