"""Thin httpx client for an OpenAI-compatible /chat/completions endpoint.

Sends reasoning_effort: "none" by default, verified on 2026-09-15 against Ollama 0.34.0 that
this stops qwen3 thinking; a `/no_think` prompt suffix alone was not enough (a 20-token JSON
answer reported 542 completion tokens without this field).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import httpx


@dataclass
class ChatResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str


def _from_fixture(fixture_path: str) -> ChatResult:
    # Test-only bypass so unit tests and the crash test need no live model;
    # a fake HTTP server would be more code for the same guarantee.
    data = json.loads(open(fixture_path).read())
    return ChatResult(
        content=data["content"],
        prompt_tokens=data["prompt_tokens"],
        completion_tokens=data["completion_tokens"],
        latency_ms=0.0,
        model=data["model"],
    )


def chat_completion(
    messages: list[dict],
    model: str,
    response_format: dict | None = None,
    timeout: float = 60.0,
) -> ChatResult:
    fixture_path = os.environ.get("LLM_FIXTURE")
    if fixture_path:
        return _from_fixture(fixture_path)

    base_url = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
    api_key = os.environ.get("LLM_API_KEY", "")

    body: dict = {
        "model": model,
        "messages": messages,
        "reasoning_effort": "none",
    }
    if response_format is not None:
        body["response_format"] = response_format

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    start = time.monotonic()
    with httpx.Client(timeout=timeout) as client:
        response = client.post(f"{base_url}/chat/completions", json=body, headers=headers)
    latency_ms = (time.monotonic() - start) * 1000

    response.raise_for_status()
    data = response.json()
    usage = data.get("usage", {})
    return ChatResult(
        content=data["choices"][0]["message"]["content"],
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        latency_ms=latency_ms,
        model=data.get("model", model),
    )
