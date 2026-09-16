"""Thin httpx client for an OpenAI-compatible chat/embeddings API (Ollama by default).

Verified locally on 2026-09-15 against Ollama 0.34.0: usage is reported, response_format with
json_schema returns valid JSON, and a `/no_think` prompt suffix alone does not stop qwen3
thinking. `reasoning_effort: "none"` does stop it, so it is sent by default on every chat call.
"""

import os
import time

import httpx
from pydantic import BaseModel


class LLMResponse(BaseModel):
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str


def _base_url() -> str:
    return os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")


def _headers() -> dict:
    api_key = os.environ.get("LLM_API_KEY", "")
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _client() -> httpx.Client:
    return httpx.Client()


def chat_completion(
    messages: list[dict],
    model: str,
    response_format: dict | None = None,
    reasoning_effort: str | None = "none",
    timeout: float = 60.0,
) -> LLMResponse:
    payload = {"model": model, "messages": messages}
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    if response_format is not None:
        payload["response_format"] = response_format

    start = time.monotonic()
    with _client() as client:
        response = client.post(
            f"{_base_url()}/chat/completions",
            json=payload,
            headers=_headers(),
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
        model=data.get("model", model),
    )


def embed(texts: list[str], model: str, timeout: float = 60.0) -> list[list[float]]:
    payload = {"model": model, "input": texts}

    with _client() as client:
        response = client.post(
            f"{_base_url()}/embeddings",
            json=payload,
            headers=_headers(),
            timeout=timeout,
        )
    response.raise_for_status()
    data = response.json()
    return [item["embedding"] for item in data["data"]]
