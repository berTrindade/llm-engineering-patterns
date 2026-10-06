import os
from pathlib import Path

import httpx

from hitl import llm

FIXTURE = Path(__file__).parent / "fixtures" / "proposal_approve.json"


def test_fixture_mode_skips_network(monkeypatch):
    monkeypatch.setenv("LLM_FIXTURE", str(FIXTURE))

    def boom(*args, **kwargs):
        raise AssertionError("network call should not happen in fixture mode")

    monkeypatch.setattr(httpx.Client, "post", boom)

    result = llm.chat_completion(messages=[{"role": "user", "content": "hi"}], model="qwen3:8b")

    assert result.model == "qwen3:8b"
    assert result.prompt_tokens == 210
    assert result.completion_tokens == 38
    assert '"action": "approve"' in result.content


def test_fixture_mode_env_cleared(monkeypatch):
    monkeypatch.delenv("LLM_FIXTURE", raising=False)
    assert os.environ.get("LLM_FIXTURE") is None
