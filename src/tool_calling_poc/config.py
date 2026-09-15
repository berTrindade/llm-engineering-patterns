from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    llm_base_url: str
    llm_api_key: str
    cheap_model: str
    strong_model: str
    embed_model: str


def load_config() -> Config:
    return Config(
        llm_base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
        llm_api_key=os.environ.get("LLM_API_KEY", ""),
        cheap_model=os.environ.get("CHEAP_MODEL", "qwen3:8b"),
        strong_model=os.environ.get("STRONG_MODEL", "qwen3.5:35b"),
        embed_model=os.environ.get("EMBED_MODEL", "nomic-embed-text"),
    )
