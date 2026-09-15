"""Cheap character-based token estimator, calibrated against real API usage.

There is no local tokenizer for qwen models, so we estimate tokens as
characters / chars_per_token and separately record the real prompt_tokens the API
reports. See calibrate.py for how chars_per_token is fitted, and the README Limits
section for the estimator's known ceiling.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    return max(1, round(len(text) / chars_per_token))


def estimate_messages_tokens(
    messages: list[dict], chars_per_token: float = DEFAULT_CHARS_PER_TOKEN
) -> int:
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return max(1, round(total_chars / chars_per_token))


def load_calibration(path: Path = Path("data/calibration.json")) -> float:
    if not path.exists():
        return DEFAULT_CHARS_PER_TOKEN
    data = json.loads(path.read_text())
    return float(data["chars_per_token"])
