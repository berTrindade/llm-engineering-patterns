"""Cheap token estimator, calibrated against real API usage.

There is no local tokenizer for qwen models, so we estimate tokens with a small linear
model fitted on real usage: tokens ~= a*chars + b*n_messages + c. The b*n_messages term
exists because every message in a chat payload carries a fixed chat-template overhead
(role wrapper, separators) independent of its content length; a single ratio of
chars-per-token (fitted only on short, few-message prompts) folds that fixed overhead
into the ratio and is biased for anything with a different message count. See
calibrate.py for how a, b and c are fitted, and the README Limits section for the
estimator's known ceiling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CalibrationFit:
    a: float  # tokens per character
    b: float  # tokens per message (fixed per-message chat-template overhead)
    c: float  # constant intercept


# Roughly chars/4 with no message overhead: a reasonable fallback when no calibration
# data is available yet.
DEFAULT_FIT = CalibrationFit(a=0.25, b=0.0, c=0.0)


def estimate_tokens_for(chars: int, n_messages: int, fit: CalibrationFit = DEFAULT_FIT) -> int:
    return max(1, round(fit.a * chars + fit.b * n_messages + fit.c))


def estimate_tokens(text: str, fit: CalibrationFit = DEFAULT_FIT) -> int:
    """Back-compat single-string helper: treats the text as one message."""
    return estimate_tokens_for(chars=len(text), n_messages=1, fit=fit)


def estimate_messages_tokens(messages: list[dict], fit: CalibrationFit = DEFAULT_FIT) -> int:
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return estimate_tokens_for(chars=total_chars, n_messages=len(messages), fit=fit)


def load_calibration(path: Path = Path("data/calibration.json")) -> CalibrationFit:
    if not path.exists():
        return DEFAULT_FIT
    data = json.loads(path.read_text())
    return CalibrationFit(a=float(data["a"]), b=float(data["b"]), c=float(data["c"]))
