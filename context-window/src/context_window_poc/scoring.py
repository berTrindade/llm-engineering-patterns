"""Fact recall scoring: how many planted early facts survive into the final answer."""

from __future__ import annotations

import re

_PUNCTUATION = re.compile(r"[^a-z0-9\s]")
_WHITESPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    text = text.lower()
    text = _PUNCTUATION.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def fact_recall(answer: str, expected_facts: list[str]) -> float:
    if not expected_facts:
        return 1.0
    normalised_answer = normalise(answer)
    hits = sum(1 for fact in expected_facts if normalise(fact) in normalised_answer)
    return hits / len(expected_facts)
