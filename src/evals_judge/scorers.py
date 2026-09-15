"""Deterministic scorers: no model calls, just pattern checks on the reply text."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from evals_judge.dataset import ReplyCase

MIN_WORDS = 8
MAX_WORDS = 100

FORBIDDEN_PATTERNS = [
    re.compile(r"refund (your|the) personalis", re.IGNORECASE),
    re.compile(r"same[- ]day delivery", re.IGNORECASE),
]

ORDER_NUMBER_RE = re.compile(r"NW-\d{6}")


@dataclass
class ScorerResult:
    name: str
    passed: bool
    detail: str


def order_number_echoed(case: ReplyCase) -> ScorerResult:
    match = ORDER_NUMBER_RE.search(case.customer_message)
    if not match:
        return ScorerResult("order_number_echoed", True, "not applicable: no order number in message")
    order_number = match.group(0)
    if order_number in case.reply:
        return ScorerResult("order_number_echoed", True, f"{order_number} echoed in reply")
    return ScorerResult("order_number_echoed", False, f"{order_number} missing from reply")


def no_forbidden_promise(case: ReplyCase) -> ScorerResult:
    for pattern in FORBIDDEN_PATTERNS:
        if pattern.search(case.reply):
            return ScorerResult("no_forbidden_promise", False, f"matched forbidden pattern: {pattern.pattern}")
    return ScorerResult("no_forbidden_promise", True, "no forbidden promise found")


def length_within_bounds(case: ReplyCase) -> ScorerResult:
    n_words = len(case.reply.split())
    if n_words < MIN_WORDS:
        return ScorerResult("length_within_bounds", False, f"{n_words} words, below {MIN_WORDS}")
    if n_words > MAX_WORDS:
        return ScorerResult("length_within_bounds", False, f"{n_words} words, above {MAX_WORDS}")
    return ScorerResult("length_within_bounds", True, f"{n_words} words")


SCORERS: list[Callable[[ReplyCase], ScorerResult]] = [
    order_number_echoed,
    no_forbidden_promise,
    length_within_bounds,
]


def score_case(case: ReplyCase) -> list[ScorerResult]:
    return [scorer(case) for scorer in SCORERS]


def scorer_agreement(cases: list[ReplyCase], scorer: Callable[[ReplyCase], ScorerResult]) -> float:
    if not cases:
        return 0.0
    agree = 0
    for case in cases:
        predicted = "pass" if scorer(case).passed else "fail"
        if predicted == case.designed_label:
            agree += 1
    return agree / len(cases)
