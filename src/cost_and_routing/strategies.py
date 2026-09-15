"""Runs the three strategies (always cheap, always strong, routed) over a question set.

Each function calls the real model over the network, so it is exercised by evals/run.py,
never by pytest. A failure on any single question (bad output, timeout, exception) is
recorded as an incorrect CaseResult with a reason, never raised past the loop.
"""

from dataclasses import dataclass

import httpx

from cost_and_routing.llm import chat_completion
from cost_and_routing.pricing import ModelPrice, cost_usd
from cost_and_routing.routing import SYSTEM_PROMPT, route
from cost_and_routing.scoring import is_correct
from evals.cases import Question


@dataclass
class CaseResult:
    question_id: str
    strategy: str
    threshold: float | None
    correct: bool
    escalated: bool
    cost_usd: float
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    confidence: float | None
    failure_reason: str | None


def _single_model_result(
    question: Question, strategy: str, model: str, prices: dict[str, ModelPrice]
) -> CaseResult:
    try:
        response = chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question.question},
            ],
            model=model,
        )
    except (httpx.HTTPError, KeyError) as exc:
        return CaseResult(
            question_id=question.id,
            strategy=strategy,
            threshold=None,
            correct=False,
            escalated=False,
            cost_usd=0.0,
            latency_ms=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            confidence=None,
            failure_reason=f"{type(exc).__name__}: {exc}",
        )

    return CaseResult(
        question_id=question.id,
        strategy=strategy,
        threshold=None,
        correct=is_correct(response.content, question.required_facts),
        escalated=False,
        cost_usd=cost_usd(response.prompt_tokens, response.completion_tokens, prices[model]),
        latency_ms=response.latency_ms,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        confidence=None,
        failure_reason=None,
    )


def run_always_cheap(
    questions: list[Question], model: str, prices: dict[str, ModelPrice]
) -> list[CaseResult]:
    return [_single_model_result(q, "always_cheap", model, prices) for q in questions]


def run_always_strong(
    questions: list[Question], model: str, prices: dict[str, ModelPrice]
) -> list[CaseResult]:
    return [_single_model_result(q, "always_strong", model, prices) for q in questions]


def run_routed(
    questions: list[Question],
    cheap_model: str,
    strong_model: str,
    threshold: float,
    prices: dict[str, ModelPrice],
) -> list[CaseResult]:
    results = []
    for question in questions:
        try:
            routed = route(question.question, cheap_model, strong_model, threshold, prices)
        except httpx.HTTPError as exc:
            results.append(
                CaseResult(
                    question_id=question.id,
                    strategy="routed",
                    threshold=threshold,
                    correct=False,
                    escalated=False,
                    cost_usd=0.0,
                    latency_ms=0.0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    confidence=None,
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        prompt_tokens = routed.cheap_attempt.llm_response.prompt_tokens
        completion_tokens = routed.cheap_attempt.llm_response.completion_tokens
        if routed.strong_attempt is not None:
            prompt_tokens += routed.strong_attempt.llm_response.prompt_tokens
            completion_tokens += routed.strong_attempt.llm_response.completion_tokens

        results.append(
            CaseResult(
                question_id=question.id,
                strategy="routed",
                threshold=threshold,
                correct=is_correct(routed.final_answer, question.required_facts),
                escalated=routed.escalated,
                cost_usd=routed.total_cost_usd,
                latency_ms=routed.total_latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                confidence=routed.cheap_attempt.confidence,
                failure_reason=routed.cheap_attempt.failure_reason,
            )
        )
    return results
