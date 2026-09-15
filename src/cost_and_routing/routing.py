"""Cheap-first routing: the cheap model answers with a confidence score, and the
strong model is only called when confidence is low or the cheap model asks for it.
"""

import json
from dataclasses import dataclass

from cost_and_routing.llm import LLMResponse, chat_completion
from cost_and_routing.pricing import ModelPrice, cost_usd

CHEAP_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "cheap_answer",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
                "needs_escalation": {"type": "boolean"},
            },
            "required": ["answer", "confidence", "needs_escalation"],
        },
    },
}

_SYSTEM_PROMPT = (
    "You are a support assistant for Fernway Travel, an invented travel booking company. "
    "Answer the customer's question using the policy facts you know. If you are not "
    "confident in the answer, set a low confidence or set needs_escalation to true."
)


@dataclass
class Attempt:
    answer: str
    confidence: float | None
    needs_escalation: bool
    llm_response: LLMResponse
    failure_reason: str | None = None


@dataclass
class RoutedResult:
    final_answer: str
    escalated: bool
    cheap_attempt: Attempt
    strong_attempt: Attempt | None
    total_cost_usd: float
    total_latency_ms: float


def answer_cheap(question: str, model: str) -> Attempt:
    response = chat_completion(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        model=model,
        response_format=CHEAP_SCHEMA,
    )

    try:
        parsed = json.loads(response.content)
        return Attempt(
            answer=parsed["answer"],
            confidence=float(parsed["confidence"]),
            needs_escalation=bool(parsed["needs_escalation"]),
            llm_response=response,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        # The cheap model failed to produce valid structured output. This is a genuine
        # failure of the cheap attempt: force escalation so the routed strategy still
        # produces an answer, and record why for the caller's per-case failure log.
        return Attempt(
            answer="",
            confidence=None,
            needs_escalation=True,
            llm_response=response,
            failure_reason="cheap model returned invalid JSON for the answer schema",
        )


def answer_strong(question: str, model: str) -> Attempt:
    response = chat_completion(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        model=model,
    )
    return Attempt(
        answer=response.content,
        confidence=None,
        needs_escalation=False,
        llm_response=response,
    )


def route(
    question: str,
    cheap_model: str,
    strong_model: str,
    threshold: float,
    prices: dict[str, ModelPrice],
) -> RoutedResult:
    cheap_attempt = answer_cheap(question, cheap_model)

    low_confidence = cheap_attempt.confidence is None or cheap_attempt.confidence < threshold
    should_escalate = low_confidence or cheap_attempt.needs_escalation

    cheap_cost = cost_usd(
        cheap_attempt.llm_response.prompt_tokens,
        cheap_attempt.llm_response.completion_tokens,
        prices[cheap_model],
    )

    if not should_escalate:
        return RoutedResult(
            final_answer=cheap_attempt.answer,
            escalated=False,
            cheap_attempt=cheap_attempt,
            strong_attempt=None,
            total_cost_usd=cheap_cost,
            total_latency_ms=cheap_attempt.llm_response.latency_ms,
        )

    strong_attempt = answer_strong(question, strong_model)
    strong_cost = cost_usd(
        strong_attempt.llm_response.prompt_tokens,
        strong_attempt.llm_response.completion_tokens,
        prices[strong_model],
    )

    return RoutedResult(
        final_answer=strong_attempt.answer,
        escalated=True,
        cheap_attempt=cheap_attempt,
        strong_attempt=strong_attempt,
        total_cost_usd=cheap_cost + strong_cost,
        total_latency_ms=cheap_attempt.llm_response.latency_ms
        + strong_attempt.llm_response.latency_ms,
    )
