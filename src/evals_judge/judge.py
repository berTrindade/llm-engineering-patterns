"""LLM-as-judge: single-reply verdicts and A/B pair verdicts, both structured output."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from evals_judge.dataset import ReplyCase
from evals_judge.llm import chat

RUBRIC_PATH = Path(__file__).resolve().parent.parent.parent / "evals" / "rubric_prompt.md"

JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["pass", "fail"]},
                "failed_criteria": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": ["verdict", "failed_criteria", "reason"],
            "additionalProperties": False,
        },
    },
}

PAIR_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "pair_verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"winner": {"type": "string", "enum": ["A", "B"]}},
            "required": ["winner"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class JudgeVerdict:
    case_id: str
    verdict: str | None
    failed_criteria: list[str] = field(default_factory=list)
    reason: str = ""
    model: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


@dataclass
class PairVerdict:
    pair_id: str
    winner: str | None
    model: str = ""
    latency_ms: float = 0.0
    error: str | None = None


def _rubric_text() -> str:
    return RUBRIC_PATH.read_text()


def build_prompt(case: ReplyCase, policy_text: str) -> list[dict]:
    system = f"{_rubric_text()}\n\nPolicy:\n{policy_text}"
    user = f"Customer message:\n{case.customer_message}\n\nSupport reply:\n{case.reply}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def judge_case(case: ReplyCase, policy_text: str, model: str) -> JudgeVerdict:
    messages = build_prompt(case, policy_text)
    try:
        response = chat(messages, model=model, response_format=JUDGE_SCHEMA)
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        return JudgeVerdict(case_id=case.id, verdict=None, model=model, error=str(exc))

    try:
        data = json.loads(response.content)
        verdict = data["verdict"]
        failed_criteria = data.get("failed_criteria", [])
        reason = data.get("reason", "")
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return JudgeVerdict(
            case_id=case.id,
            verdict=None,
            model=response.model,
            latency_ms=response.latency_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            error=f"invalid judge output: {exc}",
        )

    return JudgeVerdict(
        case_id=case.id,
        verdict=verdict,
        failed_criteria=failed_criteria,
        reason=reason,
        model=response.model,
        latency_ms=response.latency_ms,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
    )


def _pair_prompt(customer_message: str, reply_a: str, reply_b: str, policy_text: str) -> list[dict]:
    system = (
        f"{_rubric_text()}\n\nPolicy:\n{policy_text}\n\nYou are given two candidate replies to "
        "the same customer message, A and B. Pick the one that better satisfies the criteria "
        "above. Respond with the winner only, using the given JSON schema."
    )
    user = (
        f"Customer message:\n{customer_message}\n\nReply A:\n{reply_a}\n\nReply B:\n{reply_b}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def judge_pair(
    pair_id: str,
    customer_message: str,
    reply_a: str,
    reply_b: str,
    policy_text: str,
    model: str,
) -> PairVerdict:
    messages = _pair_prompt(customer_message, reply_a, reply_b, policy_text)
    try:
        response = chat(messages, model=model, response_format=PAIR_SCHEMA)
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        return PairVerdict(pair_id=pair_id, winner=None, model=model, error=str(exc))

    try:
        data = json.loads(response.content)
        winner = data["winner"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return PairVerdict(
            pair_id=pair_id,
            winner=None,
            model=response.model,
            latency_ms=response.latency_ms,
            error=f"invalid judge output: {exc}",
        )

    return PairVerdict(
        pair_id=pair_id, winner=winner, model=response.model, latency_ms=response.latency_ms
    )
