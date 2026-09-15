"""The agent's proposal step: read a request and the policy, propose an action.

Invalid model output is returned as a ProposalResult with ok=False and an error, never
guessed into a plausible-looking proposal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import ValidationError

from hitl.llm import chat_completion
from hitl.models import Proposal

SYSTEM_PROMPT = (
    "You are the claims assistant for Tallow & Finch, a homeware retailer. Read the request "
    "and the policy, then propose one action. Only use facts given in the request; do not "
    "invent amounts, dates or receipt status. Respond with the JSON object only."
)


@dataclass
class ProposalResult:
    ok: bool
    proposal: Proposal | None
    error: str | None
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str


def _response_format() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {"name": "proposal", "schema": Proposal.model_json_schema()},
    }


def propose(request: dict, policy_text: str, model: str) -> ProposalResult:
    user_prompt = (
        f"Policy:\n{policy_text}\n\nRequest:\n{json.dumps(request, indent=2)}\n\n"
        'Reply with JSON: {"action": "approve"|"reject"|"partial_refund", '
        '"amount": number or null, "policy_clauses": [string, ...], "rationale": string}.'
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    result = chat_completion(messages=messages, model=model, response_format=_response_format())

    try:
        proposal = Proposal.model_validate_json(result.content)
    except (ValidationError, json.JSONDecodeError) as exc:
        return ProposalResult(
            ok=False,
            proposal=None,
            error=str(exc),
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_ms=result.latency_ms,
            model=result.model,
        )

    return ProposalResult(
        ok=True,
        proposal=proposal,
        error=None,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        latency_ms=result.latency_ms,
        model=result.model,
    )
