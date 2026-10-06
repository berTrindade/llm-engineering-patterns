"""Pydantic schema for the agent's proposal on an expense or refund request."""

from typing import Literal

from pydantic import BaseModel


class Proposal(BaseModel):
    action: Literal["approve", "reject", "partial_refund"]
    amount: float | None = None
    policy_clauses: list[str]
    rationale: str
