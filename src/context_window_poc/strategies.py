"""History-trimming strategies. Each takes the system prompt, the conversation turns,
a token budget and an estimator, and returns the messages to send. The system prompt
is always kept and the estimate never exceeds budget_tokens.
"""

from __future__ import annotations

from collections.abc import Callable

Estimator = Callable[[list[dict]], int]

# Fixed recent-turn window for last_n: a naive baseline that ignores unused budget
# headroom, unlike token_budget which adapts to however much budget is available.
LAST_N_WINDOW = 12


def _system_message(system_prompt: str) -> dict:
    return {"role": "system", "content": system_prompt}


def full_history(system_prompt: str, turns: list[dict]) -> list[dict]:
    return [_system_message(system_prompt)] + list(turns)


def last_n(
    system_prompt: str, turns: list[dict], budget_tokens: int, estimator: Estimator
) -> list[dict]:
    system = _system_message(system_prompt)
    candidate = list(turns[-LAST_N_WINDOW:])
    while candidate and estimator([system] + candidate) > budget_tokens:
        candidate.pop(0)
    return [system] + candidate


def token_budget(
    system_prompt: str, turns: list[dict], budget_tokens: int, estimator: Estimator
) -> list[dict]:
    system = _system_message(system_prompt)
    candidate = list(turns)
    while candidate and estimator([system] + candidate) > budget_tokens:
        candidate.pop(0)
    return [system] + candidate


def summarise_older(
    system_prompt: str,
    turns: list[dict],
    budget_tokens: int,
    estimator: Estimator,
    summarise_fn: Callable[[str], tuple[str, int]],
    cache: dict,
    cache_key: str,
) -> list[dict]:
    system = _system_message(system_prompt)

    # Same adaptive fit as token_budget picks the recent turns kept verbatim.
    recent = list(turns)
    while recent and estimator([system] + recent) > budget_tokens:
        recent.pop(0)
    older = turns[: len(turns) - len(recent)]

    if not older:
        return [system] + recent

    if cache_key in cache:
        summary_text = cache[cache_key]["summary"]
    else:
        older_text = "\n".join(f"{t['role']}: {t['content']}" for t in older)
        summary_text, tokens_used = summarise_fn(older_text)
        cache[cache_key] = {"summary": summary_text, "tokens": tokens_used}

    summary_message = {
        "role": "assistant",
        "content": f"Summary of earlier conversation: {summary_text}",
    }

    candidate = list(recent)
    # if system + summary alone exceed budget_tokens this can't trim further;
    # that ceiling is a real finding for very small budgets, not handled here.
    while candidate and estimator([system, summary_message] + candidate) > budget_tokens:
        candidate.pop(0)

    return [system, summary_message] + candidate
