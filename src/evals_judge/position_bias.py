"""Position-bias check: pairs of replies to the same message, one better by design.

Reuses the scenario templates and flaw injectors from `dataset.py` so each pair's two replies
answer the exact same customer message, instead of trying to retrofit pairing onto the
already-generated 60-case dataset.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from evals_judge.dataset import (
    FLAW_TYPES,
    ITEMS,
    SCENARIOS,
    _apply_forbidden_promise,
    _apply_ignores_question,
    _apply_missing_order_number,
    _apply_rude_tone,
    _apply_wrong_policy_fact,
    _noun,
)
from evals_judge.judge import judge_pair


@dataclass
class Pair:
    id: str
    customer_message: str
    reply_better: str
    reply_worse: str
    flaw_type: str


def _worse_reply(good_reply: str, flaw_type: str, scenario: dict, item: str) -> str:
    if flaw_type == "wrong_policy_fact":
        return _apply_wrong_policy_fact(good_reply, scenario["wrong_fact"])
    if flaw_type == "forbidden_promise":
        return _apply_forbidden_promise(good_reply, scenario["name"], item)
    if flaw_type == "rude_tone":
        return _apply_rude_tone(good_reply)
    if flaw_type == "ignores_question":
        return _apply_ignores_question(item)
    raise ValueError(f"unknown flaw type: {flaw_type}")


def generate_pairs(seed: int = 7, n: int = 20) -> list[Pair]:
    rng = random.Random(seed)
    flaw_cycle = (FLAW_TYPES * ((n // len(FLAW_TYPES)) + 1))[:n]
    rng.shuffle(flaw_cycle)

    pairs: list[Pair] = []
    for i in range(n):
        flaw_type = flaw_cycle[i]
        scenario = SCENARIOS[i % len(SCENARIOS)]
        item = rng.choice(ITEMS)
        order_number = f"NW-{rng.randint(100000, 999999)}"

        message = scenario["message"].format(item=item, noun=_noun(item), order=order_number)
        good_reply = scenario["good_reply"].format(item=item, noun=_noun(item), order=order_number)

        if flaw_type == "missing_order_number":
            worse_reply = _apply_missing_order_number(good_reply, order_number)
        else:
            worse_reply = _worse_reply(good_reply, flaw_type, scenario, item)

        pairs.append(
            Pair(
                id=f"pair-{i + 1:03d}",
                customer_message=message,
                reply_better=good_reply,
                reply_worse=worse_reply,
                flaw_type=flaw_type,
            )
        )
    return pairs


def run_position_bias(
    pairs: list[Pair], policy_text: str, model: str, judge_pair_fn=judge_pair, on_record=None
) -> dict:
    records = []
    failures = 0
    flips = 0
    flip_denominator = 0
    a_wins = 0
    valid_trials = 0

    for pair in pairs:
        r1 = judge_pair_fn(
            f"{pair.id}-ab", pair.customer_message, pair.reply_better, pair.reply_worse,
            policy_text, model,
        )
        r2 = judge_pair_fn(
            f"{pair.id}-ba", pair.customer_message, pair.reply_worse, pair.reply_better,
            policy_text, model,
        )

        run1_winner = {"A": "better", "B": "worse"}.get(r1.winner)
        run2_winner = {"B": "better", "A": "worse"}.get(r2.winner)

        failures += r1.failed + r2.failed
        for verdict in (r1, r2):
            if verdict.winner in ("A", "B"):
                valid_trials += 1
                if verdict.winner == "A":
                    a_wins += 1

        flipped = None
        if run1_winner is not None and run2_winner is not None:
            flip_denominator += 1
            flipped = run1_winner != run2_winner
            if flipped:
                flips += 1

        records.append(
            {
                "pair_id": pair.id,
                "flaw_type": pair.flaw_type,
                "run1_winner": run1_winner,
                "run2_winner": run2_winner,
                "run1_error": r1.error,
                "run2_error": r2.error,
                "flipped": flipped,
            }
        )
        if on_record is not None:
            on_record(records)

    return {
        "flip_rate": flips / flip_denominator if flip_denominator else 0.0,
        "first_position_win_share": a_wins / valid_trials if valid_trials else 0.0,
        "failures": failures,
        "records": records,
    }
