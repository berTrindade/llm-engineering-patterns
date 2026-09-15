"""Seeded, pure-Python generator for the Northwind Home support-reply dataset.

No LLM involved: every message and reply comes from fixed templates, so the
same seed always reproduces the same `data/replies.jsonl`.
"""

from __future__ import annotations

import dataclasses
import json
import random
from dataclasses import dataclass
from pathlib import Path

FLAW_TYPES = [
    "wrong_policy_fact",
    "missing_order_number",
    "forbidden_promise",
    "rude_tone",
    "ignores_question",
]

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "replies.jsonl"

ITEMS = [
    "a ceramic vase",
    "a linen bedspread",
    "a wooden coffee table",
    "a set of glass tumblers",
    "an engraved cheese board",
    "a wool throw",
    "a brass table lamp",
    "a set of kitchen knives",
    "a rattan storage basket",
    "a marble chopping board",
    "a set of scented candles",
    "an oak bookshelf",
]

SCENARIOS = [
    {
        "name": "returns",
        "message": (
            "Hi, I ordered {item} (order {order}) three weeks ago and it's not what I "
            "expected. Can I return it for a refund?"
        ),
        "good_reply": (
            "Hello! Thanks for reaching out about order {order}. You're welcome to return "
            "{item} within 30 days of delivery for a full refund. Since your order was over "
            "GBP 50, the return postage is free; just send it back with your order number "
            "({order}) and we'll process the refund once it arrives."
        ),
        "wrong_fact": ("30 days", "60 days"),
    },
    {
        "name": "delivery",
        "message": (
            "Hi, when will my order {order} for {item} arrive? I chose standard delivery."
        ),
        "good_reply": (
            "Hi, thanks for checking in on order {order}. Standard delivery takes 3 to 5 "
            "business days, so {item} should be with you soon. We'll email you tracking "
            "details once it's dispatched."
        ),
        "wrong_fact": ("3 to 5 business days", "1 to 2 business days"),
    },
    {
        "name": "damaged",
        "message": "Hello, order {order} arrived and {item} is damaged. What do I do?",
        "good_reply": (
            "I'm sorry to hear that, order {order} arriving damaged is not what we want. "
            "Please send us a photo of the damage along with your order number {order} and "
            "we'll send a free replacement straight away, no need to return the damaged item."
        ),
        "wrong_fact": ("a free replacement", "a 20% discount code"),
    },
    {
        "name": "personalised",
        "message": (
            "Hi, I had {item} engraved with my name for order {order} but I've changed my "
            "mind. Can I get a refund?"
        ),
        "good_reply": (
            "Thanks for getting in touch about order {order}. Personalised items like your "
            "engraved {item} can't be returned or refunded once made, unless it arrives "
            "faulty or damaged. Let us know if there's an issue with the item itself and "
            "we'll help from there."
        ),
        "wrong_fact": (
            "can't be returned or refunded once made",
            "can be returned within 30 days like any other item",
        ),
    },
    {
        "name": "status",
        "message": (
            "Hi, can you confirm the status of order {order}? I ordered {item} and haven't "
            "heard anything."
        ),
        "good_reply": (
            "Hi, thanks for checking on order {order}. Your {item} is on its way and, since "
            "you chose standard delivery, it should arrive within 3 to 5 business days of "
            "dispatch. Let us know if it doesn't turn up in that window."
        ),
        "wrong_fact": ("3 to 5 business days", "the same day"),
    },
]


@dataclass
class ReplyCase:
    id: str
    customer_message: str
    order_number: str
    reply: str
    designed_label: str
    flaw_type: str | None


def _apply_wrong_policy_fact(reply: str, wrong_fact: tuple[str, str]) -> str:
    find, replace = wrong_fact
    return reply.replace(find, replace, 1)


def _apply_missing_order_number(reply: str, order_number: str) -> str:
    return reply.replace(order_number, "your order")


def _apply_forbidden_promise(reply: str, scenario_name: str, item: str) -> str:
    if scenario_name == "personalised":
        extra = (
            f" As a one-off, we can still refund your personalised {item} even though it's "
            "not faulty."
        )
    else:
        extra = " We'll also arrange same-day delivery for you at no extra cost."
    return reply + extra


def _apply_rude_tone(reply: str) -> str:
    return "Honestly, this is obvious if you'd checked our policy first. " + reply


def _apply_ignores_question(item: str) -> str:
    return (
        f"Thanks for contacting Northwind Home about {item}! Let us know if there's "
        "anything else we can help with."
    )


def _flaw_assignment(rng: random.Random, n: int) -> list[tuple[str, str | None]]:
    n_fail = n // 2
    n_pass = n - n_fail
    per_flaw = n_fail // len(FLAW_TYPES)
    assignment: list[tuple[str, str | None]] = []
    for flaw in FLAW_TYPES:
        assignment.extend([("fail", flaw)] * per_flaw)
    assignment.extend([("pass", None)] * n_pass)
    rng.shuffle(assignment)
    return assignment


def generate_dataset(seed: int = 42, n: int = 60) -> list[ReplyCase]:
    rng = random.Random(seed)
    assignment = _flaw_assignment(rng, n)

    cases: list[ReplyCase] = []
    for i in range(n):
        label, flaw_type = assignment[i]
        scenario = SCENARIOS[i % len(SCENARIOS)]
        item = rng.choice(ITEMS)
        order_number = f"NW-{rng.randint(100000, 999999)}"

        message = scenario["message"].format(item=item, order=order_number)
        good_reply = scenario["good_reply"].format(item=item, order=order_number)

        if label == "pass":
            reply = good_reply
        elif flaw_type == "wrong_policy_fact":
            reply = _apply_wrong_policy_fact(good_reply, scenario["wrong_fact"])
        elif flaw_type == "missing_order_number":
            reply = _apply_missing_order_number(good_reply, order_number)
        elif flaw_type == "forbidden_promise":
            reply = _apply_forbidden_promise(good_reply, scenario["name"], item)
        elif flaw_type == "rude_tone":
            reply = _apply_rude_tone(good_reply)
        else:
            reply = _apply_ignores_question(item)

        cases.append(
            ReplyCase(
                id=f"case-{i + 1:03d}",
                customer_message=message,
                order_number=order_number,
                reply=reply,
                designed_label=label,
                flaw_type=flaw_type,
            )
        )
    return cases


def main() -> None:
    cases = generate_dataset()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open("w") as f:
        for case in cases:
            f.write(json.dumps(dataclasses.asdict(case)) + "\n")
    print(f"Wrote {len(cases)} cases to {DATA_PATH}")


if __name__ == "__main__":
    main()
