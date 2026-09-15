"""Deterministic generator for invented Ledgerline support ticket threads.

No LLM involved. Each thread plants four facts (account id, plan name, error code,
time zone) in the first two exchanges, fills the middle with unrelated support
back-and-forth that never repeats those facts, then closes with a customer question
that needs them. Everything is invented; see data/README.md.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

FACT_KEYS = ("account_id", "plan_name", "error_code", "timezone")

PLAN_NAMES = ["Starter", "Growth", "Scale", "Enterprise"]
TIMEZONES = [
    "UTC",
    "GMT",
    "Europe/Lisbon",
    "Europe/London",
    "America/New_York",
    "Asia/Singapore",
    "Australia/Sydney",
]

# Unrelated support topics. None of these reference account ids, plan names, error
# codes or time zones, so filling a thread with them never repeats a planted fact.
FILLER_PAIRS = [
    (
        "Quick one while we're at it: can I export my invoices as CSV instead of PDF?",
        "Yes, the invoice list has an Export as CSV option in the top right corner.",
    ),
    (
        "Is there a dark mode in the Ledgerline app?",
        "Not yet, it's on our roadmap but I don't have a date to share.",
    ),
    (
        "Can I add a second team member to review invoices before they go out?",
        "Yes, go to Settings > Team and send an invite with the Reviewer role.",
    ),
    (
        "Does Ledgerline support recurring invoices for retainer clients?",
        "It does, you can set a monthly or weekly schedule when creating an invoice.",
    ),
    (
        "How long do you keep sent invoices in the archive?",
        "Sent invoices are kept indefinitely unless you delete them yourself.",
    ),
    (
        "Is there a mobile app or is it browser only right now?",
        "Browser only for now, but the pages are responsive on mobile.",
    ),
    (
        "Can I change the currency symbol on my invoice template?",
        "Yes, under Settings > Invoice Template there's a currency dropdown.",
    ),
    (
        "Do you support VAT reverse charge for EU clients?",
        "Yes, tick the reverse charge box on the client's tax settings.",
    ),
    (
        "Is there a way to send a payment reminder automatically?",
        "Yes, reminders can be scheduled 3, 7 and 14 days after the due date.",
    ),
    (
        "Can I attach a PDF contract to an invoice?",
        "Yes, the attachment option is below the line items when you edit an invoice.",
    ),
    (
        "What's the limit on how many clients I can add?",
        "There's no hard limit on the number of clients you can add.",
    ),
    (
        "Can I customise the email subject line when sending an invoice?",
        "Yes, under Settings > Email Templates you can edit the subject line.",
    ),
    (
        "Do you have an API for pulling invoice data into our own dashboard?",
        "Yes, there's a REST API, the docs are linked from the Developer settings page.",
    ),
    (
        "Is two-factor authentication available for our team logins?",
        "Yes, each team member can enable it individually under their profile.",
    ),
    (
        "Can late fees be added automatically to overdue invoices?",
        "Yes, you can set a flat fee or a percentage under Settings > Late Fees.",
    ),
    (
        "How do refunds show up on the client's statement?",
        "Refunds appear as a negative line item on the next statement we generate.",
    ),
]

OPENING_USER_TEMPLATES = [
    "Hi, I'm having trouble with our account. Our account ID is {account_id} and we're on the {plan_name} plan.",
    "Hello, we need help on our {plan_name} plan. For reference our account ID is {account_id}.",
    "Hi there, our account ({account_id}) is on {plan_name} and something has gone wrong.",
]

OPENING_ASSISTANT_TEMPLATES = [
    "Thanks for reaching out. Could you tell me the exact error code you're seeing, and what time zone you're in so I can check the logs?",
    "Sorry to hear that. Can you share the error code shown and your time zone, so I can pull the right logs?",
    "Got it, thanks. What error code did the app show, and which time zone should I use when I check the activity log?",
]

OPENING_USER_FOLLOWUP_TEMPLATES = [
    "The error code was {error_code} and we're in the {timezone} time zone.",
    "It showed {error_code}. Our time zone is {timezone}.",
    "We saw {error_code} pop up. For the logs, we're on {timezone}.",
]

OPENING_ASSISTANT_ACK_TEMPLATES = [
    "Thanks, I've noted that and I'm looking into it now.",
    "Got it, noted. I'll dig into this and update you shortly.",
    "Thank you, that's logged on our side. Give me a moment to investigate.",
]

CLOSING_QUESTION_TEMPLATES = [
    "Before we close this out, can you confirm the account ID, the plan we're on, the original error code you logged, and the time zone you have on file for us?",
    "One last thing before we wrap up: could you read back our account ID, plan, the error code from earlier, and our time zone?",
    "Can you confirm for our records what account ID, plan, error code and time zone you have noted for this ticket?",
]


@dataclass(frozen=True)
class Thread:
    thread_id: str
    facts: dict[str, str]
    turns: list[dict]
    expected_facts: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _make_facts(rng: random.Random) -> dict[str, str]:
    return {
        "account_id": f"ACC-{rng.randint(1000, 9999)}",
        "plan_name": rng.choice(PLAN_NAMES),
        "error_code": f"ERR-{rng.randint(100, 999)}",
        "timezone": rng.choice(TIMEZONES),
    }


def _opening_turns(rng: random.Random, facts: dict[str, str]) -> list[dict]:
    return [
        {"role": "user", "content": rng.choice(OPENING_USER_TEMPLATES).format(**facts)},
        {"role": "assistant", "content": rng.choice(OPENING_ASSISTANT_TEMPLATES)},
        {"role": "user", "content": rng.choice(OPENING_USER_FOLLOWUP_TEMPLATES).format(**facts)},
        {"role": "assistant", "content": rng.choice(OPENING_ASSISTANT_ACK_TEMPLATES)},
    ]


def _filler_turns(rng: random.Random, n_pairs: int) -> list[dict]:
    turns: list[dict] = []
    for _ in range(n_pairs):
        user_text, assistant_text = rng.choice(FILLER_PAIRS)
        turns.append({"role": "user", "content": user_text})
        turns.append({"role": "assistant", "content": assistant_text})
    return turns


def generate_thread(seed: int, index: int) -> Thread:
    rng = random.Random(seed + index)
    facts = _make_facts(rng)

    # Odd total in [41, 79] keeps the final turn's index even (role "user").
    total_turns = rng.choice(range(41, 80, 2))
    opening = _opening_turns(rng, facts)
    filler_pairs = (total_turns - len(opening) - 1) // 2
    filler = _filler_turns(rng, filler_pairs)
    closing = [{"role": "user", "content": rng.choice(CLOSING_QUESTION_TEMPLATES)}]

    turns = opening + filler + closing
    expected_facts = [facts[key] for key in FACT_KEYS]

    return Thread(
        thread_id=f"ticket-{index:04d}",
        facts=facts,
        turns=turns,
        expected_facts=expected_facts,
    )


def generate_dataset(n: int = 20, base_seed: int = 20260915) -> list[Thread]:
    return [generate_thread(seed=base_seed, index=i) for i in range(1, n + 1)]


def main() -> None:
    out_dir = Path("data/threads")
    out_dir.mkdir(parents=True, exist_ok=True)
    threads = generate_dataset()
    for thread in threads:
        path = out_dir / f"{thread.thread_id}.json"
        path.write_text(json.dumps(thread.to_dict(), indent=2) + "\n")
    print(f"Wrote {len(threads)} threads to {out_dir}")


if __name__ == "__main__":
    main()
