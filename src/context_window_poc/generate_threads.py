"""Deterministic generator for invented Ledgerline support ticket threads.

No LLM involved. Each thread plants four facts (account id, plan name, error code,
time zone) in the first two exchanges, fills the middle with unrelated support
back-and-forth that never repeats those facts, then closes with a customer question
that needs them. Full-history threads land at roughly 34,000-40,000 characters (about
14,000-17,000 tokens by the fit in data/calibration.json), so at 2000, 4000 and 8000
token budgets the trimming strategies actually have to trim something.
Everything is invented; see data/README.md.
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

# Pasted content pools used to pad filler turns to a realistic length (pasted log
# excerpts, pasted invoice line items). None of these reference account ids, plan
# names, error codes or time zones.
LOG_LEVELS = ["INFO", "WARN", "ERROR", "DEBUG"]
LOG_COMPONENTS = [
    "invoicing-api",
    "auth-service",
    "pdf-renderer",
    "webhook-dispatcher",
    "billing-worker",
    "export-service",
    "notification-queue",
    "scheduler",
]
LOG_MESSAGES = [
    "request timed out after 30s",
    "retrying operation (attempt 2 of 3)",
    "connection reset by peer",
    "validation failed for field 'amount'",
    "queue depth exceeded threshold",
    "cache miss, falling back to database",
    "rate limit exceeded, backing off",
    "payload exceeded max size",
    "unexpected null value in response",
    "successfully processed batch of 24 records",
    "token refresh succeeded",
    "scheduled job completed in 842ms",
    "health check passed",
    "circuit breaker opened for downstream call",
    "deduplicated 3 identical events",
]

INVOICE_ITEMS = [
    "Consulting hours",
    "Design review",
    "Monthly retainer",
    "Software licence",
    "Onboarding session",
    "Support hours",
    "API usage overage",
    "Custom report",
    "Training session",
    "Data migration",
    "Priority support",
    "Integration setup",
]


def _log_excerpt(rng: random.Random, n_lines: int) -> str:
    lines = []
    hour, minute, second = rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59)
    day, month = rng.randint(1, 28), rng.randint(1, 12)
    for _ in range(n_lines):
        timestamp = f"2026-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}Z"
        level = rng.choice(LOG_LEVELS)
        component = rng.choice(LOG_COMPONENTS)
        message = rng.choice(LOG_MESSAGES)
        lines.append(f"{timestamp} [{level}] {component}: {message}")
        second += rng.randint(1, 45)
        minute += second // 60
        second %= 60
        hour = (hour + minute // 60) % 24
        minute %= 60
    return "\n".join(lines)


def _invoice_lines(rng: random.Random, n_lines: int) -> str:
    lines = []
    for _ in range(n_lines):
        qty = rng.randint(1, 12)
        item = rng.choice(INVOICE_ITEMS)
        amount = rng.randint(20, 450)
        lines.append(f"{qty} x {item} - GBP {amount}.00")
    return "\n".join(lines)


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


def _filler_pair(rng: random.Random) -> tuple[dict, dict]:
    user_text, assistant_text = rng.choice(FILLER_PAIRS)

    # Realistic bulk: a pasted log excerpt on most replies (this is what actually
    # makes threads long, the way a real agent pastes logs while investigating), and
    # sometimes a pasted invoice on the customer's side too.
    log_block = _log_excerpt(rng, rng.randint(15, 30))
    assistant_text = f"{assistant_text}\n\nHere's what I'm seeing in the logs:\n{log_block}"

    if rng.random() < 0.3:
        invoice_block = _invoice_lines(rng, rng.randint(6, 16))
        user_text = f"{user_text}\n\nFor reference, here are the line items in question:\n{invoice_block}"

    return (
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    )


# Character target for a full thread, applied before the final padding step, which
# carries threads somewhat past it: actual output is roughly 34,000-40,000 characters,
# about 14,000-17,000 tokens by the fit in data/calibration.json. That is well above
# the largest 8,000 token budget under test.
TARGET_CHARS_RANGE = (28_000, 36_000)
MIN_FILLER_PAIRS, MAX_FILLER_PAIRS = 18, 37


def generate_thread(seed: int, index: int) -> Thread:
    rng = random.Random(seed + index)
    facts = _make_facts(rng)

    opening = _opening_turns(rng, facts)
    closing = [{"role": "user", "content": rng.choice(CLOSING_QUESTION_TEMPLATES)}]
    target_chars = rng.randint(*TARGET_CHARS_RANGE)

    filler: list[dict] = []
    current_chars = sum(len(t["content"]) for t in opening + closing)
    pairs_built = 0
    while True:
        user_turn, assistant_turn = _filler_pair(rng)
        filler.append(user_turn)
        filler.append(assistant_turn)
        current_chars += len(user_turn["content"]) + len(assistant_turn["content"])
        pairs_built += 1
        if pairs_built >= MAX_FILLER_PAIRS:
            break
        if pairs_built >= MIN_FILLER_PAIRS and current_chars >= target_chars:
            break

    if current_chars < target_chars:
        # Turn-count ceiling reached before the character target: pad the last turn
        # with more pasted log lines rather than adding more turns.
        deficit = target_chars - current_chars
        extra_lines = max(5, deficit // 55)
        filler[-1]["content"] += "\n\nAdditional log context:\n" + _log_excerpt(rng, extra_lines)

    # Total turns = 4 opening + 2*pairs_built filler + 1 closing, always odd, so the
    # final turn's index is even and its role is "user".
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
