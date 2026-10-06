import math
from pathlib import Path

from context_window_poc.generate_threads import FACT_KEYS, generate_dataset, generate_thread
from context_window_poc.tokens import estimate_tokens_for, load_calibration

LARGEST_BUDGET = 8000
MARGIN_FACTOR = 1.5

# Measured with the committed calibration, the same fit the eval uses. DEFAULT_FIT is a
# chars/4 fallback for when no calibration exists; it underestimates this corpus (dense
# log lines, invoice rows and IDs) by around 40%, so it cannot say whether real prompts
# clear the budget.
FIT = load_calibration(Path(__file__).resolve().parent.parent / "data" / "calibration.json")


def _estimated_tokens(thread) -> int:
    chars = sum(len(t["content"]) for t in thread.turns)
    return estimate_tokens_for(chars=chars, n_messages=len(thread.turns) + 1, fit=FIT)


def test_thread_length_in_range():
    thread = generate_thread(seed=20260915, index=1)
    assert 40 <= len(thread.turns) <= 80


def test_turns_alternate_starting_with_user():
    thread = generate_thread(seed=20260915, index=1)
    for i, turn in enumerate(thread.turns):
        expected_role = "user" if i % 2 == 0 else "assistant"
        assert turn["role"] == expected_role


def test_facts_planted_in_first_twenty_percent():
    thread = generate_thread(seed=20260915, index=2)
    boundary = math.ceil(0.2 * len(thread.turns))
    early_text = " ".join(t["content"] for t in thread.turns[:boundary])
    for key in FACT_KEYS:
        assert thread.facts[key] in early_text


def test_facts_not_repeated_after_first_twenty_percent():
    thread = generate_thread(seed=20260915, index=3)
    boundary = math.ceil(0.2 * len(thread.turns))
    later_text = " ".join(t["content"] for t in thread.turns[boundary:-1])
    for key in FACT_KEYS:
        assert thread.facts[key] not in later_text


def test_final_turn_is_a_question_that_does_not_restate_facts():
    thread = generate_thread(seed=20260915, index=4)
    final_turn = thread.turns[-1]
    assert final_turn["role"] == "user"
    for key in FACT_KEYS:
        assert thread.facts[key] not in final_turn["content"]


def test_expected_facts_has_four_values_in_fact_key_order():
    thread = generate_thread(seed=20260915, index=5)
    assert thread.expected_facts == [thread.facts[key] for key in FACT_KEYS]


def test_generate_thread_is_deterministic():
    a = generate_thread(seed=20260915, index=6)
    b = generate_thread(seed=20260915, index=6)
    assert a.facts == b.facts
    assert a.turns == b.turns


def test_generate_dataset_produces_twenty_unique_threads():
    threads = generate_dataset(n=20, base_seed=20260915)
    assert len(threads) == 20
    ids = {t.thread_id for t in threads}
    assert len(ids) == 20
    account_ids = {t.facts["account_id"] for t in threads}
    assert len(account_ids) > 1


def test_every_thread_exceeds_the_largest_budget_by_a_clear_margin():
    threads = generate_dataset(n=20, base_seed=20260915)
    for thread in threads:
        estimated = _estimated_tokens(thread)
        assert estimated > LARGEST_BUDGET * MARGIN_FACTOR, (
            f"{thread.thread_id} estimated at {estimated} tokens, "
            f"not clearly above {LARGEST_BUDGET * MARGIN_FACTOR:.0f}"
        )
