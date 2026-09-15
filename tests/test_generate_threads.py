import math

from context_window_poc.generate_threads import FACT_KEYS, generate_dataset, generate_thread


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
