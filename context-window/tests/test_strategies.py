from context_window_poc.strategies import (
    LAST_N_WINDOW,
    full_history,
    last_n,
    summarise_older,
    token_budget,
)

SYSTEM_PROMPT = "You are a Ledgerline support agent. Be concise and accurate."


def _fake_estimator(messages: list[dict]) -> int:
    return sum(len(m["content"]) for m in messages) // 4


def _make_turns(n: int) -> list[dict]:
    turns = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        # 40 chars each -> 10 estimated tokens per turn
        turns.append({"role": role, "content": f"turn {i:03d} " + "x" * 33})
    return turns


def test_full_history_keeps_everything():
    turns = _make_turns(20)
    messages = full_history(SYSTEM_PROMPT, turns)
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1:] == turns


def test_last_n_never_drops_system_prompt():
    turns = _make_turns(60)
    messages = last_n(SYSTEM_PROMPT, turns, budget_tokens=2000, estimator=_fake_estimator)
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}


def test_last_n_never_exceeds_budget():
    turns = _make_turns(60)
    messages = last_n(SYSTEM_PROMPT, turns, budget_tokens=50, estimator=_fake_estimator)
    assert _fake_estimator(messages) <= 50


def test_last_n_caps_at_fixed_window_even_with_huge_budget():
    turns = _make_turns(60)
    messages = last_n(SYSTEM_PROMPT, turns, budget_tokens=1_000_000, estimator=_fake_estimator)
    assert len(messages) - 1 <= LAST_N_WINDOW


def test_token_budget_never_drops_system_prompt():
    turns = _make_turns(60)
    messages = token_budget(SYSTEM_PROMPT, turns, budget_tokens=2000, estimator=_fake_estimator)
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}


def test_token_budget_never_exceeds_budget():
    turns = _make_turns(60)
    messages = token_budget(SYSTEM_PROMPT, turns, budget_tokens=50, estimator=_fake_estimator)
    assert _fake_estimator(messages) <= 50


def test_token_budget_adapts_further_than_last_n_when_budget_is_large():
    turns = _make_turns(60)
    generous_budget = 2000  # far larger than LAST_N_WINDOW * ~10 tokens/turn
    last_n_messages = last_n(SYSTEM_PROMPT, turns, generous_budget, _fake_estimator)
    token_budget_messages = token_budget(SYSTEM_PROMPT, turns, generous_budget, _fake_estimator)
    assert len(token_budget_messages) > len(last_n_messages)


def _fake_summarise(older_text: str) -> tuple[str, int]:
    return f"Summary of {len(older_text)} chars of earlier conversation.", 25


def test_summarise_older_never_drops_system_prompt():
    turns = _make_turns(60)
    cache: dict = {}
    messages = summarise_older(
        SYSTEM_PROMPT, turns, 200, _fake_estimator, _fake_summarise, cache, "thread-1:200"
    )
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}


def test_summarise_older_never_exceeds_budget():
    turns = _make_turns(60)
    cache: dict = {}
    messages = summarise_older(
        SYSTEM_PROMPT, turns, 200, _fake_estimator, _fake_summarise, cache, "thread-1:200"
    )
    assert _fake_estimator(messages) <= 200


def test_summarise_older_caches_summary_call():
    turns = _make_turns(60)
    cache: dict = {}
    calls = {"n": 0}

    def counting_summarise(older_text: str) -> tuple[str, int]:
        calls["n"] += 1
        return _fake_summarise(older_text)

    summarise_older(SYSTEM_PROMPT, turns, 200, _fake_estimator, counting_summarise, cache, "t1:200")
    summarise_older(SYSTEM_PROMPT, turns, 200, _fake_estimator, counting_summarise, cache, "t1:200")

    assert calls["n"] == 1
    assert cache["t1:200"]["tokens"] == 25
