from context_window_poc.scoring import fact_recall, normalise


def test_normalise_strips_punctuation_and_case():
    assert normalise("Acct  #A-102!") == normalise("acct a 102")


def test_normalise_collapses_whitespace():
    assert normalise("a   b\tc\n") == "a b c"


def test_fact_recall_all_present():
    answer = "Your account is ACC-4821 on the Starter plan, error ERR-503, timezone GMT."
    facts = ["ACC-4821", "Starter", "ERR-503", "GMT"]
    assert fact_recall(answer, facts) == 1.0


def test_fact_recall_none_present():
    answer = "I'm not sure, let me check and get back to you."
    facts = ["ACC-4821", "Starter", "ERR-503", "GMT"]
    assert fact_recall(answer, facts) == 0.0


def test_fact_recall_partial_and_order_independent():
    answer = "Plan: Starter. Error: ERR-503."
    facts = ["ERR-503", "Starter", "ACC-4821", "GMT"]
    assert fact_recall(answer, facts) == 0.5
