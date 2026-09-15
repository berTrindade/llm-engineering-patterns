from cost_and_routing.scoring import is_correct, normalise


def test_normalise_lowercases_and_collapses_whitespace() -> None:
    assert normalise("  Hello   WORLD  ") == "hello world"


def test_normalise_currency_pound_sign_and_words_match() -> None:
    assert normalise("£45") == normalise("45 pounds")
    assert normalise("£45") == normalise("GBP 45")
    assert normalise("£45") == normalise("45 GBP")


def test_normalise_number_words_match_digits() -> None:
    assert normalise("twenty-four hours") == normalise("24 hours")
    assert normalise("seven business days") == normalise("7 business days")


def test_normalise_strips_punctuation() -> None:
    assert normalise("£45.") == normalise("45 pounds")


def test_is_correct_true_when_all_facts_present() -> None:
    answer = "The change fee is £45 plus any fare difference, refunded within 7 business days."
    assert is_correct(answer, ["£45", "7 business days"])


def test_is_correct_false_when_a_fact_is_missing() -> None:
    answer = "The change fee is £45 plus any fare difference."
    assert not is_correct(answer, ["£45", "7 business days"])


def test_is_correct_matches_number_word_variants() -> None:
    answer = "You must request it at least forty-eight hours before departure."
    assert is_correct(answer, ["48 hours"])


def test_is_correct_case_insensitive() -> None:
    answer = "STANDARD FARE INCLUDES ONE CHECKED BAG."
    assert is_correct(answer, ["Standard", "one checked bag"])
