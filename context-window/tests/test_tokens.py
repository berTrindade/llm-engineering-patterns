import json

from context_window_poc.tokens import (
    DEFAULT_FIT,
    CalibrationFit,
    estimate_messages_tokens,
    estimate_tokens,
    estimate_tokens_for,
    load_calibration,
)


def test_estimate_tokens_for_uses_fit():
    fit = CalibrationFit(a=0.25, b=10.0, c=2.0)
    # 400 chars * 0.25 + 1 message * 10 + 2 = 100 + 10 + 2 = 112
    assert estimate_tokens_for(chars=400, n_messages=1, fit=fit) == 112


def test_estimate_tokens_for_rounds_and_floors_at_one():
    fit = CalibrationFit(a=0.0, b=0.0, c=0.0)
    assert estimate_tokens_for(chars=0, n_messages=0, fit=fit) == 1


def test_estimate_tokens_treats_text_as_one_message():
    fit = CalibrationFit(a=0.25, b=10.0, c=2.0)
    assert estimate_tokens("a" * 400, fit=fit) == 112


def test_estimate_messages_tokens_sums_content_and_counts_messages():
    fit = CalibrationFit(a=0.25, b=10.0, c=2.0)
    messages = [
        {"role": "system", "content": "a" * 40},
        {"role": "user", "content": "a" * 20},
    ]
    # 60 chars * 0.25 + 2 messages * 10 + 2 = 15 + 20 + 2 = 37
    assert estimate_messages_tokens(messages, fit=fit) == 37


def test_estimate_messages_tokens_uses_default_fit_when_none_given():
    messages = [{"role": "user", "content": "a" * 40}]
    assert estimate_messages_tokens(messages) == estimate_messages_tokens(messages, DEFAULT_FIT)


def test_load_calibration_defaults_when_missing(tmp_path):
    missing = tmp_path / "calibration.json"
    assert load_calibration(missing) == DEFAULT_FIT


def test_load_calibration_reads_stored_fit(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"a": 0.3, "b": 12.0, "c": 5.0}))
    assert load_calibration(path) == CalibrationFit(a=0.3, b=12.0, c=5.0)
