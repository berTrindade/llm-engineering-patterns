import json

from context_window_poc.calibrate import fit_ratio
from context_window_poc.tokens import (
    DEFAULT_CHARS_PER_TOKEN,
    estimate_messages_tokens,
    estimate_tokens,
    load_calibration,
)


def test_estimate_tokens_uses_ratio():
    assert estimate_tokens("a" * 40, chars_per_token=4.0) == 10


def test_estimate_tokens_rounds_and_floors_at_one():
    assert estimate_tokens("", chars_per_token=4.0) == 1
    assert estimate_tokens("ab", chars_per_token=4.0) == 1


def test_estimate_messages_tokens_sums_content():
    messages = [
        {"role": "system", "content": "a" * 40},
        {"role": "user", "content": "a" * 20},
    ]
    assert estimate_messages_tokens(messages, chars_per_token=4.0) == 15


def test_load_calibration_defaults_when_missing(tmp_path):
    missing = tmp_path / "calibration.json"
    assert load_calibration(missing) == DEFAULT_CHARS_PER_TOKEN


def test_load_calibration_reads_stored_ratio(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"chars_per_token": 3.5}))
    assert load_calibration(path) == 3.5


def test_fit_ratio_sums_chars_over_tokens():
    samples = [("a" * 40, 10), ("a" * 20, 10)]
    # total chars 60, total tokens 20 -> ratio 3.0
    assert fit_ratio(samples) == 3.0
