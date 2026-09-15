import re

from structured_outputs.generator import generate_cases
from structured_outputs.schema import Invoice


def test_generation_is_deterministic():
    first = generate_cases(seed=42, n=5)
    second = generate_cases(seed=42, n=5)
    assert first == second


def test_produces_n_unique_cases():
    cases = generate_cases(seed=42, n=50)
    assert len(cases) == 50
    case_ids = {case["case_id"] for case in cases}
    assert len(case_ids) == 50


def test_every_invoice_validates():
    cases = generate_cases(seed=42, n=50)
    for case in cases:
        Invoice.model_validate(case["invoice"])


def test_currency_renderings_are_varied():
    cases = generate_cases(seed=42, n=50)
    all_text = "\n".join(case["email_text"] for case in cases)
    for token in ("£", "GBP", "EUR", "$"):
        assert token in all_text, f"expected {token!r} to appear somewhere in 50 cases"


def test_po_number_present_and_absent():
    cases = generate_cases(seed=42, n=50)
    po_numbers = [case["invoice"]["po_number"] for case in cases]
    assert any(po is not None for po in po_numbers)
    assert any(po is None for po in po_numbers)


def test_date_formats_are_varied():
    cases = generate_cases(seed=42, n=50)
    all_text = "\n".join(case["email_text"] for case in cases)
    assert re.search(r"\b\d{2}/\d{2}/\d{2}\b", all_text), "expected a DD/MM/YY date"
    assert re.search(r"\b\d{1,2}(st|nd|rd|th) [A-Z][a-z]+ \d{4}\b", all_text), (
        "expected an ordinal 'Do Month YYYY' date"
    )
    assert re.search(r"\b\d{4}-\d{2}-\d{2}\b", all_text), "expected an ISO date"
