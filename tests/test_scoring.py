from structured_outputs.extract import ExtractionResult
from structured_outputs.schema import Invoice
from structured_outputs.scoring import aggregate, score_case

BASE = {
    "supplier_name": "Acme Ltd",
    "invoice_number": "INV-1",
    "invoice_date": "2026-03-03",
    "due_date": "2026-04-03",
    "currency": "GBP",
    "line_items": [
        {"description": "Widgets", "quantity": "2", "unit_price": "5.00"},
        {"description": "Gadgets", "quantity": "1", "unit_price": "3.00"},
    ],
    "subtotal": "13.00",
    "vat": "2.60",
    "total": "15.60",
    "po_number": "PO-1",
}


def _invoice(**overrides) -> Invoice:
    data = {**BASE, **overrides}
    return Invoice.model_validate(data)


def test_identical_invoices_score_all_true():
    truth = _invoice()
    predicted = _invoice()
    scores = score_case(truth, predicted)
    assert all(scores.values())


def test_money_field_compared_to_2dp():
    truth = _invoice(total="15.60")
    predicted = _invoice(total="15.601")
    scores = score_case(truth, predicted)
    assert scores["total"] is True


def test_money_field_mismatch_detected():
    truth = _invoice(total="15.60")
    predicted = _invoice(total="16.00")
    scores = score_case(truth, predicted)
    assert scores["total"] is False


def test_line_items_compared_as_multiset():
    truth = _invoice()
    reordered = dict(BASE)
    reordered["line_items"] = list(reversed(BASE["line_items"]))
    predicted = _invoice(**reordered)
    scores = score_case(truth, predicted)
    assert scores["line_items"] is True


def test_line_items_mismatch_detected():
    truth = _invoice()
    predicted = _invoice(
        line_items=[{"description": "Widgets", "quantity": "99", "unit_price": "5.00"}]
    )
    scores = score_case(truth, predicted)
    assert scores["line_items"] is False


def test_missing_prediction_scores_all_false():
    truth = _invoice()
    scores = score_case(truth, None)
    assert not any(scores.values())


def _result(case_id, approach, valid_first, valid_repair, invoice, error, latency=100.0):
    return ExtractionResult(
        case_id=case_id,
        approach=approach,
        valid_first_try=valid_first,
        valid_after_repair=valid_repair,
        invoice=invoice,
        error=error,
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=latency,
    )


def test_aggregate_over_one_approach():
    truth = _invoice()
    results = [
        _result("case-1", "json_schema", True, True, truth, None),
        _result("case-2", "json_schema", False, False, None, "bad json"),
    ]
    ground_truths = {"case-1": truth, "case-2": truth}
    agg = aggregate(results, ground_truths)
    stats = agg["json_schema"]
    assert stats["valid_first_try_rate"] == 0.5
    assert stats["valid_after_repair_rate"] == 0.5
    assert stats["mean_latency_ms"] == 100.0
    assert stats["mean_prompt_tokens"] == 10
    assert stats["mean_completion_tokens"] == 5
    assert stats["failing_case_ids"] == [{"case_id": "case-2", "reason": "bad json"}]
    assert 0.0 <= stats["field_accuracy"] <= 1.0
