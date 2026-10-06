"""Per-case field scoring and per-approach aggregation."""

from collections import Counter, defaultdict
from decimal import ROUND_HALF_UP, Decimal

from structured_outputs.extract import ExtractionResult
from structured_outputs.schema import Invoice

MONEY_FIELDS = ("subtotal", "vat", "total")
SIMPLE_FIELDS = ("supplier_name", "invoice_number", "currency", "po_number")
DATE_FIELDS = ("invoice_date", "due_date")
FIELD_NAMES = (*SIMPLE_FIELDS, *DATE_FIELDS, *MONEY_FIELDS, "line_items")


def _round2(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _line_item_multiset(invoice: Invoice) -> Counter:
    return Counter(
        (item.description, _round2(item.quantity), _round2(item.unit_price))
        for item in invoice.line_items
    )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def score_case(ground_truth: Invoice, predicted: Invoice | None) -> dict[str, bool]:
    if predicted is None:
        return dict.fromkeys(FIELD_NAMES, False)

    scores: dict[str, bool] = {}
    for field in SIMPLE_FIELDS:
        scores[field] = getattr(ground_truth, field) == getattr(predicted, field)
    for field in DATE_FIELDS:
        scores[field] = getattr(ground_truth, field) == getattr(predicted, field)
    for field in MONEY_FIELDS:
        scores[field] = _round2(getattr(ground_truth, field)) == _round2(
            getattr(predicted, field)
        )
    scores["line_items"] = _line_item_multiset(ground_truth) == _line_item_multiset(predicted)
    return scores


def aggregate(
    results: list[ExtractionResult], ground_truths: dict[str, Invoice]
) -> dict[str, dict]:
    by_approach: dict[str, list[ExtractionResult]] = defaultdict(list)
    for result in results:
        by_approach[result.approach].append(result)

    output: dict[str, dict] = {}
    for approach, approach_results in by_approach.items():
        n = len(approach_results)
        # Failed cases have no latency or token counts, so they are left out of those means;
        # they still count as invalid and wrong in the rates, and in failures.
        measured = [r for r in approach_results if not r.failed]
        field_scores: list[bool] = []
        failing_case_ids = []
        for result in approach_results:
            truth = ground_truths[result.case_id]
            case_scores = score_case(truth, result.invoice)
            field_scores.extend(case_scores.values())
            if not result.valid_after_repair:
                failing_case_ids.append({"case_id": result.case_id, "reason": result.error})

        output[approach] = {
            "valid_first_try_rate": sum(r.valid_first_try for r in approach_results) / n,
            "valid_after_repair_rate": sum(r.valid_after_repair for r in approach_results) / n,
            "field_accuracy": sum(field_scores) / len(field_scores) if field_scores else 0.0,
            "mean_latency_ms": _mean([r.latency_ms for r in measured]),
            "mean_prompt_tokens": _mean([r.prompt_tokens for r in measured]),
            "mean_completion_tokens": _mean([r.completion_tokens for r in measured]),
            "failures": n - len(measured),
            "failing_case_ids": failing_case_ids,
        }
    return output
