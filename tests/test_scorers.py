from evals_judge.dataset import ReplyCase
from evals_judge.scorers import (
    SCORERS,
    length_within_bounds,
    no_forbidden_promise,
    order_number_echoed,
    score_case,
    scorer_agreement,
)


def _case(order_number, message, reply, designed_label="pass", flaw_type=None):
    return ReplyCase(
        id="t",
        customer_message=message,
        order_number=order_number,
        reply=reply,
        designed_label=designed_label,
        flaw_type=flaw_type,
    )


def test_order_number_echoed_pass_when_present():
    case = _case("NW-111111", "order NW-111111 please", "About order NW-111111, all good.")
    result = order_number_echoed(case)
    assert result.passed is True


def test_order_number_echoed_fail_when_missing():
    case = _case("NW-111111", "order NW-111111 please", "Thanks for your message, all good.")
    result = order_number_echoed(case)
    assert result.passed is False


def test_order_number_echoed_not_applicable_when_message_has_no_order_number():
    case = _case(None, "I have a general question", "Sure, happy to help.")
    result = order_number_echoed(case)
    assert result.passed is True
    assert "not applicable" in result.detail


def test_no_forbidden_promise_fails_on_personalised_refund():
    case = _case(
        "NW-100001",
        "personalised item refund?",
        "We will refund your personalised item as a one-off exception.",
    )
    result = no_forbidden_promise(case)
    assert result.passed is False


def test_no_forbidden_promise_fails_on_same_day_delivery():
    case = _case("NW-100001", "when will it arrive", "We'll arrange same-day delivery for you.")
    result = no_forbidden_promise(case)
    assert result.passed is False


def test_no_forbidden_promise_passes_on_clean_reply():
    case = _case(
        "NW-100001", "when will it arrive", "It should arrive within 3 to 5 business days."
    )
    result = no_forbidden_promise(case)
    assert result.passed is True


def test_length_within_bounds_fails_on_short_reply():
    case = _case("NW-100001", "hi", "Thanks!")
    assert length_within_bounds(case).passed is False


def test_length_within_bounds_passes_on_reasonable_reply():
    case = _case(
        "NW-100001",
        "hi",
        "Thanks for reaching out about your order, we'll get this sorted for you very soon.",
    )
    assert length_within_bounds(case).passed is True


def test_length_within_bounds_fails_on_very_long_reply():
    case = _case("NW-100001", "hi", "word " * 150)
    assert length_within_bounds(case).passed is False


def test_score_case_runs_all_scorers():
    case = _case(
        "NW-100001", "order NW-100001 here", "About order NW-100001, thanks for reaching out today."
    )
    results = score_case(case)
    assert len(results) == len(SCORERS)
    assert {r.name for r in results} == {
        "order_number_echoed",
        "no_forbidden_promise",
        "length_within_bounds",
    }


def test_scorer_agreement_with_designed_labels():
    cases = [
        _case(
            "NW-100001", "order NW-100001", "Order NW-100001, all sorted.", designed_label="pass"
        ),
        _case(
            "NW-200002",
            "order NW-200002",
            "All sorted for you.",
            designed_label="fail",
            flaw_type="missing_order_number",
        ),
    ]
    agreement = scorer_agreement(cases, order_number_echoed)
    assert agreement == 1.0
