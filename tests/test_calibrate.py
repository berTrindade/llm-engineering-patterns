import pytest

from context_window_poc.calibrate import fit_tokens_model, held_out_error
from context_window_poc.tokens import CalibrationFit

TRUE_FIT = CalibrationFit(a=0.28, b=9.0, c=3.0)


def _make_sample(chars: int, n_messages: int) -> tuple[int, int, int]:
    tokens = round(TRUE_FIT.a * chars + TRUE_FIT.b * n_messages + TRUE_FIT.c)
    return (chars, n_messages, tokens)


def test_fit_tokens_model_recovers_known_coefficients_on_noise_free_data():
    samples = [
        _make_sample(200, 1),
        _make_sample(1500, 2),
        _make_sample(4000, 5),
        _make_sample(9000, 8),
        _make_sample(20000, 15),
        _make_sample(35000, 22),
    ]

    fit = fit_tokens_model(samples)

    assert fit.a == pytest.approx(TRUE_FIT.a)
    assert fit.b == pytest.approx(TRUE_FIT.b)
    assert fit.c == pytest.approx(TRUE_FIT.c)


def test_fit_tokens_model_needs_varied_samples():
    # All the same (chars, n_messages) pair: the system is singular, can't solve for 3 unknowns.
    samples = [(1000, 3, 300), (1000, 3, 300), (1000, 3, 300)]
    try:
        fit_tokens_model(samples)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_held_out_error_is_zero_for_a_perfect_fit():
    held_out = [_make_sample(700, 3), _make_sample(12000, 10)]
    assert held_out_error(TRUE_FIT, held_out) == 0.0


def test_held_out_error_reports_percentage_deviation():
    fit = CalibrationFit(a=0.25, b=0.0, c=0.0)
    # real tokens 100, estimate 0.25*300=75 -> 25% off
    held_out = [(300, 1, 100)]
    assert held_out_error(fit, held_out) == 25.0
