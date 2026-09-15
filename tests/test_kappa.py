import pytest

from evals_judge.kappa import cohens_kappa


def test_cohens_kappa_textbook_example():
    # Classic two-rater yes/no contingency table used in kappa tutorials:
    #             B yes   B no
    #   A yes       20      5
    #   A no        10     15
    # po = 0.7, pe = 0.5, kappa = 0.4
    labels_a = ["yes"] * 25 + ["no"] * 25
    labels_b = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    assert cohens_kappa(labels_a, labels_b) == pytest.approx(0.4)


def test_cohens_kappa_perfect_agreement():
    labels_a = ["pass", "fail", "pass", "pass", "fail"]
    labels_b = ["pass", "fail", "pass", "pass", "fail"]
    assert cohens_kappa(labels_a, labels_b) == pytest.approx(1.0)


def test_cohens_kappa_chance_agreement_is_zero():
    labels_a = ["yes", "yes", "no", "no"]
    labels_b = ["yes", "no", "yes", "no"]
    assert cohens_kappa(labels_a, labels_b) == pytest.approx(0.0)


def test_cohens_kappa_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        cohens_kappa(["pass"], ["pass", "fail"])
