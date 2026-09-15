"""Cohen's kappa: inter-rater agreement corrected for chance."""

from __future__ import annotations

from collections import Counter


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    if len(labels_a) != len(labels_b):
        raise ValueError("labels_a and labels_b must be the same length")
    n = len(labels_a)
    if n == 0:
        raise ValueError("labels_a and labels_b must not be empty")

    po = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b) / n

    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    categories = set(counts_a) | set(counts_b)
    pe = sum((counts_a.get(c, 0) / n) * (counts_b.get(c, 0) / n) for c in categories)

    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)
