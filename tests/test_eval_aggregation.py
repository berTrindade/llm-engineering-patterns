from evals.run import CaseResult, aggregate, format_table


def _row_for(rows, strategy, budget):
    return next(r for r in rows if r["strategy"] == strategy and r["budget"] == budget)


def test_aggregate_computes_means_and_counts_failures():
    results = [
        CaseResult("t1", "last_n", 2000, 1.0, 500, 0, 120.0, 5.0, False, None),
        CaseResult("t2", "last_n", 2000, 0.5, 600, 0, 140.0, 10.0, False, None),
        CaseResult("t3", "last_n", 2000, None, None, 0, None, None, True, "timeout"),
    ]

    rows = aggregate(results)
    row = _row_for(rows, "last_n", 2000)

    assert row["n"] == 3
    assert row["failures"] == 1
    assert row["fact_recall"] == 0.5  # (1.0 + 0.5 + 0) / 3, failures count as 0
    assert row["prompt_tokens_real"] == 550  # mean over successes only
    assert row["latency_ms"] == 130.0
    assert row["estimator_error_pct"] == 7.5


def test_aggregate_groups_by_strategy_and_budget():
    results = [
        CaseResult("t1", "last_n", 2000, 1.0, 500, 0, 100.0, 1.0, False, None),
        CaseResult("t1", "last_n", 4000, 1.0, 900, 0, 110.0, 1.0, False, None),
        CaseResult("t1", "token_budget", 2000, 0.75, 480, 0, 90.0, 2.0, False, None),
    ]

    rows = aggregate(results)
    keys = {(r["strategy"], r["budget"]) for r in rows}
    assert keys == {("last_n", 2000), ("last_n", 4000), ("token_budget", 2000)}


def test_aggregate_averages_extra_tokens_for_summarise_older():
    results = [
        CaseResult("t1", "summarise_older", 2000, 1.0, 500, 30, 100.0, 1.0, False, None),
        CaseResult("t2", "summarise_older", 2000, 1.0, 520, 50, 100.0, 1.0, False, None),
    ]

    rows = aggregate(results)
    row = _row_for(rows, "summarise_older", 2000)
    assert row["extra_tokens"] == 40


def test_format_table_contains_headers_and_rows():
    results = [CaseResult("t1", "last_n", 2000, 1.0, 500, 0, 100.0, 1.0, False, None)]
    rows = aggregate(results)
    table = format_table(rows)

    for header in (
        "strategy",
        "budget",
        "fact_recall",
        "prompt_tokens",
        "extra_tokens",
        "latency",
        "estimator_error",
    ):
        assert header in table
    assert "last_n" in table
    assert "2000" in table
