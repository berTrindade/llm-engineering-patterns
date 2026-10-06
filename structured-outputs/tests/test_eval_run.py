import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import httpx

from evals.run import load_cases, print_table, run_pipeline
from structured_outputs.llm import LLMResponse

VALID_JSON = (
    '{"supplier_name": "Acme Ltd", "invoice_number": "INV-1", '
    '"invoice_date": "2026-03-03", "due_date": null, "currency": "GBP", '
    '"line_items": [{"description": "Widgets", "quantity": "1", "unit_price": "10.00"}], '
    '"subtotal": "10.00", "vat": "0.00", "total": "10.00", "po_number": null}'
)

FIXTURE_CASES = [
    {"case_id": "case-a", "invoice": json.loads(VALID_JSON), "email_text": "invoice a"},
    {"case_id": "case-b", "invoice": json.loads(VALID_JSON), "email_text": "invoice b"},
]


def _always_valid_call_fn(**kwargs):
    return LLMResponse(
        content=VALID_JSON,
        tool_calls=None,
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=1.0,
        model="qwen3:8b",
    )


def test_run_pipeline_covers_all_three_approaches():
    result = run_pipeline(
        FIXTURE_CASES, model="qwen3:8b", limit=None, call_fn=_always_valid_call_fn
    )
    assert set(result["approaches"]) == {"prompt_only", "json_mode", "json_schema"}
    for stats in result["approaches"].values():
        assert stats["valid_first_try_rate"] == 1.0
        assert stats["field_accuracy"] == 1.0
        assert stats["failing_case_ids"] == []


def test_run_pipeline_respects_limit():
    calls = []

    def counting_call_fn(**kwargs):
        calls.append(kwargs)
        return _always_valid_call_fn(**kwargs)

    run_pipeline(FIXTURE_CASES, model="qwen3:8b", limit=1, call_fn=counting_call_fn)
    assert len(calls) == 3  # one case x three approaches


def test_run_pipeline_records_failed_case_instead_of_raising(tmp_path):
    def json_mode_times_out(**kwargs):
        if kwargs["response_format"] == {"type": "json_object"}:
            raise httpx.ReadTimeout("timed out")
        return _always_valid_call_fn(**kwargs)

    out_path = tmp_path / "results.json"
    result = run_pipeline(
        FIXTURE_CASES, model="qwen3:8b", call_fn=json_mode_times_out, out_path=out_path
    )

    json_mode = result["approaches"]["json_mode"]
    assert json_mode["failures"] == 2
    assert json_mode["valid_after_repair_rate"] == 0.0
    assert json_mode["field_accuracy"] == 0.0
    assert json_mode["mean_latency_ms"] is None
    assert json_mode["failing_case_ids"][0]["reason"] == "ReadTimeout: timed out"
    assert result["approaches"]["json_schema"]["failures"] == 0
    assert result["approaches"]["json_schema"]["valid_first_try_rate"] == 1.0
    checkpoint = json.loads(out_path.read_text())
    assert checkpoint["complete"] is False
    assert checkpoint["approaches"]["json_mode"]["failures"] == 2


def test_print_table_lists_approach_names(capsys):
    result = run_pipeline(
        FIXTURE_CASES, model="qwen3:8b", limit=None, call_fn=_always_valid_call_fn
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_table(result)
    output = buf.getvalue()
    for approach in ("prompt_only", "json_mode", "json_schema"):
        assert approach in output


def test_load_cases_reads_jsonl(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in FIXTURE_CASES) + "\n")
    loaded = load_cases(Path(path))
    assert loaded == FIXTURE_CASES
