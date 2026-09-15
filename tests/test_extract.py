from structured_outputs.extract import (
    ExtractionResult,
    extract_json_mode,
    extract_json_schema,
    extract_prompt_only,
)
from structured_outputs.llm import LLMResponse

VALID_JSON = (
    '{"supplier_name": "Acme Ltd", "invoice_number": "INV-1", '
    '"invoice_date": "2026-03-03", "due_date": null, "currency": "GBP", '
    '"line_items": [{"description": "Widgets", "quantity": "1", "unit_price": "10.00"}], '
    '"subtotal": "10.00", "vat": "0.00", "total": "10.00", "po_number": null}'
)


class FakeCallFn:
    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=None,
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=1.0,
        model="qwen3:8b",
    )


def test_extract_prompt_only_valid_first_try():
    call_fn = FakeCallFn([_response(f"```json\n{VALID_JSON}\n```")])
    result = extract_prompt_only("case-1", "some email", "qwen3:8b", call_fn=call_fn)
    assert isinstance(result, ExtractionResult)
    assert result.valid_first_try is True
    assert result.valid_after_repair is True
    assert result.invoice.supplier_name == "Acme Ltd"
    assert result.error is None
    assert len(call_fn.calls) == 1


def test_extract_json_mode_valid_first_try():
    call_fn = FakeCallFn([_response(VALID_JSON)])
    result = extract_json_mode("case-1", "some email", "qwen3:8b", call_fn=call_fn)
    assert result.valid_first_try is True
    assert call_fn.calls[0]["response_format"] == {"type": "json_object"}


def test_extract_json_schema_valid_first_try():
    call_fn = FakeCallFn([_response(VALID_JSON)])
    result = extract_json_schema("case-1", "some email", "qwen3:8b", call_fn=call_fn)
    assert result.valid_first_try is True
    assert call_fn.calls[0]["response_format"]["type"] == "json_schema"


def test_repair_retry_succeeds():
    call_fn = FakeCallFn([_response("not json at all"), _response(VALID_JSON)])
    result = extract_json_schema("case-1", "some email", "qwen3:8b", call_fn=call_fn)
    assert result.valid_first_try is False
    assert result.valid_after_repair is True
    assert result.invoice.supplier_name == "Acme Ltd"
    assert len(call_fn.calls) == 2
    repair_messages = call_fn.calls[1]["messages"]
    repair_text = " ".join(m["content"] for m in repair_messages if isinstance(m["content"], str))
    assert "not valid JSON" in repair_text
    assert result.prompt_tokens == 20
    assert result.completion_tokens == 10


def test_repair_retry_fails_both_times():
    call_fn = FakeCallFn([_response("not json"), _response("still not json")])
    result = extract_json_schema("case-1", "some email", "qwen3:8b", call_fn=call_fn)
    assert result.valid_first_try is False
    assert result.valid_after_repair is False
    assert result.invoice is None
    assert result.error is not None
    assert len(call_fn.calls) == 2
