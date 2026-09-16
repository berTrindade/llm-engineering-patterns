"""Three approaches to extracting an Invoice from a messy email, each with one repair retry."""

import json
from dataclasses import dataclass

from pydantic import ValidationError

from structured_outputs.llm import call_llm
from structured_outputs.schema import Invoice

SYSTEM_PROMPT = (
    "You extract supplier invoice data from a messy email into JSON. "
    "Reply with a single JSON object only, matching this shape: "
    "supplier_name (string), invoice_number (string), invoice_date (YYYY-MM-DD), "
    "due_date (YYYY-MM-DD or null), currency (one of GBP, EUR, USD), "
    "line_items (list of {description, quantity, unit_price}), "
    "subtotal, vat, total (numbers as strings), po_number (string or null)."
)


@dataclass
class ExtractionResult:
    case_id: str
    approach: str
    valid_first_try: bool
    valid_after_repair: bool
    invoice: Invoice | None
    error: str | None
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    failed: bool = False


def _build_messages(email_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": email_text},
    ]


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text


def _parse_invoice(content: str) -> tuple[Invoice | None, str | None]:
    text = _strip_fences(content)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None, "response was not valid JSON"
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            return None, f"response was not valid JSON: {exc}"
    try:
        return Invoice.model_validate(data), None
    except ValidationError as exc:
        return None, str(exc)


def _extract(
    case_id: str,
    email_text: str,
    model: str,
    approach: str,
    response_format: dict | None,
    call_fn,
) -> ExtractionResult:
    messages = _build_messages(email_text)
    response = call_fn(messages=messages, model=model, response_format=response_format)
    prompt_tokens = response.prompt_tokens
    completion_tokens = response.completion_tokens
    latency_ms = response.latency_ms

    invoice, error = _parse_invoice(response.content)
    if invoice is not None:
        return ExtractionResult(
            case_id=case_id,
            approach=approach,
            valid_first_try=True,
            valid_after_repair=True,
            invoice=invoice,
            error=None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )

    repair_messages = messages + [
        {"role": "assistant", "content": response.content},
        {
            "role": "user",
            "content": (
                f"That response was invalid: {error}. "
                "Reply again with corrected JSON only, matching the schema exactly."
            ),
        },
    ]
    repair_response = call_fn(
        messages=repair_messages, model=model, response_format=response_format
    )
    prompt_tokens += repair_response.prompt_tokens
    completion_tokens += repair_response.completion_tokens
    latency_ms += repair_response.latency_ms

    invoice, repair_error = _parse_invoice(repair_response.content)
    return ExtractionResult(
        case_id=case_id,
        approach=approach,
        valid_first_try=False,
        valid_after_repair=invoice is not None,
        invoice=invoice,
        error=None if invoice is not None else repair_error,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
    )


def extract_prompt_only(
    case_id: str, email_text: str, model: str, call_fn=call_llm
) -> ExtractionResult:
    return _extract(case_id, email_text, model, "prompt_only", None, call_fn)


def extract_json_mode(
    case_id: str, email_text: str, model: str, call_fn=call_llm
) -> ExtractionResult:
    return _extract(
        case_id, email_text, model, "json_mode", {"type": "json_object"}, call_fn
    )


def extract_json_schema(
    case_id: str, email_text: str, model: str, call_fn=call_llm
) -> ExtractionResult:
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "Invoice", "schema": Invoice.model_json_schema()},
    }
    return _extract(case_id, email_text, model, "json_schema", response_format, call_fn)
