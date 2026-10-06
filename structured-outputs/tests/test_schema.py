from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from structured_outputs.schema import Invoice


def _valid_invoice_dict():
    return {
        "supplier_name": "Acme Supplies Ltd",
        "invoice_number": "INV-1001",
        "invoice_date": "2026-03-03",
        "due_date": "2026-04-03",
        "currency": "GBP",
        "line_items": [
            {"description": "Widgets", "quantity": "10", "unit_price": "2.50"},
        ],
        "subtotal": "25.00",
        "vat": "5.00",
        "total": "30.00",
        "po_number": "PO-42",
    }


def test_valid_invoice_builds():
    invoice = Invoice.model_validate(_valid_invoice_dict())
    assert invoice.supplier_name == "Acme Supplies Ltd"
    assert invoice.invoice_date == date(2026, 3, 3)
    assert invoice.line_items[0].quantity == Decimal("10")
    assert invoice.line_items[0].unit_price == Decimal("2.50")
    assert invoice.total == Decimal("30.00")
    assert invoice.po_number == "PO-42"


def test_invalid_currency_rejected():
    data = _valid_invoice_dict()
    data["currency"] = "JPY"
    with pytest.raises(ValidationError):
        Invoice.model_validate(data)


def test_optional_fields_accept_none():
    data = _valid_invoice_dict()
    data["due_date"] = None
    data["po_number"] = None
    invoice = Invoice.model_validate(data)
    assert invoice.due_date is None
    assert invoice.po_number is None


def test_decimal_fields_coerce_from_string():
    data = _valid_invoice_dict()
    invoice = Invoice.model_validate(data)
    assert isinstance(invoice.subtotal, Decimal)
    assert isinstance(invoice.line_items[0].quantity, Decimal)
