from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel


class LineItem(BaseModel):
    description: str
    quantity: Decimal
    unit_price: Decimal


class Invoice(BaseModel):
    supplier_name: str
    invoice_number: str
    invoice_date: date
    due_date: date | None
    currency: Literal["GBP", "EUR", "USD"]
    line_items: list[LineItem]
    subtotal: Decimal
    vat: Decimal
    total: Decimal
    po_number: str | None
