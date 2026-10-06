"""Seeded, pure-python generator for invented supplier invoice emails.

No LLM is used here. Running this module writes data/cases.jsonl.
"""

import json
import random
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

SEED = 42
N_CASES = 50

SUPPLIERS = [
    "Northbridge Stationery Ltd",
    "Faircroft Logistics",
    "Meadowlane Print Co",
    "Dunmore IT Supplies",
    "Greystone Facilities",
    "Ashwell Packaging",
    "Bellhaven Catering",
    "Corbett Hardware",
    "Elderfield Cleaning Services",
    "Foxglove Office Furniture",
]

ITEM_DESCRIPTIONS = [
    "A4 copier paper (box of 5 reams)",
    "Ergonomic desk chair",
    "USB-C docking station",
    "Cleaning supplies bundle",
    "Catering for 20 people",
    "Toner cartridge, black",
    "Cardboard shipping boxes (pack of 50)",
    "Whiteboard, 120x90cm",
    "LED desk lamp",
    "Courier delivery, next day",
    "Server rack shelf",
    "Visitor badges (pack of 100)",
]

SIGNOFFS = ["Kind regards,", "Best regards,", "Many thanks,", "Regards,"]
SENDER_NAMES = ["Priya Shah", "Tom Ellery", "Marcus Webb", "Sian Cole", "Danny Osei"]

_ORDINAL_SUFFIX = {1: "st", 2: "nd", 3: "rd"}


def _ordinal(day: int) -> str:
    if 11 <= day % 100 <= 13:
        return "th"
    return _ORDINAL_SUFFIX.get(day % 10, "th")


def _fmt_date(rng: random.Random, d: date) -> str:
    style = rng.choice(["ordinal", "slash", "iso"])
    if style == "ordinal":
        return f"{d.day}{_ordinal(d.day)} {d.strftime('%B')} {d.year}"
    if style == "slash":
        return d.strftime("%d/%m/%y")
    return d.isoformat()


def _fmt_money(amount: Decimal, symbol: str) -> str:
    quantised = amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{symbol}{quantised:,.2f}"


CURRENCY_RENDERINGS = {
    "GBP": ["£", "GBP"],
    "EUR": ["EUR", "€"],
    "USD": ["$", "USD"],
}


def _currency_symbol(currency: str, rng: random.Random) -> str:
    rendering = rng.choice(CURRENCY_RENDERINGS[currency])
    return rendering if rendering in ("£", "€", "$") else rendering + " "


def _render_line_items(rng: random.Random, items: list[dict], symbol: str) -> str:
    style = rng.choice(["table", "bullets", "prose"])
    if style == "table":
        lines = ["Description          Qty   Unit price"]
        for item in items:
            lines.append(
                f"{item['description']:<20} {item['quantity']:>3}   "
                f"{_fmt_money(item['unit_price'], symbol)}"
            )
        return "\n".join(lines)
    if style == "bullets":
        return "\n".join(
            f"- {item['description']} x{item['quantity']} @ "
            f"{_fmt_money(item['unit_price'], symbol)} each"
            for item in items
        )
    parts = [
        f"{item['quantity']} x {item['description']} at "
        f"{_fmt_money(item['unit_price'], symbol)} each"
        for item in items
    ]
    return "We are billing you for " + ", ".join(parts) + "."


def _forwarded_header(rng: random.Random, supplier: str) -> str:
    sender = rng.choice(SENDER_NAMES)
    return (
        "---------- Forwarded message ---------\n"
        f"From: {sender} <accounts@{supplier.split()[0].lower()}.example.com>\n"
        f"Subject: Invoice attached\n\n"
    )


def _signature(rng: random.Random, supplier: str) -> str:
    sender = rng.choice(SENDER_NAMES)
    return f"\n\n{rng.choice(SIGNOFFS)}\n{sender}\n{supplier}\n"


def _build_case(rng: random.Random, index: int) -> dict:
    supplier = rng.choice(SUPPLIERS)
    currency = rng.choice(["GBP", "EUR", "USD"])
    symbol = _currency_symbol(currency, rng)

    invoice_date = date(2026, 1, 1) + timedelta(days=rng.randint(0, 300))
    has_due_date = rng.random() < 0.8
    due_date = invoice_date + timedelta(days=rng.choice([14, 30])) if has_due_date else None

    n_items = rng.randint(1, 4)
    items = []
    for _ in range(n_items):
        quantity = rng.randint(1, 20)
        unit_price = Decimal(rng.randint(150, 25000)) / 100
        items.append(
            {
                "description": rng.choice(ITEM_DESCRIPTIONS),
                "quantity": quantity,
                "unit_price": unit_price.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            }
        )

    subtotal = sum(
        (item["unit_price"] * item["quantity"] for item in items), Decimal("0.00")
    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    vat_stated = rng.random() < 0.6
    vat_rate = Decimal("0.20")
    vat = (subtotal * vat_rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    total = subtotal + vat

    has_po = rng.random() < 0.7
    po_number = f"PO-{rng.randint(1000, 9999)}" if has_po else None

    invoice_number = f"INV-{2026000 + index}"

    body_lines = []
    if rng.random() < 0.5:
        body_lines.append(_forwarded_header(rng, supplier))

    body_lines.append(f"Invoice {invoice_number} from {supplier}")
    body_lines.append(f"Invoice date: {_fmt_date(rng, invoice_date)}")
    if due_date is not None:
        body_lines.append(f"Due: {_fmt_date(rng, due_date)}")
    if po_number is not None:
        body_lines.append(f"PO number: {po_number}")
    body_lines.append(f"Currency: {currency}")
    body_lines.append("")
    body_lines.append(_render_line_items(rng, items, symbol))
    body_lines.append("")
    body_lines.append(f"Subtotal: {_fmt_money(subtotal, symbol)}")
    if vat_stated:
        body_lines.append(f"VAT (20%): {_fmt_money(vat, symbol)}")
    body_lines.append(f"Total due: {_fmt_money(total, symbol)}")

    email_text = "\n".join(body_lines)
    if rng.random() < 0.5:
        email_text += _signature(rng, supplier)

    invoice = {
        "supplier_name": supplier,
        "invoice_number": invoice_number,
        "invoice_date": invoice_date.isoformat(),
        "due_date": due_date.isoformat() if due_date else None,
        "currency": currency,
        "line_items": [
            {
                "description": item["description"],
                "quantity": str(item["quantity"]),
                "unit_price": str(item["unit_price"]),
            }
            for item in items
        ],
        "subtotal": str(subtotal),
        "vat": str(vat),
        "total": str(total),
        "po_number": po_number,
    }

    return {
        "case_id": f"case-{index:03d}",
        "invoice": invoice,
        "email_text": email_text,
    }


def generate_cases(seed: int = SEED, n: int = N_CASES) -> list[dict]:
    rng = random.Random(seed)
    return [_build_case(rng, i) for i in range(n)]


def main() -> None:
    cases = generate_cases()
    out_path = Path(__file__).resolve().parents[2] / "data" / "cases.jsonl"
    with out_path.open("w") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")
    print(f"wrote {len(cases)} cases to {out_path}")


if __name__ == "__main__":
    main()
