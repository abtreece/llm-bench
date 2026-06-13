"""Transaction analytics over payment-processor CSV exports.

Data-analysis companion to app.money and the second test target for the
local LLM coding benchmark. Single file, no third-party deps.

CSV schema (header row required):
    txn_id    unique string id
    merchant  merchant name
    status    completed | refunded | failed
    currency  ISO 4217 code present in app.money.CURRENCIES
    amount    decimal string; may be empty when the processor reported
              no settled amount for the transaction
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable

from app.money import (
    Money,
    UnknownCurrencyError,
    add,
    minor_units,
    round_to_minor_units,
)


class AnalysisError(Exception):
    """Base class for all analysis-related errors."""


class MalformedRowError(AnalysisError):
    """Raised when a CSV row cannot be interpreted."""


@dataclass(frozen=True)
class Transaction:
    txn_id: str
    merchant: str
    status: str  # completed | refunded | failed
    currency: str
    amount: Decimal | None  # None when the processor reported no amount


_REQUIRED_NONEMPTY = ("txn_id", "merchant", "status", "currency")
_STATUSES = {"completed", "refunded", "failed"}


def load_transactions(path: str | Path) -> list[Transaction]:
    """Parse a transactions CSV into Transaction records.

    Empty amount cells become None. A missing field, an unknown status or
    currency, or an unparseable amount raises MalformedRowError.
    """
    txns: list[Transaction] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            # reader.line_num tracks physical lines, staying accurate
            # across quoted fields with embedded newlines.
            line = reader.line_num
            fields = {k: (v or "").strip() for k, v in raw.items() if k}
            missing = [f for f in _REQUIRED_NONEMPTY if not fields.get(f)]
            if missing:
                raise MalformedRowError(f"line {line}: missing {missing}")
            status = fields["status"].lower()
            if status not in _STATUSES:
                raise MalformedRowError(
                    f"line {line}: unknown status {fields['status']!r}"
                )
            currency = fields["currency"]
            try:
                minor_units(currency)
            except UnknownCurrencyError:
                raise MalformedRowError(
                    f"line {line}: unknown currency {currency!r}"
                )
            cell = fields.get("amount", "")
            if cell:
                try:
                    amount: Decimal | None = Decimal(cell)
                except InvalidOperation:
                    raise MalformedRowError(f"line {line}: bad amount {cell!r}")
            else:
                amount = None
            txns.append(
                Transaction(
                    txn_id=fields["txn_id"],
                    merchant=fields["merchant"],
                    status=status,
                    currency=currency,
                    amount=amount,
                )
            )
    return txns


def revenue_by_currency(txns: Iterable[Transaction]) -> dict[str, Money]:
    """Total settled revenue per currency.

    Only completed transactions count toward revenue; refunded and failed
    transactions are excluded. Transactions with no amount are skipped.
    """
    totals: dict[str, Money] = {}
    for t in txns:
        if t.status != "completed" or t.amount is None:
            continue
        m = round_to_minor_units(Money(t.amount, t.currency))
        if t.currency in totals:
            totals[t.currency] = add(totals[t.currency], m)
        else:
            totals[t.currency] = m
    return totals


def average_order_value(txns: Iterable[Transaction], currency: str) -> Money | None:
    """Mean completed transaction amount for one currency.

    Transactions with no recorded amount are excluded from both the sum
    and the count. Returns None when no transaction qualifies.
    """
    amounts = [
        t.amount
        for t in txns
        if t.status == "completed" and t.currency == currency and t.amount is not None
    ]
    if not amounts:
        return None
    return round_to_minor_units(Money(sum(amounts) / len(amounts), currency))


def top_merchants(
    txns: Iterable[Transaction], n: int, currency: str
) -> list[tuple[str, Money]]:
    """Top n merchants by total completed revenue in one currency.

    Sorted by revenue descending; ties broken by merchant name ascending.
    Transactions with no amount are skipped. Raises ValueError if n < 1.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    totals: dict[str, Decimal] = {}
    for t in txns:
        if t.status != "completed" or t.currency != currency or t.amount is None:
            continue
        totals[t.merchant] = totals.get(t.merchant, Decimal("0")) + t.amount
    ranked = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        (name, round_to_minor_units(Money(amt, currency)))
        for name, amt in ranked[:n]
    ]
