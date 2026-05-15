"""Money: currency-aware arithmetic, parsing, formatting.

A small, self-contained payments-adjacent library used as the test target
for the local LLM coding benchmark. Single file, no third-party deps.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Iterable


CURRENCIES: dict[str, int] = {
    "USD": 2, "EUR": 2, "GBP": 2, "CAD": 2, "AUD": 2,
    "JPY": 0, "KRW": 0,
    "KWD": 3, "BHD": 3,
}

_SYMBOLS: dict[str, str] = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}


class MoneyError(Exception):
    """Base class for all money-related errors."""


class UnknownCurrencyError(MoneyError):
    """Raised when a currency code is not in the CURRENCIES table."""


class CurrencyMismatchError(MoneyError):
    """Raised when an operation requires two Money values to share a currency."""


class MoneyParseError(MoneyError):
    """Raised when a string cannot be parsed into a Money value."""


@dataclass(frozen=True)
class Money:
    amount: Decimal
    currency: str

    def __post_init__(self) -> None:
        if self.currency not in CURRENCIES:
            raise UnknownCurrencyError(self.currency)
        if not isinstance(self.amount, Decimal):
            object.__setattr__(self, "amount", Decimal(str(self.amount)))


def minor_units(currency: str) -> int:
    """Return the number of decimal places used by the currency's minor unit."""
    if currency not in CURRENCIES:
        raise UnknownCurrencyError(currency)
    return CURRENCIES[currency]


def _quantum(currency: str) -> Decimal:
    digits = minor_units(currency)
    if digits == 0:
        return Decimal("1")
    return Decimal("1").scaleb(-digits)


def minor_to_major(minor: int, currency: str) -> Money:
    """Convert an integer count of minor units to a Money value.

    minor_to_major(12345, "USD") -> Money(Decimal("123.45"), "USD")
    minor_to_major(12345, "JPY") -> Money(Decimal("12345"),  "JPY")
    minor_to_major(12345, "KWD") -> Money(Decimal("12.345"), "KWD")
    """
    if not isinstance(minor, int):
        raise TypeError("minor must be int")
    digits = minor_units(currency)
    amount = (Decimal(minor) * Decimal("1").scaleb(-digits)).quantize(_quantum(currency))
    return Money(amount, currency)


def major_to_minor(m: Money) -> int:
    """Convert a Money value to its integer minor-unit count."""
    digits = minor_units(m.currency)
    scaled = (m.amount * Decimal("1").scaleb(digits)).quantize(
        Decimal("1"), rounding=ROUND_HALF_EVEN
    )
    return int(scaled)


def round_to_minor_units(m: Money, rounding: str = ROUND_HALF_EVEN) -> Money:
    """Quantize the amount to the currency's minor units using banker's rounding by default."""
    return Money(m.amount.quantize(_quantum(m.currency), rounding=rounding), m.currency)


def _require_same_currency(a: Money, b: Money) -> None:
    if a.currency != b.currency:
        raise CurrencyMismatchError(f"{a.currency} vs {b.currency}")


def add(a: Money, b: Money) -> Money:
    """Add two Money values of the same currency."""
    _require_same_currency(a, b)
    return round_to_minor_units(Money(a.amount + b.amount, a.currency))


def subtract(a: Money, b: Money) -> Money:
    """Return a - b (same currency required)."""
    _require_same_currency(a, b)
    return round_to_minor_units(Money(a.amount - b.amount, a.currency))


def multiply(m: Money, factor) -> Money:
    """Multiply a Money value by a scalar; result is quantized to the minor unit."""
    f = factor if isinstance(factor, Decimal) else Decimal(str(factor))
    return round_to_minor_units(Money(m.amount * f, m.currency))


def percentage(m: Money, pct) -> Money:
    """Return pct% of m. pct=10 means 10%."""
    p = pct if isinstance(pct, Decimal) else Decimal(str(pct))
    return round_to_minor_units(Money(m.amount * p / Decimal(100), m.currency))


def negate(m: Money) -> Money:
    """Return -m, preserving the currency."""
    return Money(-m.amount, m.currency)


def is_zero(m: Money) -> bool:
    """True iff the amount is numerically zero."""
    return m.amount == 0


def compare(a: Money, b: Money) -> int:
    """Three-way compare. Returns -1, 0, or 1. Currency mismatch raises."""
    _require_same_currency(a, b)
    if a.amount < b.amount:
        return -1
    if a.amount > b.amount:
        return 1
    return 0


def allocate(m: Money, ratios: Iterable[int]) -> list[Money]:
    """Split m into parts proportional to ratios, distributing any remainder
    one minor unit at a time to earlier buckets so the total is preserved.

    allocate(Money(0.05, USD), [1, 1, 1]) -> [0.02, 0.02, 0.01]
    """
    ratios = list(ratios)
    if not ratios or any(r < 0 for r in ratios):
        raise ValueError("ratios must be a non-empty list of non-negative ints")
    total_ratio = sum(ratios)
    if total_ratio == 0:
        raise ValueError("ratios must sum to > 0")
    total_minor = major_to_minor(m)
    sign = 1 if total_minor >= 0 else -1
    total_abs = abs(total_minor)
    shares = [total_abs * r // total_ratio for r in ratios]
    remainder = total_abs - sum(shares)
    for i in range(remainder):
        shares[i % len(shares)] += 1
    return [minor_to_major(sign * s, m.currency) for s in shares]


_AMOUNT_RE = re.compile(r"^\s*(-?)\s*([\d,]+)(?:\.(\d+))?\s*$")


def parse_money(s: str, currency: str) -> Money:
    """Parse a numeric string into Money under the given currency.

    Accepts optional thousands separators (commas), a leading minus,
    and an optional fractional part. Result is rounded to the currency's
    minor units. Raises MoneyParseError on failure.
    """
    if not isinstance(s, str):
        raise MoneyParseError(f"expected str, got {type(s).__name__}")
    match = _AMOUNT_RE.match(s)
    if not match:
        raise MoneyParseError(f"could not parse {s!r}")
    sign = match.group(1)
    whole = match.group(2).replace(",", "")
    frac = match.group(3) or ""
    if not whole:
        raise MoneyParseError(f"missing whole part in {s!r}")
    raw = Decimal(f"{sign}{whole}.{frac}") if frac else Decimal(f"{sign}{whole}")
    return round_to_minor_units(Money(raw, currency))


_TAG_RE = re.compile(r"^\s*([A-Z]{3})\s+(.+?)\s*$|^\s*(.+?)\s+([A-Z]{3})\s*$")


def parse_money_str(s: str) -> Money:
    """Parse "USD 1,234.56" or "1,234.56 USD" into Money."""
    if not isinstance(s, str):
        raise MoneyParseError(f"expected str, got {type(s).__name__}")
    match = _TAG_RE.match(s)
    if not match:
        raise MoneyParseError(f"could not parse {s!r}")
    if match.group(1):
        currency, amount = match.group(1), match.group(2)
    else:
        amount, currency = match.group(3), match.group(4)
    return parse_money(amount, currency)


def format_amount(m: Money) -> str:
    """Numeric portion only, fixed to the currency's minor units.

    format_amount(Money(5, "USD")) -> "5.00"
    format_amount(Money(5, "JPY")) -> "5"
    format_amount(Money(5, "KWD")) -> "5.000"
    """
    rounded = round_to_minor_units(m)
    digits = minor_units(m.currency)
    if digits == 0:
        return f"{int(rounded.amount)}"
    return f"{rounded.amount:.{digits}f}"


def format_money(m: Money) -> str:
    """Format with thousands separator and currency symbol (or 3-letter code).

    format_money(Money(1234.56, "USD")) -> "$1,234.56"
    format_money(Money(1234,    "JPY")) -> "¥1,234"
    format_money(Money(1234.567,"KWD")) -> "KWD 1,234.567"
    format_money(Money(-1.50,   "USD")) -> "-$1.50"
    """
    rounded = round_to_minor_units(m)
    digits = minor_units(m.currency)
    sign = "-" if rounded.amount < 0 else ""
    absamt = abs(rounded.amount)
    whole_int = int(absamt)
    if digits == 0:
        body = f"{whole_int:,}"
    else:
        frac = int((absamt - whole_int).scaleb(digits))
        body = f"{whole_int:,}.{frac:0{digits}d}"
    symbol = _SYMBOLS.get(m.currency)
    if symbol:
        return f"{sign}{symbol}{body}"
    return f"{sign}{m.currency} {body}"
