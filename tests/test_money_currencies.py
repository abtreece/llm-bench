import pytest
from decimal import Decimal
from dataclasses import FrozenInstanceError

from app.money import (
    CURRENCIES,
    Money,
    UnknownCurrencyError,
    minor_units,
)


def test_usd_has_two_minor_units():
    assert minor_units("USD") == 2


def test_jpy_has_zero_minor_units():
    assert minor_units("JPY") == 0


def test_kwd_has_three_minor_units():
    assert minor_units("KWD") == 3


def test_unknown_currency_minor_units_raises():
    with pytest.raises(UnknownCurrencyError):
        minor_units("ZZZ")


def test_money_construction_validates_currency():
    with pytest.raises(UnknownCurrencyError):
        Money(Decimal("1.00"), "ZZZ")


def test_money_is_frozen():
    m = Money(Decimal("1.00"), "USD")
    with pytest.raises(FrozenInstanceError):
        m.amount = Decimal("2.00")  # type: ignore[misc]


def test_money_coerces_non_decimal_amount():
    m = Money(123, "USD")
    assert m.amount == Decimal("123")
    assert isinstance(m.amount, Decimal)


def test_currencies_table_includes_majors():
    for code in ("USD", "EUR", "GBP", "JPY", "KWD"):
        assert code in CURRENCIES
