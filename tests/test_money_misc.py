import pytest
from decimal import Decimal

from app.money import (
    CurrencyMismatchError,
    Money,
    compare,
    is_zero,
    negate,
)


def test_negate_positive():
    assert negate(Money(Decimal("1.00"), "USD")) == Money(Decimal("-1.00"), "USD")


def test_negate_negative():
    assert negate(Money(Decimal("-1.00"), "USD")) == Money(Decimal("1.00"), "USD")


def test_negate_zero():
    assert negate(Money(Decimal("0"), "USD")) == Money(Decimal("0"), "USD")


def test_negate_preserves_currency():
    assert negate(Money(Decimal("1.00"), "EUR")).currency == "EUR"


def test_is_zero_true():
    assert is_zero(Money(Decimal("0"), "USD"))
    assert is_zero(Money(Decimal("0.00"), "USD"))


def test_is_zero_false():
    assert not is_zero(Money(Decimal("0.01"), "USD"))
    assert not is_zero(Money(Decimal("-0.01"), "USD"))


def test_compare_less():
    assert (
        compare(Money(Decimal("1.00"), "USD"), Money(Decimal("2.00"), "USD")) == -1
    )


def test_compare_equal():
    assert compare(Money(Decimal("1.00"), "USD"), Money(Decimal("1.00"), "USD")) == 0


def test_compare_greater():
    assert compare(Money(Decimal("2.00"), "USD"), Money(Decimal("1.00"), "USD")) == 1


def test_compare_mismatched_currency_raises():
    with pytest.raises(CurrencyMismatchError):
        compare(Money(Decimal("1.00"), "USD"), Money(Decimal("1.00"), "EUR"))
