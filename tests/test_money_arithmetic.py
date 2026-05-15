import pytest
from decimal import Decimal

from app.money import (
    CurrencyMismatchError,
    Money,
    add,
    multiply,
    percentage,
    subtract,
)


def test_add_same_currency():
    a = Money(Decimal("1.50"), "USD")
    b = Money(Decimal("2.25"), "USD")
    assert add(a, b) == Money(Decimal("3.75"), "USD")


def test_add_different_currency_raises():
    with pytest.raises(CurrencyMismatchError):
        add(Money(Decimal("1.00"), "USD"), Money(Decimal("1.00"), "EUR"))


def test_subtract_same_currency():
    a = Money(Decimal("5.00"), "USD")
    b = Money(Decimal("1.25"), "USD")
    assert subtract(a, b) == Money(Decimal("3.75"), "USD")


def test_subtract_different_currency_raises():
    with pytest.raises(CurrencyMismatchError):
        subtract(Money(Decimal("1.00"), "USD"), Money(Decimal("1.00"), "EUR"))


def test_subtract_to_negative():
    a = Money(Decimal("1.00"), "USD")
    b = Money(Decimal("5.00"), "USD")
    assert subtract(a, b) == Money(Decimal("-4.00"), "USD")


def test_multiply_basic():
    assert multiply(Money(Decimal("1.00"), "USD"), Decimal("0.1")) == Money(
        Decimal("0.10"), "USD"
    )


def test_multiply_rounds_banker_down_on_even_neighbor():
    # 0.05 * 0.5 = 0.025; HALF_EVEN -> 0.02 (preceding digit 2 is even)
    assert multiply(Money(Decimal("0.05"), "USD"), Decimal("0.5")) == Money(
        Decimal("0.02"), "USD"
    )


def test_multiply_accepts_float():
    assert multiply(Money(Decimal("10.00"), "USD"), 2) == Money(Decimal("20.00"), "USD")


def test_percentage_ten_percent_of_hundred():
    assert percentage(Money(Decimal("100.00"), "USD"), 10) == Money(
        Decimal("10.00"), "USD"
    )


def test_percentage_fractional():
    assert percentage(Money(Decimal("100.00"), "USD"), Decimal("8.25")) == Money(
        Decimal("8.25"), "USD"
    )


def test_percentage_rounds_to_minor_units():
    # 1/3 of $1.00 = $0.333... -> $0.33
    result = percentage(Money(Decimal("1.00"), "USD"), Decimal("100") / Decimal("3"))
    assert result == Money(Decimal("0.33"), "USD")
