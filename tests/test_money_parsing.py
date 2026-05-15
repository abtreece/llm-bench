import pytest
from decimal import Decimal

from app.money import Money, MoneyParseError, parse_money, parse_money_str


def test_parse_simple():
    assert parse_money("123.45", "USD") == Money(Decimal("123.45"), "USD")


def test_parse_with_thousands_separator():
    assert parse_money("1,234.56", "USD") == Money(Decimal("1234.56"), "USD")


def test_parse_negative():
    assert parse_money("-1.00", "USD") == Money(Decimal("-1.00"), "USD")


def test_parse_no_decimal():
    assert parse_money("100", "USD") == Money(Decimal("100.00"), "USD")


def test_parse_jpy_keeps_no_decimal():
    assert parse_money("12345", "JPY") == Money(Decimal("12345"), "JPY")


def test_parse_kwd_three_decimal():
    assert parse_money("12.345", "KWD") == Money(Decimal("12.345"), "KWD")


def test_parse_rounds_to_minor_units():
    # USD has 2 decimals; 1.005 with HALF_EVEN rounds to 1.00 (0 is even)
    assert parse_money("1.005", "USD") == Money(Decimal("1.00"), "USD")


def test_parse_negative_rounds_banker():
    # -1.005 with HALF_EVEN -> -1.00 (0 is even)
    assert parse_money("-1.005", "USD") == Money(Decimal("-1.00"), "USD")


def test_parse_invalid_raises():
    with pytest.raises(MoneyParseError):
        parse_money("not a number", "USD")


def test_parse_non_string_raises():
    with pytest.raises(MoneyParseError):
        parse_money(123, "USD")  # type: ignore[arg-type]


def test_parse_money_str_prefix_form():
    assert parse_money_str("USD 1,234.56") == Money(Decimal("1234.56"), "USD")


def test_parse_money_str_suffix_form():
    assert parse_money_str("1,234.56 USD") == Money(Decimal("1234.56"), "USD")


def test_parse_money_str_invalid_raises():
    with pytest.raises(MoneyParseError):
        parse_money_str("just numbers 123")
