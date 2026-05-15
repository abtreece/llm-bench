from decimal import Decimal

from app.money import Money, format_amount, format_money


def test_format_amount_usd():
    assert format_amount(Money(Decimal("5"), "USD")) == "5.00"


def test_format_amount_jpy_no_decimal():
    assert format_amount(Money(Decimal("5"), "JPY")) == "5"


def test_format_amount_kwd_three_decimal():
    assert format_amount(Money(Decimal("5"), "KWD")) == "5.000"


def test_format_amount_has_no_thousands_separator():
    assert format_amount(Money(Decimal("1234.56"), "USD")) == "1234.56"


def test_format_money_usd_uses_dollar_symbol():
    assert format_money(Money(Decimal("1234.56"), "USD")) == "$1,234.56"


def test_format_money_jpy_no_decimal():
    assert format_money(Money(Decimal("1234"), "JPY")) == "¥1,234"


def test_format_money_unknown_symbol_falls_back_to_code():
    assert format_money(Money(Decimal("1234.567"), "KWD")) == "KWD 1,234.567"


def test_format_money_negative():
    assert format_money(Money(Decimal("-1.50"), "USD")) == "-$1.50"


def test_format_money_zero():
    assert format_money(Money(Decimal("0"), "USD")) == "$0.00"


def test_format_money_eur_uses_symbol():
    assert format_money(Money(Decimal("99.99"), "EUR")) == "€99.99"
