from decimal import Decimal

from app.money import Money, major_to_minor, minor_to_major


def test_usd_minor_to_major():
    assert minor_to_major(12345, "USD") == Money(Decimal("123.45"), "USD")


def test_jpy_minor_to_major_no_decimal_shift():
    assert minor_to_major(12345, "JPY") == Money(Decimal("12345"), "JPY")


def test_kwd_minor_to_major_three_decimals():
    assert minor_to_major(12345, "KWD") == Money(Decimal("12.345"), "KWD")


def test_negative_minor_to_major():
    assert minor_to_major(-100, "USD") == Money(Decimal("-1.00"), "USD")


def test_zero_minor_to_major():
    assert minor_to_major(0, "USD") == Money(Decimal("0.00"), "USD")


def test_major_to_minor_usd():
    assert major_to_minor(Money(Decimal("123.45"), "USD")) == 12345


def test_major_to_minor_jpy():
    assert major_to_minor(Money(Decimal("12345"), "JPY")) == 12345


def test_major_to_minor_kwd():
    assert major_to_minor(Money(Decimal("12.345"), "KWD")) == 12345


def test_round_trip_usd():
    for n in (1, 100, 12345, -50, 0):
        assert major_to_minor(minor_to_major(n, "USD")) == n


def test_round_trip_jpy():
    for n in (1, 100, 12345, -50, 0):
        assert major_to_minor(minor_to_major(n, "JPY")) == n


def test_round_trip_kwd():
    for n in (1, 100, 12345, -50, 0):
        assert major_to_minor(minor_to_major(n, "KWD")) == n
