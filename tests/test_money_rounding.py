from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP

from app.money import Money, round_to_minor_units


def test_default_is_banker_rounding_even_down():
    # 0.005 -> 0.00 (preceding digit 0 is even, .5 rounds toward even)
    assert round_to_minor_units(Money(Decimal("0.005"), "USD")).amount == Decimal("0.00")


def test_banker_rounds_up_when_odd_neighbor():
    # 0.015 -> 0.02 (preceding digit 1 is odd, .5 rounds up to even 2)
    assert round_to_minor_units(Money(Decimal("0.015"), "USD")).amount == Decimal("0.02")


def test_banker_rounds_down_when_even_neighbor():
    # 0.025 -> 0.02 (preceding digit 2 is even)
    assert round_to_minor_units(Money(Decimal("0.025"), "USD")).amount == Decimal("0.02")


def test_explicit_half_up_differs_from_banker():
    # Same input where modes disagree.
    m = Money(Decimal("0.025"), "USD")
    assert round_to_minor_units(m, ROUND_HALF_UP).amount == Decimal("0.03")


def test_round_down_truncates():
    assert round_to_minor_units(
        Money(Decimal("0.029"), "USD"), ROUND_DOWN
    ).amount == Decimal("0.02")


def test_round_jpy_odd_neighbor():
    # 123.5 -> 124 (123 odd, .5 rounds up to even 124)
    assert round_to_minor_units(Money(Decimal("123.5"), "JPY")).amount == Decimal("124")


def test_round_jpy_even_neighbor():
    # 124.5 -> 124 (124 even, .5 rounds toward even)
    assert round_to_minor_units(Money(Decimal("124.5"), "JPY")).amount == Decimal("124")


def test_round_kwd_three_decimal_places():
    # 1.0005 -> 1.000 (preceding 0 even, .5 rounds toward even)
    assert round_to_minor_units(
        Money(Decimal("1.0005"), "KWD")
    ).amount == Decimal("1.000")


def test_round_preserves_currency():
    assert round_to_minor_units(Money(Decimal("1.005"), "USD")).currency == "USD"


def test_round_negative_banker():
    # -1.005 -> -1.00 (preceding 0 even, .5 rounds toward even)
    assert round_to_minor_units(
        Money(Decimal("-1.005"), "USD")
    ).amount == Decimal("-1.00")
