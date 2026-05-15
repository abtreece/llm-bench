import pytest
from decimal import Decimal

from app.money import Money, allocate, major_to_minor


def test_allocate_even_split():
    parts = allocate(Money(Decimal("3.00"), "USD"), [1, 1, 1])
    assert [p.amount for p in parts] == [
        Decimal("1.00"),
        Decimal("1.00"),
        Decimal("1.00"),
    ]


def test_allocate_remainder_distributed_left_to_right():
    # 5 cents / 3 = 1 each + 2 remainder -> [2, 2, 1]
    parts = allocate(Money(Decimal("0.05"), "USD"), [1, 1, 1])
    assert [p.amount for p in parts] == [
        Decimal("0.02"),
        Decimal("0.02"),
        Decimal("0.01"),
    ]


def test_allocate_preserves_total():
    m = Money(Decimal("1.00"), "USD")
    parts = allocate(m, [1, 2, 3])
    total = sum(major_to_minor(p) for p in parts)
    assert total == major_to_minor(m)


def test_allocate_zero_ratio_buckets_get_nothing():
    parts = allocate(Money(Decimal("1.00"), "USD"), [1, 0, 1])
    assert [p.amount for p in parts] == [
        Decimal("0.50"),
        Decimal("0.00"),
        Decimal("0.50"),
    ]


def test_allocate_empty_ratios_raises():
    with pytest.raises(ValueError):
        allocate(Money(Decimal("1.00"), "USD"), [])


def test_allocate_all_zero_ratios_raises():
    with pytest.raises(ValueError):
        allocate(Money(Decimal("1.00"), "USD"), [0, 0])


def test_allocate_negative_ratio_raises():
    with pytest.raises(ValueError):
        allocate(Money(Decimal("1.00"), "USD"), [1, -1])


def test_allocate_negative_amount():
    parts = allocate(Money(Decimal("-0.05"), "USD"), [1, 1, 1])
    assert [p.amount for p in parts] == [
        Decimal("-0.02"),
        Decimal("-0.02"),
        Decimal("-0.01"),
    ]


def test_allocate_jpy_no_minor_units():
    # 100 yen / 3 = 33 each + 1 remainder -> [34, 33, 33]
    parts = allocate(Money(Decimal("100"), "JPY"), [1, 1, 1])
    assert [p.amount for p in parts] == [
        Decimal("34"),
        Decimal("33"),
        Decimal("33"),
    ]


def test_allocate_unequal_ratios():
    # $1.00 split 1:2:3 -> 16 + 33 + 50 = 99 minor units + 1 remainder -> [17, 33, 50]
    parts = allocate(Money(Decimal("1.00"), "USD"), [1, 2, 3])
    assert [p.amount for p in parts] == [
        Decimal("0.17"),
        Decimal("0.33"),
        Decimal("0.50"),
    ]
