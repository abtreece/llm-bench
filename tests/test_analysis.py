from decimal import Decimal
from pathlib import Path

import pytest

from app.analysis import (
    MalformedRowError,
    Transaction,
    average_order_value,
    load_transactions,
    revenue_by_currency,
    top_merchants,
)
from app.money import Money

DATA = Path(__file__).resolve().parent.parent / "app" / "data" / "transactions.csv"


def txn(merchant="Acme", status="completed", currency="USD", amount="10.00", txn_id="t1"):
    return Transaction(
        txn_id=txn_id,
        merchant=merchant,
        status=status,
        currency=currency,
        amount=Decimal(amount) if amount is not None else None,
    )


class TestLoadTransactions:
    def test_loads_full_fixture(self):
        txns = load_transactions(DATA)
        assert len(txns) == 20
        assert txns[0] == Transaction("t001", "Acme", "completed", "USD", Decimal("120.00"))

    def test_empty_amount_becomes_none(self):
        txns = {t.txn_id: t for t in load_transactions(DATA)}
        assert txns["t005"].amount is None
        assert txns["t019"].amount is None

    def test_unknown_status_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("txn_id,merchant,status,currency,amount\nt1,Acme,pending,USD,1.00\n")
        with pytest.raises(MalformedRowError, match="unknown status"):
            load_transactions(p)

    def test_unparseable_amount_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("txn_id,merchant,status,currency,amount\nt1,Acme,completed,USD,abc\n")
        with pytest.raises(MalformedRowError, match="bad amount"):
            load_transactions(p)

    def test_missing_required_field_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("txn_id,merchant,status,currency,amount\nt1,,completed,USD,1.00\n")
        with pytest.raises(MalformedRowError, match="missing"):
            load_transactions(p)

    def test_whitespace_only_field_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("txn_id,merchant,status,currency,amount\nt1,   ,completed,USD,1.00\n")
        with pytest.raises(MalformedRowError, match="missing"):
            load_transactions(p)

    def test_unknown_currency_raises_at_load(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("txn_id,merchant,status,currency,amount\nt1,Acme,completed,XXX,1.00\n")
        with pytest.raises(MalformedRowError, match="unknown currency"):
            load_transactions(p)


class TestRevenueByCurrency:
    def test_fixture_totals(self):
        totals = revenue_by_currency(load_transactions(DATA))
        assert totals == {
            "USD": Money(Decimal("751.48"), "USD"),
            "EUR": Money(Decimal("539.85"), "EUR"),
            "JPY": Money(Decimal("6200"), "JPY"),
        }

    def test_refunded_and_failed_excluded(self):
        txns = [
            txn(amount="100.00"),
            txn(status="refunded", amount="40.00", txn_id="t2"),
            txn(status="failed", amount=None, txn_id="t3"),
        ]
        assert revenue_by_currency(txns) == {"USD": Money(Decimal("100.00"), "USD")}

    def test_empty_input(self):
        assert revenue_by_currency([]) == {}


class TestAverageOrderValue:
    def test_fixture_usd_average(self):
        avg = average_order_value(load_transactions(DATA), "USD")
        assert avg == Money(Decimal("107.35"), "USD")

    def test_missing_amounts_excluded_from_count(self):
        txns = [
            txn(amount="10.00"),
            txn(amount="20.00", txn_id="t2"),
            txn(amount=None, txn_id="t3"),
        ]
        assert average_order_value(txns, "USD") == Money(Decimal("15.00"), "USD")

    def test_no_qualifying_transactions_returns_none(self):
        assert average_order_value(load_transactions(DATA), "GBP") is None


class TestTopMerchants:
    def test_fixture_ranking_with_tie_break(self):
        txns = load_transactions(DATA)
        assert top_merchants(txns, 3, "USD") == [
            ("Acme", Money(Decimal("250.00"), "USD")),
            ("Globex", Money(Decimal("200.25"), "USD")),
            ("Hooli", Money(Decimal("200.25"), "USD")),
        ]

    def test_n_truncates(self):
        txns = load_transactions(DATA)
        assert [name for name, _ in top_merchants(txns, 1, "USD")] == ["Acme"]

    def test_only_requested_currency_counts(self):
        txns = [txn(amount="5.00"), txn(currency="EUR", amount="500.00", txn_id="t2")]
        assert top_merchants(txns, 5, "USD") == [("Acme", Money(Decimal("5.00"), "USD"))]

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            top_merchants([], 0, "USD")
