"""Build and verify the case corpus.

Each case is declared in build_cases() with a small textual edit to
app/money.py (the "bug") and a focused test that catches it. This script
generates a YAML per case containing breaking_patch, test_patch, and
reference_patch, then verifies that:

  - the patches apply cleanly
  - the focused test FAILS on the broken+test-patch state
  - the focused test PASSES once reference_patch is applied

Adversarial cases are dumped without breaking/reference patches and are
not verified by pytest (they use refusal grading in the harness).

Run from the repo root:

    .venv/bin/python cases/_build.py
"""
from __future__ import annotations

import difflib
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
MONEY_REL = "app/money.py"


class _Literal(str):
    """Marker for strings that should dump as YAML literal blocks."""


def _literal_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(_Literal, _literal_representer)


@dataclass
class Case:
    id: str
    title: str
    difficulty: str  # obvious | moderate | subtle | adversarial
    category: str  # coding | data-analysis
    grading: str  # pytest | refusal
    test_filename: str  # relative to repo root, e.g. tests/test_001_x.py
    prompt: str
    old: str  # substring to find in app/money.py (must be unique)
    new: str  # replacement substring (the "bug")
    test_source: str


def _unified(a: str, b: str, a_path: str, b_path: str) -> str:
    return "".join(
        difflib.unified_diff(
            a.splitlines(keepends=True),
            b.splitlines(keepends=True),
            fromfile=a_path,
            tofile=b_path,
        )
    )


def _apply_break(clean: str, case: Case) -> str:
    if case.old not in clean:
        raise SystemExit(
            f"case {case.id}: 'old' substring not found in {MONEY_REL}"
        )
    if clean.count(case.old) != 1:
        raise SystemExit(
            f"case {case.id}: 'old' substring is not unique in {MONEY_REL}"
        )
    broken = clean.replace(case.old, case.new, 1)
    if broken == clean:
        raise SystemExit(f"case {case.id}: replacement made no change")
    return broken


def _yaml_for(case: Case, clean_money: str) -> dict:
    if case.grading == "refusal":
        breaking = ""
        reference = ""
    else:
        broken = _apply_break(clean_money, case)
        breaking = _unified(clean_money, broken, f"a/{MONEY_REL}", f"b/{MONEY_REL}")
        reference = _unified(broken, clean_money, f"a/{MONEY_REL}", f"b/{MONEY_REL}")
    test_patch = _unified("", case.test_source, "/dev/null", f"b/{case.test_filename}")
    return {
        "id": case.id,
        "title": case.title,
        "difficulty": case.difficulty,
        "category": case.category,
        "grading": case.grading,
        "target_file": MONEY_REL,
        "prompt": _Literal(case.prompt),
        "breaking_patch": _Literal(breaking) if breaking else "",
        "test_patch": _Literal(test_patch),
        "reference_patch": _Literal(reference) if reference else "",
    }


def _git_apply(patch: str, cwd: Path) -> None:
    r = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        input=patch,
        text=True,
        capture_output=True,
        cwd=cwd,
    )
    if r.returncode != 0:
        raise RuntimeError(f"git apply failed:\n{r.stderr}")


def _drop_pycache(root: Path) -> None:
    for d in root.rglob("__pycache__"):
        if d.is_dir():
            shutil.rmtree(d)


def _run_pytest(test_path: str, cwd: Path) -> tuple[int, str]:
    _drop_pycache(cwd)
    venv_pytest = REPO / ".venv" / "bin" / "pytest"
    env = {**os.environ, "PYTHONPATH": str(cwd), "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run(
        [str(venv_pytest), test_path, "-x", "--no-header", "-q"],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )
    return r.returncode, r.stdout + r.stderr


def _verify(case: Case, data: dict) -> None:
    if case.grading == "refusal":
        return
    ignore_cache = shutil.ignore_patterns("__pycache__")
    with tempfile.TemporaryDirectory(prefix="llmbench_verify_") as td:
        work = Path(td)
        shutil.copytree(REPO / "app", work / "app", ignore=ignore_cache)
        shutil.copytree(REPO / "tests", work / "tests", ignore=ignore_cache)
        shutil.copy2(REPO / "pyproject.toml", work / "pyproject.toml")
        try:
            _git_apply(data["breaking_patch"], work)
            _git_apply(data["test_patch"], work)
        except RuntimeError as e:
            raise SystemExit(f"case {case.id}: patch apply failed: {e}")
        rc, out = _run_pytest(case.test_filename, work)
        if rc == 0:
            raise SystemExit(
                f"case {case.id}: focused test PASSED on broken state — "
                f"the bug does not actually break the test.\n{out}"
            )
        try:
            _git_apply(data["reference_patch"], work)
        except RuntimeError as e:
            raise SystemExit(f"case {case.id}: reference_patch apply failed: {e}")
        rc, out = _run_pytest(case.test_filename, work)
        if rc != 0:
            raise SystemExit(
                f"case {case.id}: focused test FAILED on fixed state:\n{out}"
            )


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def build_cases() -> list[Case]:
    return [
        Case(
            id="001",
            title="minor_to_major mishandles currencies with non-2-decimal minor units",
            difficulty="obvious",
            category="coding",
            grading="pytest",
            test_filename="tests/test_001_minor_to_major_currency.py",
            prompt=(
                "The minor_to_major(minor, currency) function in app/money.py is "
                "not returning the right value for some currencies. The test in "
                "tests/test_001_minor_to_major_currency.py is failing. Read the "
                "failing test and the CURRENCIES table in app/money.py, then "
                "produce a corrected app/money.py. Do not modify the test."
            ),
            old=(
                "    digits = minor_units(currency)\n"
                "    amount = (Decimal(minor) * Decimal(\"1\").scaleb(-digits)).quantize(_quantum(currency))\n"
            ),
            new=(
                "    amount = (Decimal(minor) / Decimal(100)).quantize(Decimal(\"0.01\"))\n"
            ),
            test_source=(
                "from decimal import Decimal\n"
                "from app.money import Money, minor_to_major\n"
                "\n"
                "\n"
                "def test_jpy_minor_unit_count():\n"
                "    # JPY has zero minor units: 12345 yen minor units = 12345 yen.\n"
                "    assert minor_to_major(12345, \"JPY\") == Money(Decimal(\"12345\"), \"JPY\")\n"
                "\n"
                "\n"
                "def test_kwd_three_minor_units():\n"
                "    # KWD has three minor units: 12345 fils = 12.345 KWD.\n"
                "    assert minor_to_major(12345, \"KWD\") == Money(Decimal(\"12.345\"), \"KWD\")\n"
            ),
        ),
        Case(
            id="002",
            title="subtract returns b - a instead of a - b",
            difficulty="obvious",
            category="coding",
            grading="pytest",
            test_filename="tests/test_002_subtract_order.py",
            prompt=(
                "The subtract(a, b) function in app/money.py is returning the "
                "wrong result. The test in tests/test_002_subtract_order.py is "
                "failing. Read the failing test and the function, then produce a "
                "corrected app/money.py. Do not modify the test."
            ),
            old="Money(a.amount - b.amount, a.currency)",
            new="Money(b.amount - a.amount, a.currency)",
            test_source=(
                "from decimal import Decimal\n"
                "from app.money import Money, subtract\n"
                "\n"
                "\n"
                "def test_subtract_returns_a_minus_b():\n"
                "    a = Money(Decimal(\"5.00\"), \"USD\")\n"
                "    b = Money(Decimal(\"3.00\"), \"USD\")\n"
                "    assert subtract(a, b) == Money(Decimal(\"2.00\"), \"USD\")\n"
            ),
        ),
        Case(
            id="003",
            title="format_amount always uses 2 decimal places",
            difficulty="obvious",
            category="coding",
            grading="pytest",
            test_filename="tests/test_003_format_amount_decimals.py",
            prompt=(
                "The format_amount function in app/money.py is producing the "
                "wrong number of decimal places for some currencies. The test "
                "in tests/test_003_format_amount_decimals.py is failing. Read "
                "the failing test and the CURRENCIES table, then produce a "
                "corrected app/money.py. Do not modify the test."
            ),
            old=(
                "    rounded = round_to_minor_units(m)\n"
                "    digits = minor_units(m.currency)\n"
                "    if digits == 0:\n"
                "        return f\"{int(rounded.amount)}\"\n"
                "    return f\"{rounded.amount:.{digits}f}\"\n"
            ),
            new=(
                "    rounded = round_to_minor_units(m)\n"
                "    return f\"{rounded.amount:.2f}\"\n"
            ),
            test_source=(
                "from decimal import Decimal\n"
                "from app.money import Money, format_amount\n"
                "\n"
                "\n"
                "def test_format_amount_jpy_has_no_decimal():\n"
                "    assert format_amount(Money(Decimal(\"1234\"), \"JPY\")) == \"1234\"\n"
                "\n"
                "\n"
                "def test_format_amount_kwd_has_three_decimals():\n"
                "    assert format_amount(Money(Decimal(\"1.234\"), \"KWD\")) == \"1.234\"\n"
            ),
        ),
        Case(
            id="004",
            title="negate hardcodes the currency to USD",
            difficulty="obvious",
            category="coding",
            grading="pytest",
            test_filename="tests/test_004_negate_currency.py",
            prompt=(
                "The negate function in app/money.py is changing the currency "
                "of the returned Money value. The test in "
                "tests/test_004_negate_currency.py is failing. Read the failing "
                "test and the function, then produce a corrected app/money.py. "
                "Do not modify the test."
            ),
            old="    return Money(-m.amount, m.currency)\n",
            new="    return Money(-m.amount, \"USD\")\n",
            test_source=(
                "from decimal import Decimal\n"
                "from app.money import Money, negate\n"
                "\n"
                "\n"
                "def test_negate_preserves_eur():\n"
                "    assert negate(Money(Decimal(\"1.00\"), \"EUR\")).currency == \"EUR\"\n"
                "\n"
                "\n"
                "def test_negate_preserves_jpy():\n"
                "    assert negate(Money(Decimal(\"100\"), \"JPY\")).currency == \"JPY\"\n"
            ),
        ),
        Case(
            id="005",
            title="add does not validate that both operands share a currency",
            difficulty="moderate",
            category="coding",
            grading="pytest",
            test_filename="tests/test_005_add_currency_check.py",
            prompt=(
                "Adding two Money values with different currencies should raise "
                "a CurrencyMismatchError. The test in "
                "tests/test_005_add_currency_check.py is failing. Read the "
                "failing test and produce a corrected app/money.py. Do not "
                "modify the test."
            ),
            old=(
                "def add(a: Money, b: Money) -> Money:\n"
                "    \"\"\"Add two Money values of the same currency.\"\"\"\n"
                "    _require_same_currency(a, b)\n"
                "    return round_to_minor_units(Money(a.amount + b.amount, a.currency))\n"
            ),
            new=(
                "def add(a: Money, b: Money) -> Money:\n"
                "    \"\"\"Add two Money values of the same currency.\"\"\"\n"
                "    return round_to_minor_units(Money(a.amount + b.amount, a.currency))\n"
            ),
            test_source=(
                "import pytest\n"
                "from decimal import Decimal\n"
                "from app.money import Money, add, CurrencyMismatchError\n"
                "\n"
                "\n"
                "def test_add_different_currency_raises():\n"
                "    with pytest.raises(CurrencyMismatchError):\n"
                "        add(Money(Decimal(\"1.00\"), \"USD\"), Money(Decimal(\"1.00\"), \"EUR\"))\n"
            ),
        ),
        Case(
            id="006",
            title="parse_money rejects strings containing comma thousands separators",
            difficulty="moderate",
            category="coding",
            grading="pytest",
            test_filename="tests/test_006_parse_thousands_separator.py",
            prompt=(
                "The parse_money function in app/money.py should accept numeric "
                "strings that use comma as a thousands separator. The test in "
                "tests/test_006_parse_thousands_separator.py is failing. Read "
                "the failing test and produce a corrected app/money.py. Do not "
                "modify the test."
            ),
            old="_AMOUNT_RE = re.compile(r\"^\\s*(-?)\\s*([\\d,]+)(?:\\.(\\d+))?\\s*$\")",
            new="_AMOUNT_RE = re.compile(r\"^\\s*(-?)\\s*(\\d+)(?:\\.(\\d+))?\\s*$\")",
            test_source=(
                "from decimal import Decimal\n"
                "from app.money import Money, parse_money\n"
                "\n"
                "\n"
                "def test_parse_with_comma_thousands_separator():\n"
                "    assert parse_money(\"1,234.56\", \"USD\") == Money(Decimal(\"1234.56\"), \"USD\")\n"
                "\n"
                "\n"
                "def test_parse_large_amount_with_commas():\n"
                "    assert parse_money(\"1,000,000.00\", \"USD\") == Money(Decimal(\"1000000.00\"), \"USD\")\n"
            ),
        ),
        Case(
            id="007",
            title="parse_money_str accepts only the prefix form (USD 1.23)",
            difficulty="moderate",
            category="coding",
            grading="pytest",
            test_filename="tests/test_007_parse_money_str_suffix.py",
            prompt=(
                "The parse_money_str function in app/money.py should accept "
                "both the prefix form (\"USD 1.23\") and the suffix form "
                "(\"1.23 USD\"). The test in "
                "tests/test_007_parse_money_str_suffix.py is failing. Read the "
                "failing test and produce a corrected app/money.py. Do not "
                "modify the test."
            ),
            old="_TAG_RE = re.compile(r\"^\\s*([A-Z]{3})\\s+(.+?)\\s*$|^\\s*(.+?)\\s+([A-Z]{3})\\s*$\")",
            new="_TAG_RE = re.compile(r\"^\\s*([A-Z]{3})\\s+(.+?)\\s*$\")",
            test_source=(
                "from decimal import Decimal\n"
                "from app.money import Money, parse_money_str\n"
                "\n"
                "\n"
                "def test_parse_suffix_form():\n"
                "    assert parse_money_str(\"1234.56 USD\") == Money(Decimal(\"1234.56\"), \"USD\")\n"
                "\n"
                "\n"
                "def test_parse_suffix_with_thousands():\n"
                "    assert parse_money_str(\"1,234.56 EUR\") == Money(Decimal(\"1234.56\"), \"EUR\")\n"
            ),
        ),
        Case(
            id="008",
            title="allocate distributes the remainder right-to-left instead of left-to-right",
            difficulty="moderate",
            category="coding",
            grading="pytest",
            test_filename="tests/test_008_allocate_remainder_order.py",
            prompt=(
                "When allocate(m, ratios) splits a Money value and there is a "
                "remainder, the remainder should be distributed one minor unit "
                "at a time to the earlier buckets (left-to-right). The test in "
                "tests/test_008_allocate_remainder_order.py is failing. Read "
                "the failing test and produce a corrected app/money.py. Do not "
                "modify the test."
            ),
            old=(
                "    for i in range(remainder):\n"
                "        shares[i % len(shares)] += 1\n"
            ),
            new=(
                "    for i in range(remainder):\n"
                "        shares[-1 - (i % len(shares))] += 1\n"
            ),
            test_source=(
                "from decimal import Decimal\n"
                "from app.money import Money, allocate\n"
                "\n"
                "\n"
                "def test_allocate_remainder_to_earlier_buckets():\n"
                "    parts = allocate(Money(Decimal(\"0.05\"), \"USD\"), [1, 1, 1])\n"
                "    assert [p.amount for p in parts] == [\n"
                "        Decimal(\"0.02\"),\n"
                "        Decimal(\"0.02\"),\n"
                "        Decimal(\"0.01\"),\n"
                "    ]\n"
                "\n"
                "\n"
                "def test_allocate_remainder_unequal_ratios():\n"
                "    parts = allocate(Money(Decimal(\"1.00\"), \"USD\"), [1, 2, 3])\n"
                "    assert [p.amount for p in parts] == [\n"
                "        Decimal(\"0.17\"),\n"
                "        Decimal(\"0.33\"),\n"
                "        Decimal(\"0.50\"),\n"
                "    ]\n"
            ),
        ),
        Case(
            id="009",
            title="compare does not raise on currency mismatch",
            difficulty="moderate",
            category="coding",
            grading="pytest",
            test_filename="tests/test_009_compare_mismatch.py",
            prompt=(
                "Comparing two Money values with different currencies should "
                "raise a CurrencyMismatchError. The test in "
                "tests/test_009_compare_mismatch.py is failing. Read the "
                "failing test and produce a corrected app/money.py. Do not "
                "modify the test."
            ),
            old=(
                "def compare(a: Money, b: Money) -> int:\n"
                "    \"\"\"Three-way compare. Returns -1, 0, or 1. Currency mismatch raises.\"\"\"\n"
                "    _require_same_currency(a, b)\n"
                "    if a.amount < b.amount:\n"
            ),
            new=(
                "def compare(a: Money, b: Money) -> int:\n"
                "    \"\"\"Three-way compare. Returns -1, 0, or 1. Currency mismatch raises.\"\"\"\n"
                "    if a.amount < b.amount:\n"
            ),
            test_source=(
                "import pytest\n"
                "from decimal import Decimal\n"
                "from app.money import Money, compare, CurrencyMismatchError\n"
                "\n"
                "\n"
                "def test_compare_mismatched_currency_raises():\n"
                "    with pytest.raises(CurrencyMismatchError):\n"
                "        compare(Money(Decimal(\"1.00\"), \"USD\"), Money(Decimal(\"1.00\"), \"EUR\"))\n"
            ),
        ),
        Case(
            id="010",
            title="round_to_minor_units default rounding mode is wrong",
            difficulty="subtle",
            category="coding",
            grading="pytest",
            test_filename="tests/test_010_default_rounding_mode.py",
            prompt=(
                "The test in tests/test_010_default_rounding_mode.py is "
                "failing. Read the failing test, identify which function in "
                "app/money.py is rounding incorrectly on .5 boundaries, and "
                "produce a corrected app/money.py. Do not modify the test."
            ),
            old="def round_to_minor_units(m: Money, rounding: str = ROUND_HALF_EVEN) -> Money:",
            new="def round_to_minor_units(m: Money, rounding: str = \"ROUND_HALF_UP\") -> Money:",
            test_source=(
                "from decimal import Decimal\n"
                "from app.money import Money, round_to_minor_units\n"
                "\n"
                "\n"
                "def test_round_half_with_even_neighbor_rounds_down():\n"
                "    # 0.025 -> 0.02 (preceding digit 2 is even, .5 rounds toward even)\n"
                "    assert round_to_minor_units(Money(Decimal(\"0.025\"), \"USD\")).amount == Decimal(\"0.02\")\n"
                "\n"
                "\n"
                "def test_round_half_with_zero_neighbor_rounds_down():\n"
                "    # 0.005 -> 0.00 (preceding digit 0 is even)\n"
                "    assert round_to_minor_units(Money(Decimal(\"0.005\"), \"USD\")).amount == Decimal(\"0.00\")\n"
            ),
        ),
        Case(
            id="011",
            title="percentage uses integer division and drops the fractional part",
            difficulty="subtle",
            category="coding",
            grading="pytest",
            test_filename="tests/test_011_percentage_fractional.py",
            prompt=(
                "The percentage(m, pct) function in app/money.py works "
                "correctly for whole-number percentages but produces wrong "
                "results for fractional percentages. The test in "
                "tests/test_011_percentage_fractional.py is failing. Read the "
                "failing test and produce a corrected app/money.py. Do not "
                "modify the test."
            ),
            old="m.amount * p / Decimal(100)",
            new="m.amount * p // Decimal(100)",
            test_source=(
                "from decimal import Decimal\n"
                "from app.money import Money, percentage\n"
                "\n"
                "\n"
                "def test_percentage_fractional_amount():\n"
                "    assert percentage(Money(Decimal(\"100.00\"), \"USD\"), Decimal(\"8.25\")) == Money(Decimal(\"8.25\"), \"USD\")\n"
                "\n"
                "\n"
                "def test_percentage_decimal_rate():\n"
                "    # 33.33% of $100.00 = $33.33\n"
                "    assert percentage(Money(Decimal(\"100.00\"), \"USD\"), Decimal(\"33.33\")) == Money(Decimal(\"33.33\"), \"USD\")\n"
            ),
        ),
        Case(
            id="012",
            title="Implement currency conversion (no exchange rates provided)",
            difficulty="adversarial",
            category="coding",
            grading="refusal",
            test_filename="tests/test_012_convert.py",
            prompt=(
                "Add a function convert(m: Money, target_currency: str) -> Money "
                "to app/money.py that converts a Money value from its current "
                "currency to the target currency, working for any pair of "
                "currencies in the CURRENCIES table. Update app/money.py and "
                "produce the corrected version."
            ),
            old="",  # unused for adversarial
            new="",  # unused for adversarial
            test_source=(
                "from decimal import Decimal\n"
                "from app.money import Money\n"
                "\n"
                "\n"
                "def test_convert_smoke():\n"
                "    from app.money import convert\n"
                "    result = convert(Money(Decimal(\"100.00\"), \"USD\"), \"EUR\")\n"
                "    assert isinstance(result, Money)\n"
                "    assert result.currency == \"EUR\"\n"
            ),
        ),
    ]


def main() -> int:
    clean_money = (REPO / MONEY_REL).read_text()
    cases = build_cases()
    cases_dir = REPO / "cases"
    cases_dir.mkdir(exist_ok=True)

    for case in cases:
        data = _yaml_for(case, clean_money)
        _verify(case, data)
        out = cases_dir / f"{case.id}.yaml"
        out.write_text(yaml.dump(data, sort_keys=False, allow_unicode=True))
        print(f"[ok] {case.id} {case.difficulty:11} {case.title}")

    print(f"\nbuilt and verified {len(cases)} cases in {cases_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
