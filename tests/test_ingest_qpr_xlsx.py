#!/usr/bin/env python3
"""Tests for helper functions in scripts/ingest_qpr_xlsx.py."""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import pytest

pytest.importorskip("openpyxl")


def _load_ingest_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "ingest_qpr_xlsx.py"
    spec = importlib.util.spec_from_file_location("ingest_qpr_xlsx", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# _clean_str
# ---------------------------------------------------------------------------

def test_clean_str_none_returns_none():
    mod = _load_ingest_module()
    assert mod._clean_str(None) is None


def test_clean_str_whitespace_only_returns_none():
    mod = _load_ingest_module()
    assert mod._clean_str("   ") is None


def test_clean_str_normalizes_nbsp():
    mod = _load_ingest_module()
    assert mod._clean_str("foo\xa0bar") == "foo bar"


def test_clean_str_strips_whitespace():
    mod = _load_ingest_module()
    assert mod._clean_str("  hello  ") == "hello"


# ---------------------------------------------------------------------------
# _clean_float
# ---------------------------------------------------------------------------

def test_clean_float_none_returns_none():
    mod = _load_ingest_module()
    assert mod._clean_float(None) is None


def test_clean_float_int_passthrough():
    mod = _load_ingest_module()
    assert mod._clean_float(123) == pytest.approx(123.0)


def test_clean_float_float_passthrough():
    mod = _load_ingest_module()
    assert mod._clean_float(1.5) == pytest.approx(1.5)


def test_clean_float_nan_returns_none():
    mod = _load_ingest_module()
    assert mod._clean_float(float("nan")) is None


def test_clean_float_string_with_dollar_and_comma():
    mod = _load_ingest_module()
    result = mod._clean_float("$1,234.56")
    assert result == pytest.approx(1234.56)


def test_clean_float_bare_number_string():
    mod = _load_ingest_module()
    result = mod._clean_float("89382.76")
    assert result == pytest.approx(89382.76)


def test_clean_float_unparseable_returns_none():
    mod = _load_ingest_module()
    assert mod._clean_float("not-a-number") is None


# ---------------------------------------------------------------------------
# _quarter_label
# ---------------------------------------------------------------------------

def test_quarter_label_from_datetime_q1():
    mod = _load_ingest_module()
    assert mod._quarter_label(datetime(2021, 1, 1)) == "2021 Q1"


def test_quarter_label_from_datetime_q2():
    mod = _load_ingest_module()
    assert mod._quarter_label(datetime(2022, 4, 15)) == "2022 Q2"


def test_quarter_label_from_datetime_q3():
    mod = _load_ingest_module()
    assert mod._quarter_label(datetime(2023, 7, 31)) == "2023 Q3"


def test_quarter_label_from_datetime_q4():
    mod = _load_ingest_module()
    assert mod._quarter_label(datetime(2020, 12, 31)) == "2020 Q4"


def test_quarter_label_string_passthrough():
    mod = _load_ingest_module()
    assert mod._quarter_label("2022 Q3") == "2022 Q3"


def test_quarter_label_none_returns_none():
    mod = _load_ingest_module()
    assert mod._quarter_label(None) is None


# ---------------------------------------------------------------------------
# _parse_payroll_usd (via _PAYROLL_AMOUNT_RE)
# ---------------------------------------------------------------------------

def test_parse_payroll_usd_extracts_amount():
    mod = _load_ingest_module()
    line = "$89382.76 - Payroll Allocation"
    amounts = mod._parse_payroll_usd(line)
    assert amounts == pytest.approx([89382.76])


def test_parse_payroll_usd_no_payroll_keyword_returns_empty():
    mod = _load_ingest_module()
    amounts = mod._parse_payroll_usd("$50000 - Equipment Purchase")
    assert amounts == []


def test_parse_payroll_usd_empty_string_returns_empty():
    mod = _load_ingest_module()
    assert mod._parse_payroll_usd("") == []


# ---------------------------------------------------------------------------
# _resolve_file
# ---------------------------------------------------------------------------

def test_resolve_file_finds_direct_file(tmp_path):
    mod = _load_ingest_module()
    target = tmp_path / "myfile.xlsx"
    target.touch()
    result = mod._resolve_file(tmp_path, "myfile.xlsx")
    assert result == target


def test_resolve_file_finds_bracket_duplicate(tmp_path):
    mod = _load_ingest_module()
    bracket = tmp_path / "myfile[96].xlsx"
    bracket.touch()
    result = mod._resolve_file(tmp_path, "myfile.xlsx")
    assert result == bracket


def test_resolve_file_prefers_direct_over_bracket(tmp_path):
    mod = _load_ingest_module()
    direct = tmp_path / "myfile.xlsx"
    direct.touch()
    bracket = tmp_path / "myfile[96].xlsx"
    bracket.touch()
    result = mod._resolve_file(tmp_path, "myfile.xlsx")
    assert result == direct


def test_resolve_file_missing_returns_none(tmp_path):
    mod = _load_ingest_module()
    result = mod._resolve_file(tmp_path, "nonexistent.xlsx")
    assert result is None
