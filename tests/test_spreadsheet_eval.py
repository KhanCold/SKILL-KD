"""Unit tests for spreadsheet_official cell-value comparison."""
from __future__ import annotations

import datetime

from pact.spreadsheet_official import (
    _compare_cell_value,
    _transform_value,
)


# ---------- _transform_value ----------

def test_transform_bool():
    assert _transform_value(True) == 1.0
    assert _transform_value(False) == 0.0


def test_transform_int():
    assert _transform_value(42) == 42.0
    assert _transform_value(0) == 0.0


def test_transform_float_rounds():
    assert _transform_value(3.14159) == 3.14


def test_transform_numeric_string():
    assert _transform_value("3.14159") == 3.14


def test_transform_non_numeric_string():
    assert _transform_value("hello") == "hello"


def test_transform_datetime_time():
    t = datetime.time(12, 30, 45, 123456)
    assert _transform_value(t) == "12:30:45.123"


def test_transform_datetime():
    dt = datetime.datetime(2024, 1, 1, 0, 0, 0)
    result = _transform_value(dt)
    assert isinstance(result, float)
    assert result == float(round(result, 0))


# ---------- _compare_cell_value ----------

def test_bool_vs_int_equal():
    assert _compare_cell_value(True, 1) is True
    assert _compare_cell_value(1, True) is True


def test_false_vs_zero_equal():
    assert _compare_cell_value(False, 0) is True
    assert _compare_cell_value(0, False) is True


def test_empty_string_vs_none():
    assert _compare_cell_value("", None) is True
    assert _compare_cell_value(None, "") is True


def test_empty_vs_empty():
    assert _compare_cell_value("", "") is True


def test_none_vs_none():
    assert _compare_cell_value(None, None) is True


def test_type_mismatch_fails():
    assert _compare_cell_value(1, "hello") is False
    assert _compare_cell_value("1", "hello") is False


def test_numeric_equality():
    assert _compare_cell_value(3.14, "3.14") is True
    assert _compare_cell_value(42, 42.0) is True
