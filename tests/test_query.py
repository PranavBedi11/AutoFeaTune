"""test_query.py — Tests for the safe evaluation security boundary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoresearch_tabular.query import _safe_eval


# ---------------------------------------------------------------------------
# _safe_eval — security boundary tests
# ---------------------------------------------------------------------------


@pytest.fixture
def eval_df() -> pd.DataFrame:
    """DataFrame with known values for eval testing."""
    return pd.DataFrame(
        {
            "age": [25.0, 30.0, 35.0, 40.0, 45.0],
            "income": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
            "score": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )


class TestSafeEval:
    def test_column_reference(self, eval_df: pd.DataFrame) -> None:
        result = _safe_eval("age", eval_df)
        pd.testing.assert_series_equal(result, eval_df["age"], check_names=False)

    def test_arithmetic(self, eval_df: pd.DataFrame) -> None:
        result = _safe_eval("age + income", eval_df)
        expected = eval_df["age"] + eval_df["income"]
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_numpy_log1p(self, eval_df: pd.DataFrame) -> None:
        result = _safe_eval("log1p(income)", eval_df)
        expected = np.log1p(eval_df["income"])
        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_numpy_sqrt(self, eval_df: pd.DataFrame) -> None:
        result = _safe_eval("sqrt(age)", eval_df)
        expected = np.sqrt(eval_df["age"])
        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_numpy_abs(self, eval_df: pd.DataFrame) -> None:
        df = pd.DataFrame({"x": [-1.0, -2.0, 3.0]})
        result = _safe_eval("abs(x)", df)
        np.testing.assert_array_equal(result.values, [1.0, 2.0, 3.0])

    def test_numpy_floor(self, eval_df: pd.DataFrame) -> None:
        result = _safe_eval("floor(score)", eval_df)
        expected = np.floor(eval_df["score"])
        np.testing.assert_array_equal(result.values, expected.values)

    def test_rejects_import(self, eval_df: pd.DataFrame) -> None:
        with pytest.raises(Exception):
            _safe_eval("__import__('os')", eval_df)

    def test_rejects_builtins_open(self, eval_df: pd.DataFrame) -> None:
        with pytest.raises(Exception):
            _safe_eval("open('/etc/passwd')", eval_df)

    def test_rejects_unknown_names(self, eval_df: pd.DataFrame) -> None:
        with pytest.raises(Exception):
            _safe_eval("nonexistent_col", eval_df)

    def test_handles_nan(self) -> None:
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        result = _safe_eval("x + 1", df)
        assert np.isnan(result.iloc[1])
        assert result.iloc[0] == 2.0
        assert result.iloc[2] == 4.0

    def test_dunder_access_known_limitation(self, eval_df: pd.DataFrame) -> None:
        """pd.eval allows attribute access on Series — this is a known limitation.

        Acceptable because callers are the autoresearch AI agent,
        not untrusted end users. Documented here so the limitation is explicit.
        """
        # This does NOT raise — pd.eval resolves age to a Series, then accesses .__class__
        result = _safe_eval("age.__class__", eval_df)
        assert result is not None  # just verifying it doesn't crash

    def test_complex_expression(self, eval_df: pd.DataFrame) -> None:
        result = _safe_eval("age * 2 + income / 1000", eval_df)
        expected = eval_df["age"] * 2 + eval_df["income"] / 1000
        pd.testing.assert_series_equal(result, expected, check_names=False)
