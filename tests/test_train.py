"""test_train.py — Tests for critical functions in the training harness."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from autoresearch_tabular.train import (
    _compute_fold_shap,
    compute_metric,
    guard_dataframe,
)


# ---------------------------------------------------------------------------
# compute_metric
# ---------------------------------------------------------------------------


class TestComputeMetric:
    def test_rmse_known_values(self) -> None:
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        result = compute_metric(y_true, y_pred, "rmse")
        assert abs(result - 0.1) < 1e-6

    def test_mae_known_values(self) -> None:
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        result = compute_metric(y_true, y_pred, "mae")
        assert abs(result - 0.5) < 1e-6

    def test_auc_perfect(self) -> None:
        y_true = pd.Series([0, 0, 1, 1])
        y_pred = np.array([0.0, 0.1, 0.9, 1.0])
        result = compute_metric(y_true, y_pred, "auc")
        assert result == 1.0

    def test_auc_random(self) -> None:
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])
        result = compute_metric(y_true, y_pred, "auc")
        assert result == 0.5

    def test_logloss_sanity(self) -> None:
        y_true = pd.Series([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        result = compute_metric(y_true, y_pred, "logloss")
        assert result > 0  # logloss is always positive
        assert np.isfinite(result)

    def test_f1_binary(self) -> None:
        y_true = pd.Series([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        result = compute_metric(y_true, y_pred, "f1")
        assert result == 1.0

    def test_accuracy(self) -> None:
        y_true = pd.Series([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        result = compute_metric(y_true, y_pred, "accuracy")
        assert result == 1.0

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_metric(pd.Series([1]), np.array([1]), "bogus")


# ---------------------------------------------------------------------------
# guard_dataframe
# ---------------------------------------------------------------------------


class TestGuardDataframe:
    def test_replaces_inf_with_nan(self) -> None:
        df = pd.DataFrame({"a": [1.0, np.inf, -np.inf, 4.0]})
        result, _ = guard_dataframe(df, fold_idx=0, split_name="train")
        assert not np.isinf(result.values).any()

    def test_encodes_categoricals(self) -> None:
        df = pd.DataFrame({"cat": ["A", "B", "C", "A"]})
        result, encoders = guard_dataframe(df, fold_idx=0, split_name="train")
        assert "cat" in encoders
        assert result["cat"].dtype == np.float32
        # Encoded values should be integers 0, 1, 2 (as float32)
        assert set(result["cat"].unique()) == {0.0, 1.0, 2.0}

    def test_fills_nan_with_median(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0]})
        result, _ = guard_dataframe(df, fold_idx=0, split_name="train")
        assert not result.isna().any().any()
        # median of [1, 2, 4] = 2.0
        assert result["x"].iloc[2] == 2.0

    def test_reuses_label_encoders(self) -> None:
        train_df = pd.DataFrame({"cat": ["A", "B", "C"]})
        _, encoders = guard_dataframe(train_df, fold_idx=0, split_name="train")
        val_df = pd.DataFrame({"cat": ["B", "A", "D"]})  # D is unseen
        result, _ = guard_dataframe(val_df, fold_idx=0, split_name="val", label_encoders=encoders)
        # A and B should map to same ints as training; D is unseen → -1
        a_val = encoders["cat"]["A"]
        b_val = encoders["cat"]["B"]
        assert result["cat"].iloc[0] == float(b_val)
        assert result["cat"].iloc[1] == float(a_val)
        assert result["cat"].iloc[2] == -1.0

    def test_returns_float32_for_categoricals(self) -> None:
        df = pd.DataFrame({"cat": ["A", "B"], "num": [1.0, 2.0]})
        result, _ = guard_dataframe(df, fold_idx=0, split_name="train")
        assert result["cat"].dtype == np.float32

    def test_does_not_modify_original(self) -> None:
        df = pd.DataFrame({"a": [1.0, np.inf, 3.0]})
        original_values = df["a"].tolist()
        guard_dataframe(df, fold_idx=0, split_name="train")
        assert df["a"].tolist() == original_values


# ---------------------------------------------------------------------------
# _compute_fold_shap
# ---------------------------------------------------------------------------


class TestComputeFoldShap:
    def test_2d_contribs_regression(self) -> None:
        """Regression: booster returns (n_samples, n_features + 1)."""
        n_samples, n_features = 10, 3
        columns = ["feat_a", "feat_b", "feat_c"]
        X_val = pd.DataFrame(
            np.random.default_rng(0).random((n_samples, n_features)),
            columns=columns,
        )

        # Contribs: n_features columns + 1 bias column
        contribs = np.ones((n_samples, n_features + 1)) * 0.5
        contribs[:, 0] = 1.0  # feat_a gets higher SHAP

        mock_booster = MagicMock()
        mock_booster.predict.return_value = contribs
        mock_model = MagicMock()
        mock_model.get_booster.return_value = mock_booster

        result = _compute_fold_shap(mock_model, X_val)

        assert set(result.keys()) == set(columns)
        assert result["feat_a"] == pytest.approx(1.0)
        assert result["feat_b"] == pytest.approx(0.5)

    def test_3d_contribs_multiclass(self) -> None:
        """Multiclass newer XGBoost: (n_samples, n_classes, n_features + 1)."""
        n_samples, n_features, n_classes = 10, 3, 4
        columns = ["feat_a", "feat_b", "feat_c"]
        X_val = pd.DataFrame(
            np.random.default_rng(0).random((n_samples, n_features)),
            columns=columns,
        )

        contribs = np.ones((n_samples, n_classes, n_features + 1)) * 0.25

        mock_booster = MagicMock()
        mock_booster.predict.return_value = contribs
        mock_model = MagicMock()
        mock_model.get_booster.return_value = mock_booster

        result = _compute_fold_shap(mock_model, X_val)

        assert set(result.keys()) == set(columns)
        for v in result.values():
            assert v == pytest.approx(0.25)

    def test_2d_contribs_multiclass_older_xgboost(self) -> None:
        """Multiclass older XGBoost: (n_samples, n_classes * (n_features + 1))."""
        n_samples, n_features, n_classes = 10, 3, 4
        columns = ["feat_a", "feat_b", "feat_c"]
        X_val = pd.DataFrame(
            np.random.default_rng(0).random((n_samples, n_features)),
            columns=columns,
        )

        # Flat layout: n_classes * (n_features + 1) columns
        contribs = np.ones((n_samples, n_classes * (n_features + 1))) * 0.1

        mock_booster = MagicMock()
        mock_booster.predict.return_value = contribs
        mock_model = MagicMock()
        mock_model.get_booster.return_value = mock_booster

        result = _compute_fold_shap(mock_model, X_val)

        assert set(result.keys()) == set(columns)
        for v in result.values():
            assert v == pytest.approx(0.1)

    def test_unexpected_shape_returns_empty(self) -> None:
        """Unknown contrib shape → empty dict, no crash."""
        X_val = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        contribs = np.ones((2, 10))  # Wrong shape for 2 features

        mock_booster = MagicMock()
        mock_booster.predict.return_value = contribs
        mock_model = MagicMock()
        mock_model.get_booster.return_value = mock_booster

        result = _compute_fold_shap(mock_model, X_val)
        assert result == {}

    def test_exception_returns_empty(self) -> None:
        """Any exception during SHAP → empty dict."""
        X_val = pd.DataFrame({"a": [1.0]})
        mock_model = MagicMock()
        mock_model.get_booster.side_effect = RuntimeError("boom")

        result = _compute_fold_shap(mock_model, X_val)
        assert result == {}
