"""conftest.py — Shared pytest fixtures for autoresearch-tabular tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autoresearch_tabular.memory_graph import MemoryGraph


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small synthetic DataFrame with mixed column types."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, n).astype(float),
            "income": rng.normal(50000, 15000, n),
            "score": rng.uniform(0, 1, n),
            "category": rng.choice(["A", "B", "C"], n),
            "flag": rng.integers(0, 2, n),
        }
    )


@pytest.fixture
def empty_graph(tmp_path: Path) -> MemoryGraph:
    """Empty MemoryGraph backed by a temporary file."""
    graph_path = tmp_path / "test_graph.json"
    return MemoryGraph(path=graph_path)


@pytest.fixture
def graph_with_columns(empty_graph: MemoryGraph, sample_df: pd.DataFrame) -> MemoryGraph:
    """MemoryGraph pre-populated with Column nodes from sample_df."""
    empty_graph.populate_source_columns(sample_df)
    return empty_graph


@pytest.fixture
def graph_with_experiments(tmp_path: Path, sample_df: pd.DataFrame) -> MemoryGraph:
    """Rich graph with columns, 6 experiments (mix kept/reverted), features, SHAP, hypotheses."""
    mg = MemoryGraph(path=tmp_path / "rich_graph.json")
    mg.populate_source_columns(sample_df)

    # exp_1: baseline, kept
    mg.record_experiment(
        cv_score=0.80, cv_std=0.02, delta=0.0, n_features=3,
        composite_score=0.797, kept=True,
        description="col=age; op=log_transform; fit=train_only; reason=baseline",
        features_used=["age_log", "income_scaled", "flag"],
        feature_shap={"age_log": 0.4, "income_scaled": 0.3, "flag": 0.05},
        feature_shap_std={"age_log": 0.05, "income_scaled": 0.04, "flag": 0.01},
    )
    mg.register_feature_set(1, ["age_log", "income_scaled", "flag"])
    mg.register_feature("age_log", ["age"], experiment_id=1, save=False)
    mg.register_feature("income_scaled", ["income"], experiment_id=1, save=False)
    mg.register_feature("flag", ["flag"], experiment_id=1)

    # exp_2: improved, kept
    mg.record_experiment(
        cv_score=0.82, cv_std=0.02, delta=0.02, n_features=4,
        composite_score=0.816, kept=True,
        description="col=age,income; op=ratio_feature; fit=train_only; reason=interaction",
        features_used=["age_log", "income_scaled", "flag", "age_income_ratio"],
        feature_shap={"age_log": 0.35, "income_scaled": 0.25, "flag": 0.04, "age_income_ratio": 0.15},
        feature_shap_std={"age_log": 0.04, "income_scaled": 0.03, "flag": 0.01, "age_income_ratio": 0.02},
    )
    mg.register_feature_set(2, ["age_log", "income_scaled", "flag", "age_income_ratio"])
    mg.register_feature("age_income_ratio", ["age", "income"], experiment_id=2)

    # exp_3: reverted
    mg.record_experiment(
        cv_score=0.81, cv_std=0.03, delta=-0.006, n_features=5,
        composite_score=0.805, kept=False,
        description="col=score; op=sqrt_transform; fit=train_only; reason=reduce skew",
        features_used=["age_log", "income_scaled", "flag", "age_income_ratio", "score_sqrt"],
        feature_shap={"age_log": 0.3, "income_scaled": 0.2, "flag": 0.03, "age_income_ratio": 0.12, "score_sqrt": 0.01},
    )
    mg.register_feature_set(3, ["age_log", "income_scaled", "flag", "age_income_ratio", "score_sqrt"])
    mg.register_feature("score_sqrt", ["score"], experiment_id=3)

    # exp_4: reverted
    mg.record_experiment(
        cv_score=0.80, cv_std=0.02, delta=-0.016, n_features=4,
        composite_score=0.796, kept=False,
        description="col=category; op=onehot; fit=train_only; reason=expand categories",
        features_used=["age_log", "income_scaled", "flag", "cat_onehot"],
    )
    mg.register_feature_set(4, ["age_log", "income_scaled", "flag", "cat_onehot"])
    mg.register_feature("cat_onehot", ["category"], experiment_id=4)

    # exp_5: improved, kept
    mg.record_experiment(
        cv_score=0.84, cv_std=0.02, delta=0.024, n_features=4,
        composite_score=0.836, kept=True,
        description="col=age; op=quantile_bin; fit=train_only; reason=nonlinear signal",
        features_used=["age_log", "income_scaled", "age_income_ratio", "age_bin"],
        feature_shap={"age_log": 0.3, "income_scaled": 0.25, "age_income_ratio": 0.15, "age_bin": 0.18},
        feature_shap_std={"age_log": 0.03, "income_scaled": 0.03, "age_income_ratio": 0.02, "age_bin": 0.02},
    )
    mg.register_feature_set(5, ["age_log", "income_scaled", "age_income_ratio", "age_bin"])
    mg.register_feature("age_bin", ["age"], experiment_id=5)

    # exp_6: reverted
    mg.record_experiment(
        cv_score=0.83, cv_std=0.02, delta=-0.006, n_features=5,
        composite_score=0.825, kept=False,
        description="col=income; op=clip; fit=train_only; reason=outliers",
        features_used=["age_log", "income_scaled", "age_income_ratio", "age_bin", "income_clipped"],
    )
    mg.register_feature_set(6, ["age_log", "income_scaled", "age_income_ratio", "age_bin", "income_clipped"])
    mg.register_feature("income_clipped", ["income"], experiment_id=6)

    # Add hypotheses
    h1 = mg.add_hypothesis("log transform on age will reduce skew and improve score", predicted_direction="+", predicted_delta=0.01)
    mg.resolve_hypothesis(h1, experiment_id=1, kept=True, actual_delta=0.0)
    h2 = mg.add_hypothesis("ratio of age/income captures purchasing power", predicted_direction="+", predicted_delta=0.02)
    mg.resolve_hypothesis(h2, experiment_id=2, kept=True, actual_delta=0.02)
    mg.add_hypothesis("clipping income outliers might help", predicted_direction="?")

    return mg


@pytest.fixture
def query_df() -> pd.DataFrame:
    """DataFrame with deterministic values for metric/guard/eval testing."""
    return pd.DataFrame(
        {
            "age": [25.0, 30.0, 35.0, np.nan, 45.0],
            "income": [50000.0, 60000.0, np.inf, 80000.0, 90000.0],
            "category": ["A", "B", "A", "C", "B"],
        }
    )
