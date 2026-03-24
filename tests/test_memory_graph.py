"""test_memory_graph.py — Tests for the NetworkX memory graph."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autoresearch_tabular.memory_graph import MemoryGraph, load_graph
from autoresearch_tabular.inspect_graph import (
    get_saturated_columns,
    get_transform_success_rates,
    get_load_bearing_features,
    get_untried_column_transform_pairs,
    get_shap_ranking,
    get_shap_consensus,
)


# ---------------------------------------------------------------------------
# populate_source_columns
# ---------------------------------------------------------------------------


class TestPopulateSourceColumns:
    def test_creates_column_nodes(
        self, graph_with_columns: MemoryGraph, sample_df: pd.DataFrame
    ) -> None:
        cols = graph_with_columns.get_source_columns()
        col_names = {c["name"] for c in cols}
        assert col_names == set(sample_df.columns)

    def test_idempotent(
        self, graph_with_columns: MemoryGraph, sample_df: pd.DataFrame
    ) -> None:
        # Second call should not duplicate nodes
        graph_with_columns.populate_source_columns(sample_df)
        cols = graph_with_columns.get_source_columns()
        assert len(cols) == len(sample_df.columns)

    def test_column_attributes(
        self, graph_with_columns: MemoryGraph, sample_df: pd.DataFrame
    ) -> None:
        cols = {c["name"]: c for c in graph_with_columns.get_source_columns()}
        # Numeric column should have mean/std
        assert cols["age"]["mean"] > 0
        assert cols["age"]["std"] > 0
        # Category column is non-numeric — mean/std should be 0.0
        assert cols["category"]["mean"] == 0.0


# ---------------------------------------------------------------------------
# record_experiment
# ---------------------------------------------------------------------------


class TestRecordExperiment:
    def test_records_first_experiment(self, empty_graph: MemoryGraph) -> None:
        exp_id = empty_graph.record_experiment(
            cv_score=0.85,
            cv_std=0.02,
            delta=0.0,
            n_features=10,
            composite_score=0.84,
            kept=True,
            description="baseline",
        )
        assert exp_id == 1

    def test_increments_id(self, empty_graph: MemoryGraph) -> None:
        empty_graph.record_experiment(
            cv_score=0.80, cv_std=0.01, delta=0.0,
            n_features=5, composite_score=0.795, kept=True,
        )
        exp_id2 = empty_graph.record_experiment(
            cv_score=0.82, cv_std=0.01, delta=0.005,
            n_features=5, composite_score=0.815, kept=True,
        )
        assert exp_id2 == 2

    def test_kept_true_creates_improved_over_edge(self, empty_graph: MemoryGraph) -> None:
        empty_graph.record_experiment(
            cv_score=0.80, cv_std=0.01, delta=0.0,
            n_features=5, composite_score=0.795, kept=True, description="first",
        )
        empty_graph.record_experiment(
            cv_score=0.82, cv_std=0.01, delta=0.02,
            n_features=5, composite_score=0.815, kept=True, description="improved",
        )
        g = empty_graph.graph
        # There should be an IMPROVED_OVER edge from exp_2 to exp_1
        assert g.has_edge("exp_2", "exp_1")
        assert g["exp_2"]["exp_1"]["rel"] == "IMPROVED_OVER"

    def test_kept_false_no_improved_over_edge(self, empty_graph: MemoryGraph) -> None:
        empty_graph.record_experiment(
            cv_score=0.80, cv_std=0.01, delta=0.0,
            n_features=5, composite_score=0.795, kept=True,
        )
        empty_graph.record_experiment(
            cv_score=0.78, cv_std=0.01, delta=-0.015,
            n_features=5, composite_score=0.775, kept=False,
        )
        g = empty_graph.graph
        assert not g.has_edge("exp_2", "exp_1")


# ---------------------------------------------------------------------------
# get_experiment_history
# ---------------------------------------------------------------------------


class TestGetExperimentHistory:
    def test_returns_most_recent_first(self, empty_graph: MemoryGraph) -> None:
        for i in range(5):
            empty_graph.record_experiment(
                cv_score=0.8 + i * 0.01, cv_std=0.01, delta=0.01,
                n_features=5, composite_score=0.8 + i * 0.01, kept=True,
            )
        history = empty_graph.get_experiment_history(n=3)
        assert len(history) == 3
        assert history[0]["exp_id"] == 5
        assert history[1]["exp_id"] == 4

    def test_respects_n_limit(self, empty_graph: MemoryGraph) -> None:
        for _ in range(10):
            empty_graph.record_experiment(
                cv_score=0.8, cv_std=0.01, delta=0.0,
                n_features=5, composite_score=0.795, kept=True,
            )
        assert len(empty_graph.get_experiment_history(n=5)) == 5

    def test_empty_graph_returns_empty(self, empty_graph: MemoryGraph) -> None:
        assert empty_graph.get_experiment_history() == []


# ---------------------------------------------------------------------------
# get_feature_lineage
# ---------------------------------------------------------------------------


class TestGetFeatureLineage:
    def test_lineage_traces_to_source_columns(
        self, graph_with_columns: MemoryGraph
    ) -> None:
        graph_with_columns.register_feature("age_log", ["age"], experiment_id=1)
        lineage = graph_with_columns.get_feature_lineage("age_log")
        node_names = {n["name"] for n in lineage}
        assert "age" in node_names

    def test_unknown_feature_returns_empty(self, empty_graph: MemoryGraph) -> None:
        assert empty_graph.get_feature_lineage("nonexistent") == []

    def test_multi_hop_lineage(self, graph_with_columns: MemoryGraph) -> None:
        # age_log derived from age; age_income derived from age_log and income
        graph_with_columns.register_feature("age_log", ["age"], experiment_id=1)
        graph_with_columns.register_feature("age_income", ["age_log", "income"], experiment_id=2)
        lineage = graph_with_columns.get_feature_lineage("age_income")
        node_names = {n["name"] for n in lineage}
        # Should trace through age_log to age, and income directly
        assert "income" in node_names or "age_log" in node_names


# ---------------------------------------------------------------------------
# JSON round-trip serialization
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    def test_saves_and_loads(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        graph_path = tmp_path / "roundtrip.json"
        mg1 = MemoryGraph(path=graph_path)
        mg1.populate_source_columns(sample_df)
        mg1.record_experiment(
            cv_score=0.85, cv_std=0.02, delta=0.0,
            n_features=5, composite_score=0.845, kept=True, description="test",
        )
        hyp_id = mg1.add_hypothesis("log transform on income helps", predicted_direction="+")
        mg1.resolve_hypothesis(hyp_id, experiment_id=1, kept=True, actual_delta=0.01)

        # Load fresh
        mg2 = MemoryGraph(path=graph_path)
        assert len(mg2.get_source_columns()) == len(sample_df.columns)
        assert len(mg2.get_experiment_history()) == 1
        assert mg2.get_experiment_history()[0]["description"] == "test"
        assert len(mg2.get_active_hypotheses()) == 1

    def test_json_is_valid(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        graph_path = tmp_path / "valid.json"
        mg = MemoryGraph(path=graph_path)
        mg.populate_source_columns(sample_df)

        with open(graph_path) as f:
            data = json.load(f)
        # node_link_data format should have 'nodes' and edges key (NetworkX 3.x uses 'edges')
        assert "nodes" in data
        assert "edges" in data or "links" in data

    def test_load_graph_helper(self, tmp_path: Path) -> None:
        graph_path = tmp_path / "helper.json"
        mg = load_graph(graph_path)
        assert isinstance(mg, MemoryGraph)
        assert not graph_path.exists()  # not saved until .save() is called
        mg.save()
        assert graph_path.exists()


# ---------------------------------------------------------------------------
# get_saturated_columns (standalone function)
# ---------------------------------------------------------------------------


class TestGetSaturatedColumns:
    def _make_graph(self, tmp_path: Path, sample_df: pd.DataFrame) -> MemoryGraph:
        mg = MemoryGraph(path=tmp_path / "sat.json")
        mg.populate_source_columns(sample_df)
        return mg

    def test_returns_empty_with_no_experiments(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        mg = self._make_graph(tmp_path, sample_df)
        assert get_saturated_columns(mg.graph) == []

    def test_column_with_high_deltas_not_saturated(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        mg = self._make_graph(tmp_path, sample_df)
        mg.register_feature("age_log", ["age"], experiment_id=1)
        # Three experiments with large deltas
        for i in range(3):
            exp_id = mg.record_experiment(
                cv_score=0.85 + i * 0.05, cv_std=0.01,
                delta=0.05,  # well above threshold
                n_features=5, composite_score=0.84 + i * 0.05,
                kept=True, features_used=["age_log"],
            )
            mg.register_feature_set(exp_id, ["age_log"])
        result = get_saturated_columns(mg.graph, min_experiments=3, delta_threshold=0.001)
        col_names = [r["column"] for r in result]
        assert "age" not in col_names

    def test_column_with_near_zero_deltas_is_saturated(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        mg = self._make_graph(tmp_path, sample_df)
        mg.register_feature("income_log", ["income"], experiment_id=1)
        for i in range(4):
            exp_id = mg.record_experiment(
                cv_score=0.80, cv_std=0.01,
                delta=0.0001,  # tiny — below threshold
                n_features=5, composite_score=0.795,
                kept=False, features_used=["income_log"],
            )
            mg.register_feature_set(exp_id, ["income_log"])
        result = get_saturated_columns(mg.graph, min_experiments=3, delta_threshold=0.001)
        col_names = [r["column"] for r in result]
        assert "income" in col_names

    def test_below_min_experiments_not_reported(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        mg = self._make_graph(tmp_path, sample_df)
        mg.register_feature("score_sq", ["score"], experiment_id=1)
        # Only 2 experiments — below min_experiments=3
        for _ in range(2):
            exp_id = mg.record_experiment(
                cv_score=0.80, cv_std=0.01, delta=0.0,
                n_features=5, composite_score=0.795,
                kept=False, features_used=["score_sq"],
            )
            mg.register_feature_set(exp_id, ["score_sq"])
        result = get_saturated_columns(mg.graph, min_experiments=3, delta_threshold=0.001)
        assert result == []

    def test_returns_sorted_by_mean_delta_ascending(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        mg = self._make_graph(tmp_path, sample_df)
        mg.register_feature("age_log", ["age"], experiment_id=1)
        mg.register_feature("income_log", ["income"], experiment_id=1)
        # age: mean delta = 0.0001 (more saturated)
        for _ in range(3):
            exp_id = mg.record_experiment(
                cv_score=0.80, cv_std=0.01, delta=0.0001,
                n_features=5, composite_score=0.795,
                kept=False, features_used=["age_log"],
            )
            mg.register_feature_set(exp_id, ["age_log"])
        # income: mean delta = 0.0005 (less saturated but still below threshold)
        for _ in range(3):
            exp_id = mg.record_experiment(
                cv_score=0.80, cv_std=0.01, delta=0.0005,
                n_features=5, composite_score=0.795,
                kept=False, features_used=["income_log"],
            )
            mg.register_feature_set(exp_id, ["income_log"])
        result = get_saturated_columns(mg.graph, min_experiments=3, delta_threshold=0.001)
        assert len(result) == 2
        assert result[0]["column"] == "age"   # lower mean_delta first
        assert result[1]["column"] == "income"

    def test_result_contains_expected_keys(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        mg = self._make_graph(tmp_path, sample_df)
        mg.register_feature("flag_enc", ["flag"], experiment_id=1)
        for _ in range(3):
            exp_id = mg.record_experiment(
                cv_score=0.80, cv_std=0.01, delta=0.0,
                n_features=5, composite_score=0.795,
                kept=False, features_used=["flag_enc"],
            )
            mg.register_feature_set(exp_id, ["flag_enc"])
        result = get_saturated_columns(mg.graph, min_experiments=3, delta_threshold=0.001)
        assert len(result) == 1
        row = result[0]
        assert {"column", "n_experiments", "mean_delta", "max_delta"} <= row.keys()
        assert row["n_experiments"] == 3


# ---------------------------------------------------------------------------
# get_transform_success_rates (standalone function)
# ---------------------------------------------------------------------------


class TestGetTransformSuccessRates:
    def test_empty_graph_returns_empty_dict(self, empty_graph: MemoryGraph) -> None:
        assert get_transform_success_rates(empty_graph.graph) == {}

    def test_log_transforms_counted(self, empty_graph: MemoryGraph) -> None:
        empty_graph.record_experiment(
            cv_score=0.85, cv_std=0.01, delta=0.01,
            n_features=5, composite_score=0.845,
            kept=True,
            description="col=price; op=log_transform; fit=train_only; reason=reduce skew",
        )
        empty_graph.record_experiment(
            cv_score=0.84, cv_std=0.01, delta=0.0,
            n_features=5, composite_score=0.835,
            kept=False,
            description="col=quantity; op=log1p_transform; fit=train_only; reason=tail",
        )
        rates = get_transform_success_rates(empty_graph.graph)
        assert "log" in rates
        assert rates["log"]["total"] == 2
        assert rates["log"]["kept"] == 1
        assert rates["log"]["rate"] == 0.5

    def test_clustering_family_detected(self, empty_graph: MemoryGraph) -> None:
        empty_graph.record_experiment(
            cv_score=0.90, cv_std=0.01, delta=0.02,
            n_features=8, composite_score=0.892,
            kept=True,
            description="col=lat,lon; op=kmeans_cluster_geo; fit=train_only; reason=neighborhoods",
        )
        rates = get_transform_success_rates(empty_graph.graph)
        assert "clustering" in rates
        assert rates["clustering"]["kept"] == 1

    def test_feature_drop_family_detected(self, empty_graph: MemoryGraph) -> None:
        empty_graph.record_experiment(
            cv_score=0.82, cv_std=0.01, delta=0.005,
            n_features=4, composite_score=0.816,
            kept=True,
            description="col=redundant_feat; op=drop_redundant_feature; fit=train_only; reason=penalty",
        )
        rates = get_transform_success_rates(empty_graph.graph)
        assert "feature_drop" in rates
        assert rates["feature_drop"]["total"] == 1
        assert rates["feature_drop"]["rate"] == 1.0

    def test_unknown_op_classified_as_other(self, empty_graph: MemoryGraph) -> None:
        empty_graph.record_experiment(
            cv_score=0.80, cv_std=0.01, delta=0.0,
            n_features=5, composite_score=0.795,
            kept=False,
            description="something completely unrecognized",
        )
        rates = get_transform_success_rates(empty_graph.graph)
        assert "other" in rates

    def test_rates_sorted_by_rate_descending(self, empty_graph: MemoryGraph) -> None:
        # log: 2/2 = 1.0; drop: 0/2 = 0.0
        for _ in range(2):
            empty_graph.record_experiment(
                cv_score=0.85, cv_std=0.01, delta=0.01,
                n_features=5, composite_score=0.845,
                kept=True,
                description="col=x; op=log_transform; fit=train_only; reason=skew",
            )
        for _ in range(2):
            empty_graph.record_experiment(
                cv_score=0.84, cv_std=0.01, delta=0.0,
                n_features=4, composite_score=0.836,
                kept=False,
                description="col=x; op=drop_feature; fit=train_only; reason=penalty",
            )
        rates = get_transform_success_rates(empty_graph.graph)
        families = list(rates.keys())
        log_idx = families.index("log")
        drop_idx = families.index("feature_drop")
        assert log_idx < drop_idx  # log (rate=1.0) before drop (rate=0.0)

    def test_rate_values_in_zero_to_one(self, empty_graph: MemoryGraph) -> None:
        for kept in [True, False, True]:
            empty_graph.record_experiment(
                cv_score=0.80, cv_std=0.01, delta=0.01,
                n_features=5, composite_score=0.795,
                kept=kept,
                description="col=x; op=ratio_feature; fit=train_only; reason=normalize",
            )
        rates = get_transform_success_rates(empty_graph.graph)
        for fam, stats in rates.items():
            assert 0.0 <= stats["rate"] <= 1.0, f"{fam} rate out of bounds"


# ---------------------------------------------------------------------------
# get_load_bearing_features (standalone function)
# ---------------------------------------------------------------------------


class TestGetLoadBearingFeatures:
    def test_empty_graph_returns_empty(self, empty_graph: MemoryGraph) -> None:
        assert get_load_bearing_features(empty_graph.graph) == []

    def test_no_kept_experiments_returns_empty(self, empty_graph: MemoryGraph) -> None:
        exp_id = empty_graph.record_experiment(
            cv_score=0.80, cv_std=0.01, delta=0.0,
            n_features=3, composite_score=0.797,
            kept=False, features_used=["a", "b", "c"],
        )
        empty_graph.register_feature_set(exp_id, ["a", "b", "c"])
        assert get_load_bearing_features(empty_graph.graph) == []

    def test_single_kept_experiment_returns_all_its_features(
        self, empty_graph: MemoryGraph
    ) -> None:
        exp_id = empty_graph.record_experiment(
            cv_score=0.85, cv_std=0.01, delta=0.01,
            n_features=3, composite_score=0.847,
            kept=True, features_used=["f1", "f2", "f3"],
        )
        empty_graph.register_feature_set(exp_id, ["f1", "f2", "f3"])
        result = get_load_bearing_features(empty_graph.graph)
        assert set(result) == {"f1", "f2", "f3"}

    def test_intersection_across_multiple_kept_experiments(
        self, empty_graph: MemoryGraph
    ) -> None:
        # exp1 kept: [core, extra1]
        exp1 = empty_graph.record_experiment(
            cv_score=0.85, cv_std=0.01, delta=0.01,
            n_features=2, composite_score=0.848,
            kept=True, features_used=["core", "extra1"],
        )
        empty_graph.register_feature_set(exp1, ["core", "extra1"])
        # exp2 kept: [core, extra2]  — extra1 absent
        exp2 = empty_graph.record_experiment(
            cv_score=0.86, cv_std=0.01, delta=0.002,
            n_features=2, composite_score=0.858,
            kept=True, features_used=["core", "extra2"],
        )
        empty_graph.register_feature_set(exp2, ["core", "extra2"])
        result = get_load_bearing_features(empty_graph.graph)
        # Only "core" appears in every kept experiment
        assert result == ["core"]

    def test_feature_absent_in_one_kept_experiment_excluded(
        self, empty_graph: MemoryGraph
    ) -> None:
        exp1 = empty_graph.record_experiment(
            cv_score=0.85, cv_std=0.01, delta=0.01,
            n_features=3, composite_score=0.847,
            kept=True, features_used=["a", "b", "c"],
        )
        empty_graph.register_feature_set(exp1, ["a", "b", "c"])
        exp2 = empty_graph.record_experiment(
            cv_score=0.86, cv_std=0.01, delta=0.002,
            n_features=2, composite_score=0.858,
            kept=True, features_used=["a", "b"],
        )
        empty_graph.register_feature_set(exp2, ["a", "b"])
        result = get_load_bearing_features(empty_graph.graph)
        assert "c" not in result
        assert "a" in result and "b" in result

    def test_reverted_experiments_ignored(self, empty_graph: MemoryGraph) -> None:
        # kept with [a, b]; reverted with [a, b, c] — c should NOT be load-bearing
        exp1 = empty_graph.record_experiment(
            cv_score=0.85, cv_std=0.01, delta=0.01,
            n_features=2, composite_score=0.848,
            kept=True, features_used=["a", "b"],
        )
        empty_graph.register_feature_set(exp1, ["a", "b"])
        exp2 = empty_graph.record_experiment(
            cv_score=0.84, cv_std=0.01, delta=0.0,
            n_features=3, composite_score=0.837,
            kept=False, features_used=["a", "b", "c"],
        )
        empty_graph.register_feature_set(exp2, ["a", "b", "c"])
        result = get_load_bearing_features(empty_graph.graph)
        assert "c" not in result

    def test_result_is_sorted(self, empty_graph: MemoryGraph) -> None:
        exp_id = empty_graph.record_experiment(
            cv_score=0.85, cv_std=0.01, delta=0.01,
            n_features=3, composite_score=0.847,
            kept=True, features_used=["zebra", "apple", "mango"],
        )
        empty_graph.register_feature_set(exp_id, ["zebra", "apple", "mango"])
        result = get_load_bearing_features(empty_graph.graph)
        assert result == sorted(result)


# ---------------------------------------------------------------------------
# get_untried_column_transform_pairs (standalone function)
# ---------------------------------------------------------------------------


class TestGetUntriedColumnTransformPairs:
    def test_no_experiments_all_pairs_untried(
        self, graph_with_columns: MemoryGraph
    ) -> None:
        result = get_untried_column_transform_pairs(graph_with_columns.graph)
        # With columns but no experiments, every combination is untried
        assert len(result) > 0
        cols_in_result = {r["column"] for r in result}
        # Every source column should appear
        source_cols = {d["name"] for d in graph_with_columns.get_source_columns()}
        assert cols_in_result == source_cols

    def test_tried_pair_excluded(
        self, graph_with_columns: MemoryGraph
    ) -> None:
        # Record an experiment that tried log on "income"
        graph_with_columns.record_experiment(
            cv_score=0.85, cv_std=0.01, delta=0.01,
            n_features=5, composite_score=0.845,
            kept=True,
            description="col=income; op=log_transform; fit=train_only; reason=skew",
        )
        result = get_untried_column_transform_pairs(graph_with_columns.graph)
        tried_pairs = {(r["column"], r["transform_family"]) for r in result}
        # (income, log) was tried — should not appear
        assert ("income", "log") not in tried_pairs

    def test_untried_column_still_appears(
        self, graph_with_columns: MemoryGraph
    ) -> None:
        # Try log on income only — score should still show up with other families
        graph_with_columns.record_experiment(
            cv_score=0.85, cv_std=0.01, delta=0.01,
            n_features=5, composite_score=0.845,
            kept=True,
            description="col=income; op=log_transform; fit=train_only; reason=skew",
        )
        result = get_untried_column_transform_pairs(graph_with_columns.graph)
        income_pairs = [r for r in result if r["column"] == "income"]
        families_tried_on_income = {r["transform_family"] for r in income_pairs}
        # log was tried; other families should still be in the list
        assert "log" not in families_tried_on_income
        assert len(income_pairs) > 0  # other families remain

    def test_result_contains_expected_keys(
        self, graph_with_columns: MemoryGraph
    ) -> None:
        result = get_untried_column_transform_pairs(graph_with_columns.graph)
        assert len(result) > 0
        assert {"column", "transform_family"} <= result[0].keys()

    def test_empty_graph_no_columns_returns_empty(
        self, empty_graph: MemoryGraph
    ) -> None:
        result = get_untried_column_transform_pairs(empty_graph.graph)
        assert result == []

    def test_multi_column_description_marks_all_as_tried(  # noqa: E501 — keep with class
        self, graph_with_columns: MemoryGraph
    ) -> None:
        # col= lists multiple columns
        graph_with_columns.record_experiment(
            cv_score=0.85, cv_std=0.01, delta=0.01,
            n_features=5, composite_score=0.845,
            kept=True,
            description="col=age,income; op=ratio_feature; fit=train_only; reason=normalize",
        )
        result = get_untried_column_transform_pairs(graph_with_columns.graph)
        tried_pairs = {(r["column"], r["transform_family"]) for r in result}
        assert ("age", "ratio") not in tried_pairs
        assert ("income", "ratio") not in tried_pairs


# ---------------------------------------------------------------------------
# get_shap_ranking (standalone function)
# ---------------------------------------------------------------------------


class TestGetShapRanking:
    def _exp_with_shap(self, mg: MemoryGraph, shap: dict, kept: bool = True) -> int:
        return mg.record_experiment(
            cv_score=0.85, cv_std=0.01, delta=0.01,
            n_features=len(shap), composite_score=0.84,
            kept=kept,
            feature_shap=shap,
            feature_shap_std={k: v * 0.1 for k, v in shap.items()},
        )

    def test_empty_graph_returns_empty(self, empty_graph: MemoryGraph) -> None:
        assert get_shap_ranking(empty_graph.graph) == []

    def test_no_shap_data_returns_empty(self, empty_graph: MemoryGraph) -> None:
        empty_graph.record_experiment(
            cv_score=0.85, cv_std=0.01, delta=0.0,
            n_features=3, composite_score=0.847, kept=True,
        )
        assert get_shap_ranking(empty_graph.graph) == []

    def test_sorted_descending_by_mean_shap(self, empty_graph: MemoryGraph) -> None:
        self._exp_with_shap(empty_graph, {"a": 0.5, "b": 0.1, "c": 0.3})
        result = get_shap_ranking(empty_graph.graph)
        scores = [r["mean_shap"] for r in result]
        assert scores == sorted(scores, reverse=True)
        assert result[0]["feature"] == "a"

    def test_result_contains_required_keys(self, empty_graph: MemoryGraph) -> None:
        self._exp_with_shap(empty_graph, {"x": 0.4, "y": 0.2})
        result = get_shap_ranking(empty_graph.graph)
        assert len(result) == 2
        assert {"feature", "mean_shap", "shap_std"} <= result[0].keys()

    def test_specific_experiment_id_used(self, empty_graph: MemoryGraph) -> None:
        exp1 = self._exp_with_shap(empty_graph, {"a": 0.9, "b": 0.1})
        self._exp_with_shap(empty_graph, {"a": 0.1, "b": 0.9})  # exp2 — best kept
        # Ask for exp1 specifically
        result = get_shap_ranking(empty_graph.graph, experiment_id=exp1)
        assert result[0]["feature"] == "a"  # exp1 has a=0.9

    def test_shap_std_stored_correctly(self, empty_graph: MemoryGraph) -> None:
        empty_graph.record_experiment(
            cv_score=0.85, cv_std=0.01, delta=0.01,
            n_features=2, composite_score=0.848, kept=True,
            feature_shap={"x": 0.4, "y": 0.2},
            feature_shap_std={"x": 0.05, "y": 0.02},
        )
        result = get_shap_ranking(empty_graph.graph)
        by_feat = {r["feature"]: r for r in result}
        assert abs(by_feat["x"]["shap_std"] - 0.05) < 1e-6
        assert abs(by_feat["y"]["shap_std"] - 0.02) < 1e-6

    def test_zero_shap_features_included(self, empty_graph: MemoryGraph) -> None:
        self._exp_with_shap(empty_graph, {"useful": 0.5, "dead": 0.0})
        result = get_shap_ranking(empty_graph.graph)
        features = [r["feature"] for r in result]
        assert "dead" in features
        assert result[-1]["feature"] == "dead"


# ---------------------------------------------------------------------------
# get_shap_consensus (standalone function)
# ---------------------------------------------------------------------------


class TestGetShapConsensus:
    def _add_kept(self, mg: MemoryGraph, shap: dict) -> None:
        mg.record_experiment(
            cv_score=0.85, cv_std=0.01, delta=0.01,
            n_features=len(shap), composite_score=0.84,
            kept=True, feature_shap=shap,
        )

    def test_empty_graph_returns_empty(self, empty_graph: MemoryGraph) -> None:
        assert get_shap_consensus(empty_graph.graph) == []

    def test_no_kept_experiments_returns_empty(self, empty_graph: MemoryGraph) -> None:
        empty_graph.record_experiment(
            cv_score=0.85, cv_std=0.01, delta=0.0,
            n_features=2, composite_score=0.848, kept=False,
            feature_shap={"a": 0.5, "b": 0.3},
        )
        assert get_shap_consensus(empty_graph.graph) == []

    def test_single_experiment_rank_equals_shap_order(
        self, empty_graph: MemoryGraph
    ) -> None:
        self._add_kept(empty_graph, {"alpha": 0.8, "beta": 0.2, "gamma": 0.5})
        result = get_shap_consensus(empty_graph.graph)
        features = [r["feature"] for r in result]
        # alpha (0.8) = rank 1, gamma (0.5) = rank 2, beta (0.2) = rank 3
        assert features.index("alpha") < features.index("gamma")
        assert features.index("gamma") < features.index("beta")

    def test_consistent_top_feature_has_lowest_mean_rank(
        self, empty_graph: MemoryGraph
    ) -> None:
        # "core" is always #1; "extra" varies
        self._add_kept(empty_graph, {"core": 0.9, "extra": 0.1})
        self._add_kept(empty_graph, {"core": 0.85, "extra": 0.7})
        self._add_kept(empty_graph, {"core": 0.95, "extra": 0.3})
        result = get_shap_consensus(empty_graph.graph)
        assert result[0]["feature"] == "core"
        assert result[0]["mean_rank"] == 1.0

    def test_times_in_top3_counted_correctly(self, empty_graph: MemoryGraph) -> None:
        # "a" is top-1 in 2 experiments, absent in 1
        self._add_kept(empty_graph, {"a": 0.9, "b": 0.1, "c": 0.05})
        self._add_kept(empty_graph, {"a": 0.8, "b": 0.2, "c": 0.1})
        self._add_kept(empty_graph, {"b": 0.7, "c": 0.6})  # "a" absent
        result = get_shap_consensus(empty_graph.graph)
        by_feat = {r["feature"]: r for r in result}
        assert by_feat["a"]["times_in_top3"] == 2
        assert by_feat["a"]["n_experiments"] == 3

    def test_result_contains_required_keys(self, empty_graph: MemoryGraph) -> None:
        self._add_kept(empty_graph, {"x": 0.5})
        result = get_shap_consensus(empty_graph.graph)
        assert len(result) == 1
        assert {"feature", "mean_rank", "mean_shap", "times_in_top3", "n_experiments"} <= result[0].keys()

    def test_reverted_experiments_excluded(self, empty_graph: MemoryGraph) -> None:
        self._add_kept(empty_graph, {"good": 0.8})
        empty_graph.record_experiment(
            cv_score=0.80, cv_std=0.01, delta=0.0,
            n_features=2, composite_score=0.798, kept=False,
            feature_shap={"good": 0.1, "bad": 0.9},
        )
        result = get_shap_consensus(empty_graph.graph)
        features = [r["feature"] for r in result]
        # "bad" should not appear — it only occurred in a reverted experiment
        assert "bad" not in features
        assert "good" in features


# ---------------------------------------------------------------------------
# update_active_feature_statuses
# ---------------------------------------------------------------------------


class TestUpdateActiveFeatureStatuses:
    def test_kept_features_are_active(self, empty_graph: MemoryGraph) -> None:
        empty_graph.register_feature("f1", [], experiment_id=1, save=False)
        empty_graph.register_feature("f2", [], experiment_id=1)
        empty_graph.update_active_feature_statuses(["f1"])
        features = {d["name"]: d for d in empty_graph._nodes_of_type("Feature")}
        assert features["f1"]["status"] == "active"
        assert features["f2"]["status"] == "inactive"

    def test_reverted_features_become_inactive(self, empty_graph: MemoryGraph) -> None:
        # First experiment with f1, f2 — kept
        empty_graph.register_feature("f1", [], experiment_id=1, save=False)
        empty_graph.register_feature("f2", [], experiment_id=1)
        empty_graph.update_active_feature_statuses(["f1", "f2"])
        # Second experiment drops f2 — kept
        empty_graph.update_active_feature_statuses(["f1"])
        features = {d["name"]: d for d in empty_graph._nodes_of_type("Feature")}
        assert features["f1"]["status"] == "active"
        assert features["f2"]["status"] == "inactive"

    def test_empty_feature_list_deactivates_all(self, empty_graph: MemoryGraph) -> None:
        empty_graph.register_feature("f1", [], experiment_id=1)
        empty_graph.update_active_feature_statuses([])
        features = {d["name"]: d for d in empty_graph._nodes_of_type("Feature")}
        assert features["f1"]["status"] == "inactive"

    def test_get_active_features_reflects_update(self, empty_graph: MemoryGraph) -> None:
        empty_graph.register_feature("a", [], experiment_id=1, save=False)
        empty_graph.register_feature("b", [], experiment_id=1, save=False)
        empty_graph.register_feature("c", [], experiment_id=1)
        empty_graph.update_active_feature_statuses(["a", "c"])
        active_names = {d["name"] for d in empty_graph.get_active_features()}
        assert active_names == {"a", "c"}
        assert "b" not in active_names


# ---------------------------------------------------------------------------
# add_hypothesis / resolve_hypothesis
# ---------------------------------------------------------------------------


class TestHypothesisWorkflow:
    def test_add_hypothesis_returns_id(self, empty_graph: MemoryGraph) -> None:
        hyp_id = empty_graph.add_hypothesis("log of price helps", predicted_direction="+")
        assert isinstance(hyp_id, int) and hyp_id >= 1

    def test_pending_hypothesis_has_no_edge(self, empty_graph: MemoryGraph) -> None:
        empty_graph.add_hypothesis("test", predicted_direction="+")
        hyps = empty_graph.get_active_hypotheses()
        assert len(hyps) == 1
        assert hyps[0]["edge_type"] is None
        assert hyps[0]["validated"] is False

    def test_resolve_correct_prediction_gives_supports(self, empty_graph: MemoryGraph) -> None:
        empty_graph.record_experiment(
            cv_score=0.85, cv_std=0.01, delta=0.01,
            n_features=5, composite_score=0.845, kept=True,
        )
        hyp_id = empty_graph.add_hypothesis("will improve", predicted_direction="+")
        empty_graph.resolve_hypothesis(hyp_id, experiment_id=1, kept=True, actual_delta=0.01)
        hyps = empty_graph.get_active_hypotheses()
        assert hyps[0]["edge_type"] == "SUPPORTS"
        assert hyps[0]["validated"] is True

    def test_resolve_wrong_prediction_gives_contradicts(self, empty_graph: MemoryGraph) -> None:
        empty_graph.record_experiment(
            cv_score=0.80, cv_std=0.01, delta=-0.02,
            n_features=5, composite_score=0.795, kept=False,
        )
        hyp_id = empty_graph.add_hypothesis("will improve", predicted_direction="+")
        empty_graph.resolve_hypothesis(hyp_id, experiment_id=1, kept=False, actual_delta=-0.02)
        hyps = empty_graph.get_active_hypotheses()
        assert hyps[0]["edge_type"] == "CONTRADICTS"

    def test_superseded_hypothesis_excluded(self, empty_graph: MemoryGraph) -> None:
        h1 = empty_graph.add_hypothesis("old idea", predicted_direction="+")
        h2 = empty_graph.add_hypothesis("better idea", predicted_direction="+")
        empty_graph.supersede_hypothesis(old_hyp_id=h1, new_hyp_id=h2)
        hyps = empty_graph.get_active_hypotheses()
        assert len(hyps) == 1
        assert hyps[0]["hyp_id"] == h2

    def test_explore_prediction_always_supports(self, empty_graph: MemoryGraph) -> None:
        empty_graph.record_experiment(
            cv_score=0.80, cv_std=0.01, delta=-0.02,
            n_features=5, composite_score=0.795, kept=False,
        )
        hyp_id = empty_graph.add_hypothesis("just exploring", predicted_direction="?")
        empty_graph.resolve_hypothesis(hyp_id, experiment_id=1, kept=False, actual_delta=-0.02)
        hyps = empty_graph.get_active_hypotheses()
        assert hyps[0]["edge_type"] == "SUPPORTS"


# ---------------------------------------------------------------------------
# Relationship registry
# ---------------------------------------------------------------------------


class TestRelationshipRegistry:
    def test_get_relationship_types_returns_builtins(self, empty_graph: MemoryGraph) -> None:
        types = empty_graph.get_relationship_types()
        type_names = {t["rel_type"] for t in types}
        assert {"DERIVED_FROM", "USED_IN", "IMPROVED_OVER", "SUPPORTS", "CONTRADICTS", "SUPERSEDES"} <= type_names

    def test_filter_by_category(self, empty_graph: MemoryGraph) -> None:
        hyp_types = empty_graph.get_relationship_types(category="hypothesis")
        type_names = {t["rel_type"] for t in hyp_types}
        assert type_names == {"SUPPORTS", "CONTRADICTS", "SUPERSEDES"}

    def test_get_rel_types_for_category(self, empty_graph: MemoryGraph) -> None:
        lineage = empty_graph.get_rel_types_for_category("lineage")
        assert lineage == {"DERIVED_FROM"}

    def test_register_relationship_type_adds_to_registry(self, empty_graph: MemoryGraph) -> None:
        empty_graph.register_relationship_type(
            rel_type="CORRELATED_WITH",
            description="Feature correlation",
            source_type="Feature",
            target_type=["Feature"],
            category="correlation",
        )
        types = empty_graph.get_relationship_types()
        type_names = {t["rel_type"] for t in types}
        assert "CORRELATED_WITH" in type_names

    def test_register_duplicate_is_idempotent(self, empty_graph: MemoryGraph) -> None:
        empty_graph.register_relationship_type(
            rel_type="CUSTOM_REL", description="test", source_type="Feature",
            target_type=["Feature"], category="custom",
        )
        empty_graph.register_relationship_type(
            rel_type="CUSTOM_REL", description="different desc", source_type="Feature",
            target_type=["Feature"], category="custom",
        )
        matching = [t for t in empty_graph.get_relationship_types() if t["rel_type"] == "CUSTOM_REL"]
        assert len(matching) == 1
        assert matching[0]["description"] == "test"  # first registration wins

    def test_get_edges_by_type_returns_matching(self, empty_graph: MemoryGraph) -> None:
        empty_graph.record_experiment(
            cv_score=0.80, cv_std=0.01, delta=0.0, n_features=3,
            composite_score=0.797, kept=True,
        )
        empty_graph.record_experiment(
            cv_score=0.82, cv_std=0.01, delta=0.02, n_features=3,
            composite_score=0.817, kept=True,
        )
        edges = empty_graph.get_edges_by_type("IMPROVED_OVER")
        assert len(edges) == 1
        assert edges[0][2]["rel"] == "IMPROVED_OVER"

    def test_get_edges_by_type_empty_for_unknown(self, empty_graph: MemoryGraph) -> None:
        assert empty_graph.get_edges_by_type("NONEXISTENT") == []

    def test_add_edge_typed_creates_edge(self, empty_graph: MemoryGraph) -> None:
        empty_graph.register_feature("f1", [], experiment_id=1)
        empty_graph.register_feature("f2", [], experiment_id=1)
        result = empty_graph.add_edge_typed("feat_f1", "feat_f2", "CORRELATED_WITH", correlation=0.95)
        assert result is True
        assert empty_graph.graph.has_edge("feat_f1", "feat_f2")

    def test_add_edge_typed_auto_registers_unknown(self, empty_graph: MemoryGraph) -> None:
        empty_graph.register_feature("f1", [], experiment_id=1)
        empty_graph.register_feature("f2", [], experiment_id=1)
        empty_graph.add_edge_typed("feat_f1", "feat_f2", "NEW_REL_TYPE")
        type_names = {t["rel_type"] for t in empty_graph.get_relationship_types()}
        assert "NEW_REL_TYPE" in type_names

    def test_add_edge_typed_returns_false_for_missing_nodes(self, empty_graph: MemoryGraph) -> None:
        result = empty_graph.add_edge_typed("nonexistent_a", "nonexistent_b", "SOME_REL")
        assert result is False


# ---------------------------------------------------------------------------
# register_feature_set
# ---------------------------------------------------------------------------


class TestRegisterFeatureSet:
    def test_creates_used_in_edges(self, empty_graph: MemoryGraph) -> None:
        exp_id = empty_graph.record_experiment(
            cv_score=0.85, cv_std=0.01, delta=0.0, n_features=2,
            composite_score=0.848, kept=True, features_used=["f1", "f2"],
        )
        empty_graph.register_feature_set(exp_id, ["f1", "f2"])
        assert empty_graph.graph.has_edge("feat_f1", f"exp_{exp_id}")
        assert empty_graph.graph.has_edge("feat_f2", f"exp_{exp_id}")
        assert empty_graph.graph["feat_f1"][f"exp_{exp_id}"]["rel"] == "USED_IN"

    def test_creates_feature_nodes_if_missing(self, empty_graph: MemoryGraph) -> None:
        exp_id = empty_graph.record_experiment(
            cv_score=0.85, cv_std=0.01, delta=0.0, n_features=1,
            composite_score=0.849, kept=True,
        )
        empty_graph.register_feature_set(exp_id, ["new_feat"])
        assert empty_graph.graph.has_node("feat_new_feat")
        assert empty_graph.graph.nodes["feat_new_feat"]["node_type"] == "Feature"

    def test_idempotent(self, empty_graph: MemoryGraph) -> None:
        exp_id = empty_graph.record_experiment(
            cv_score=0.85, cv_std=0.01, delta=0.0, n_features=1,
            composite_score=0.849, kept=True,
        )
        empty_graph.register_feature_set(exp_id, ["f1"])
        edge_count_before = empty_graph.graph.number_of_edges()
        empty_graph.register_feature_set(exp_id, ["f1"])
        assert empty_graph.graph.number_of_edges() == edge_count_before

    def test_noop_if_experiment_missing(self, empty_graph: MemoryGraph) -> None:
        node_count_before = empty_graph.graph.number_of_nodes()
        empty_graph.register_feature_set(999, ["f1", "f2"])
        assert empty_graph.graph.number_of_nodes() == node_count_before


# ---------------------------------------------------------------------------
# get_best_experiment
# ---------------------------------------------------------------------------


class TestGetBestExperiment:
    def test_empty_graph_returns_none(self, empty_graph: MemoryGraph) -> None:
        assert empty_graph.get_best_experiment() is None

    def test_higher_is_better_returns_max(self, empty_graph: MemoryGraph) -> None:
        empty_graph.record_experiment(
            cv_score=0.80, cv_std=0.01, delta=0.0, n_features=3,
            composite_score=0.797, kept=True,
        )
        empty_graph.record_experiment(
            cv_score=0.85, cv_std=0.01, delta=0.05, n_features=3,
            composite_score=0.847, kept=True,
        )
        best = empty_graph.get_best_experiment(is_higher_better=True)
        assert best is not None
        assert best["composite_score"] == 0.847

    def test_lower_is_better_returns_min(self, empty_graph: MemoryGraph) -> None:
        empty_graph.record_experiment(
            cv_score=0.50, cv_std=0.01, delta=0.0, n_features=3,
            composite_score=0.503, kept=True,
        )
        empty_graph.record_experiment(
            cv_score=0.45, cv_std=0.01, delta=-0.05, n_features=3,
            composite_score=0.453, kept=True,
        )
        best = empty_graph.get_best_experiment(is_higher_better=False)
        assert best is not None
        assert best["composite_score"] == 0.453


# ---------------------------------------------------------------------------
# get_next_experiment_id
# ---------------------------------------------------------------------------


class TestGetNextExperimentId:
    def test_empty_graph_returns_1(self, empty_graph: MemoryGraph) -> None:
        assert empty_graph.get_next_experiment_id() == 1

    def test_after_recording_returns_max_plus_1(self, empty_graph: MemoryGraph) -> None:
        empty_graph.record_experiment(
            cv_score=0.80, cv_std=0.01, delta=0.0, n_features=3,
            composite_score=0.797, kept=True,
        )
        empty_graph.record_experiment(
            cv_score=0.82, cv_std=0.01, delta=0.02, n_features=3,
            composite_score=0.817, kept=True,
        )
        assert empty_graph.get_next_experiment_id() == 3


# ---------------------------------------------------------------------------
# get_failed_patterns
# ---------------------------------------------------------------------------


class TestGetFailedPatterns:
    def test_returns_descriptions_of_reverted(self, empty_graph: MemoryGraph) -> None:
        empty_graph.record_experiment(
            cv_score=0.80, cv_std=0.01, delta=0.0, n_features=3,
            composite_score=0.797, kept=False, description="bad idea 1",
        )
        empty_graph.record_experiment(
            cv_score=0.79, cv_std=0.01, delta=-0.01, n_features=3,
            composite_score=0.787, kept=False, description="bad idea 2",
        )
        patterns = empty_graph.get_failed_patterns()
        assert "bad idea 1" in patterns
        assert "bad idea 2" in patterns

    def test_empty_graph_returns_empty(self, empty_graph: MemoryGraph) -> None:
        assert empty_graph.get_failed_patterns() == []


# ---------------------------------------------------------------------------
# get_feature_set_diff
# ---------------------------------------------------------------------------


class TestGetFeatureSetDiff:
    def test_returns_tried_and_untried(self, graph_with_columns: MemoryGraph) -> None:
        graph_with_columns.register_feature("age_log", ["age"], experiment_id=1)
        diff = graph_with_columns.get_feature_set_diff()
        assert diff["n_features_tried"] >= 1
        assert "age" in diff["columns_tried"]
        assert "income" in diff["columns_untried"]

    def test_empty_graph_returns_zeros(self, empty_graph: MemoryGraph) -> None:
        diff = empty_graph.get_feature_set_diff()
        assert diff["n_features_tried"] == 0
        assert diff["n_features_active"] == 0
        assert diff["columns_tried"] == []
        assert diff["columns_untried"] == []


# ---------------------------------------------------------------------------
# get_source_columns
# ---------------------------------------------------------------------------


class TestGetSourceColumns:
    def test_returns_all_column_nodes(
        self, graph_with_columns: MemoryGraph, sample_df: pd.DataFrame
    ) -> None:
        cols = graph_with_columns.get_source_columns()
        col_names = {c["name"] for c in cols}
        assert col_names == set(sample_df.columns)


# ---------------------------------------------------------------------------
# ensure_dataset_signature
# ---------------------------------------------------------------------------


class TestEnsureDatasetSignature:
    def test_first_call_sets_signature(self, empty_graph: MemoryGraph) -> None:
        result = empty_graph.ensure_dataset_signature("sig_abc")
        assert result is False  # no reset
        assert empty_graph.graph.graph["dataset_signature"] == "sig_abc"

    def test_same_signature_returns_false(self, empty_graph: MemoryGraph) -> None:
        empty_graph.ensure_dataset_signature("sig_abc")
        result = empty_graph.ensure_dataset_signature("sig_abc")
        assert result is False

    def test_different_signature_resets_graph(self, empty_graph: MemoryGraph) -> None:
        empty_graph.ensure_dataset_signature("sig_abc")
        empty_graph.record_experiment(
            cv_score=0.80, cv_std=0.01, delta=0.0, n_features=3,
            composite_score=0.797, kept=True,
        )
        result = empty_graph.ensure_dataset_signature("sig_xyz", backup_on_change=False)
        assert result is True
        assert empty_graph.get_experiment_history() == []
        assert empty_graph.graph.graph["dataset_signature"] == "sig_xyz"


# ---------------------------------------------------------------------------
# JSON round-trip datetime
# ---------------------------------------------------------------------------


class TestJsonRoundTripDatetime:
    def test_timestamps_survive_roundtrip(self, tmp_path: Path) -> None:
        graph_path = tmp_path / "dt.json"
        mg1 = MemoryGraph(path=graph_path)
        mg1.record_experiment(
            cv_score=0.85, cv_std=0.01, delta=0.0, n_features=3,
            composite_score=0.847, kept=True,
        )
        mg1.save()
        mg2 = MemoryGraph(path=graph_path)
        exp = mg2.get_experiment_history()[0]
        assert "timestamp" in exp
        assert isinstance(exp["timestamp"], str)
        assert len(exp["timestamp"]) > 10

    def test_fromisoformat_parses_timestamps(self, tmp_path: Path) -> None:
        from datetime import datetime
        graph_path = tmp_path / "dt2.json"
        mg = MemoryGraph(path=graph_path)
        mg.record_experiment(
            cv_score=0.85, cv_std=0.01, delta=0.0, n_features=3,
            composite_score=0.847, kept=True,
        )
        mg.save()
        mg2 = MemoryGraph(path=graph_path)
        ts = mg2.get_experiment_history()[0]["timestamp"]
        parsed = datetime.fromisoformat(ts)
        assert isinstance(parsed, datetime)


# ---------------------------------------------------------------------------
# ensure_dataset_signature
# ---------------------------------------------------------------------------


class TestEnsureDatasetSignature:
    def test_first_call_stores_signature(self, empty_graph: MemoryGraph) -> None:
        """First call with no existing signature stores it, returns False."""
        result = empty_graph.ensure_dataset_signature("abc123")
        assert result is False
        assert empty_graph.graph.graph["dataset_signature"] == "abc123"

    def test_same_signature_no_change(self, empty_graph: MemoryGraph) -> None:
        """Second call with same signature returns False (no reset)."""
        empty_graph.ensure_dataset_signature("abc123")
        result = empty_graph.ensure_dataset_signature("abc123")
        assert result is False

    def test_different_signature_resets_graph(self, empty_graph: MemoryGraph) -> None:
        """Different signature resets the graph and returns True."""
        empty_graph.ensure_dataset_signature("abc123")
        # Add some data that should be wiped
        empty_graph.record_experiment(
            cv_score=0.8, cv_std=0.01, delta=0.0,
            n_features=3, composite_score=0.797, kept=True,
        )
        result = empty_graph.ensure_dataset_signature("xyz789")
        assert result is True
        assert empty_graph.graph.graph["dataset_signature"] == "xyz789"
        # Graph should be reset — no experiment nodes
        assert empty_graph.get_experiment_history() == []

    def test_stores_meta(self, empty_graph: MemoryGraph) -> None:
        """Meta dict is stored alongside the signature."""
        meta = {"rows": 1000, "cols": 5}
        empty_graph.ensure_dataset_signature("abc123", meta=meta)
        assert empty_graph.graph.graph["dataset_meta"] == meta

    def test_backs_up_old_graph(self, tmp_path: Path) -> None:
        """On signature change, old graph file is backed up."""
        graph_path = tmp_path / "db" / "memory_graph.json"
        graph_path.parent.mkdir(parents=True)
        mg = MemoryGraph(path=graph_path)
        mg.ensure_dataset_signature("first_sig")
        mg.record_experiment(
            cv_score=0.8, cv_std=0.01, delta=0.0,
            n_features=3, composite_score=0.797, kept=True,
        )
        # Change signature — should create backup
        mg.ensure_dataset_signature("second_sig")
        backup_files = list(graph_path.parent.glob("memory_graph.first_si*.json"))
        assert len(backup_files) == 1
