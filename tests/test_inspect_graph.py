"""test_inspect_graph.py — Tests for inspect_graph analytics, stopping, repair, and reports."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from autoresearch_tabular.memory_graph import MemoryGraph
from autoresearch_tabular.inspect_graph import (
    consecutive_failures,
    get_best_features_for_column,
    get_compressed_history,
    get_diminishing_returns_signal,
    repair_graph,
    report_ablation,
    report_hypotheses,
    report_saturated_columns,
    report_shap,
    report_timeline,
    should_stop,
)


# ---------------------------------------------------------------------------
# get_diminishing_returns_signal
# ---------------------------------------------------------------------------


class TestGetDiminishingReturnsSignal:
    def test_empty_graph_returns_zero(self, empty_graph: MemoryGraph) -> None:
        assert get_diminishing_returns_signal(empty_graph.graph) == 0.0

    def test_fewer_than_5_real_experiments_returns_zero(self, empty_graph: MemoryGraph) -> None:
        for i in range(4):
            empty_graph.record_experiment(
                cv_score=0.80 + i * 0.01, cv_std=0.01, delta=0.01,
                n_features=3, composite_score=0.797 + i * 0.01, kept=True,
            )
        assert get_diminishing_returns_signal(empty_graph.graph) == 0.0

    def test_all_zero_deltas_returns_high_value(self, empty_graph: MemoryGraph) -> None:
        for i in range(6):
            empty_graph.record_experiment(
                cv_score=0.80, cv_std=0.01, delta=0.0,
                n_features=3, composite_score=0.797, kept=False,
            )
        signal = get_diminishing_returns_signal(empty_graph.graph)
        assert signal == 1.0

    def test_mixed_deltas_returns_in_range(self, empty_graph: MemoryGraph) -> None:
        # First experiment kept to seed best_comp
        empty_graph.record_experiment(
            cv_score=0.80, cv_std=0.01, delta=0.0,
            n_features=3, composite_score=0.797, kept=True,
        )
        for i in range(6):
            empty_graph.record_experiment(
                cv_score=0.80 + i * 0.005, cv_std=0.01, delta=0.005,
                n_features=3, composite_score=0.797 + i * 0.005,
                kept=(i % 2 == 0),
            )
        signal = get_diminishing_returns_signal(empty_graph.graph)
        assert 0.0 <= signal <= 1.0


# ---------------------------------------------------------------------------
# consecutive_failures
# ---------------------------------------------------------------------------


class TestConsecutiveFailures:
    def test_empty_graph_returns_zero(self, empty_graph: MemoryGraph) -> None:
        assert consecutive_failures(empty_graph.graph) == 0

    def test_all_kept_returns_zero(self, empty_graph: MemoryGraph) -> None:
        for _ in range(3):
            empty_graph.record_experiment(
                cv_score=0.80, cv_std=0.01, delta=0.01,
                n_features=3, composite_score=0.797, kept=True,
            )
        assert consecutive_failures(empty_graph.graph) == 0

    def test_trailing_failures_counted(self, empty_graph: MemoryGraph) -> None:
        empty_graph.record_experiment(
            cv_score=0.80, cv_std=0.01, delta=0.0, n_features=3,
            composite_score=0.797, kept=True,
        )
        for _ in range(3):
            empty_graph.record_experiment(
                cv_score=0.79, cv_std=0.01, delta=-0.01, n_features=3,
                composite_score=0.787, kept=False,
            )
        assert consecutive_failures(empty_graph.graph) == 3

    def test_streak_broken_by_kept(self, empty_graph: MemoryGraph) -> None:
        empty_graph.record_experiment(
            cv_score=0.80, cv_std=0.01, delta=0.0, n_features=3,
            composite_score=0.797, kept=False,
        )
        empty_graph.record_experiment(
            cv_score=0.82, cv_std=0.01, delta=0.02, n_features=3,
            composite_score=0.817, kept=True,
        )
        empty_graph.record_experiment(
            cv_score=0.81, cv_std=0.01, delta=-0.01, n_features=3,
            composite_score=0.807, kept=False,
        )
        assert consecutive_failures(empty_graph.graph) == 1


# ---------------------------------------------------------------------------
# should_stop
# ---------------------------------------------------------------------------


class TestShouldStop:
    def test_no_experiments_returns_continue(self, empty_graph: MemoryGraph) -> None:
        stop, reason = should_stop(empty_graph.graph)
        assert stop is False
        assert "CONTINUE" in reason

    def test_consecutive_failures_exceed_max(self, empty_graph: MemoryGraph) -> None:
        for _ in range(6):
            empty_graph.record_experiment(
                cv_score=0.79, cv_std=0.01, delta=-0.01, n_features=3,
                composite_score=0.787, kept=False,
            )
        stop, reason = should_stop(empty_graph.graph, max_consecutive_failures=5)
        assert stop is True
        assert "STOP" in reason

    def test_normal_state_returns_continue(self, empty_graph: MemoryGraph) -> None:
        empty_graph.record_experiment(
            cv_score=0.80, cv_std=0.01, delta=0.0, n_features=3,
            composite_score=0.797, kept=True,
        )
        stop, reason = should_stop(empty_graph.graph)
        assert stop is False
        assert "CONTINUE" in reason

    def test_time_budget_exceeded(self, empty_graph: MemoryGraph) -> None:
        # Record experiments within the same session (gaps < 60 min) but session start > budget
        now = datetime.now()
        for i in range(3):
            exp_id = empty_graph.record_experiment(
                cv_score=0.80 + i * 0.01, cv_std=0.01, delta=0.01,
                n_features=3, composite_score=0.797 + i * 0.01, kept=True,
            )
            # Space experiments 30 mins apart, starting 90 mins ago
            ts = (now - timedelta(minutes=90 - i * 30)).isoformat()
            empty_graph.graph.nodes[f"exp_{exp_id}"]["timestamp"] = ts
        stop, reason = should_stop(empty_graph.graph, time_budget_minutes=60)
        assert stop is True
        assert "STOP" in reason


# ---------------------------------------------------------------------------
# get_best_features_for_column
# ---------------------------------------------------------------------------


class TestGetBestFeaturesForColumn:
    def test_unknown_column_returns_empty(self, empty_graph: MemoryGraph) -> None:
        assert get_best_features_for_column(empty_graph.graph, "nonexistent") == []

    def test_returns_features_from_column(
        self, graph_with_experiments: MemoryGraph
    ) -> None:
        features = get_best_features_for_column(graph_with_experiments.graph, "age")
        feat_names = {f["name"] for f in features}
        assert "age_log" in feat_names

    def test_includes_all_derived_features(
        self, graph_with_experiments: MemoryGraph
    ) -> None:
        features = get_best_features_for_column(graph_with_experiments.graph, "income")
        feat_names = {f["name"] for f in features}
        assert "income_scaled" in feat_names


# ---------------------------------------------------------------------------
# get_compressed_history
# ---------------------------------------------------------------------------


class TestGetCompressedHistory:
    def test_empty_graph_returns_string(self, empty_graph: MemoryGraph) -> None:
        result = get_compressed_history(empty_graph.graph, empty_graph)
        assert isinstance(result, str)

    def test_with_experiments_contains_references(
        self, graph_with_experiments: MemoryGraph
    ) -> None:
        result = get_compressed_history(
            graph_with_experiments.graph, graph_with_experiments
        )
        assert "exp_" in result
        assert "RECENT" in result


# ---------------------------------------------------------------------------
# repair_graph
# ---------------------------------------------------------------------------


class TestRepairGraph:
    def test_backfills_missing_exp_id(self, empty_graph: MemoryGraph) -> None:
        # Manually add experiment node without exp_id attribute
        empty_graph.graph.add_node("exp_99", node_type="Experiment", kept=True)
        changed = repair_graph(empty_graph)
        assert changed is True
        assert empty_graph.graph.nodes["exp_99"]["exp_id"] == 99

    def test_adds_missing_improved_over_edges(self, empty_graph: MemoryGraph) -> None:
        empty_graph.record_experiment(
            cv_score=0.80, cv_std=0.01, delta=0.0, n_features=3,
            composite_score=0.797, kept=True,
        )
        empty_graph.record_experiment(
            cv_score=0.82, cv_std=0.01, delta=0.02, n_features=3,
            composite_score=0.817, kept=True,
        )
        # Remove the IMPROVED_OVER edge that was auto-created
        if empty_graph.graph.has_edge("exp_2", "exp_1"):
            empty_graph.graph.remove_edge("exp_2", "exp_1")
        changed = repair_graph(empty_graph)
        assert changed is True
        assert empty_graph.graph.has_edge("exp_2", "exp_1")

    def test_backfills_used_in_edges(self, empty_graph: MemoryGraph) -> None:
        empty_graph.record_experiment(
            cv_score=0.80, cv_std=0.01, delta=0.0, n_features=2,
            composite_score=0.798, kept=True,
            features_used=["feat_a", "feat_b"],
        )
        # Don't call register_feature_set — so no USED_IN edges exist
        assert not empty_graph.graph.has_node("feat_feat_a")
        changed = repair_graph(empty_graph)
        assert changed is True
        assert empty_graph.graph.has_node("feat_feat_a")
        assert empty_graph.graph.has_edge("feat_feat_a", "exp_1")
        assert empty_graph.graph["feat_feat_a"]["exp_1"]["rel"] == "USED_IN"

    def test_already_repaired_returns_false(self, empty_graph: MemoryGraph) -> None:
        empty_graph.record_experiment(
            cv_score=0.80, cv_std=0.01, delta=0.0, n_features=1,
            composite_score=0.799, kept=True, features_used=["f1"],
        )
        empty_graph.register_feature_set(1, ["f1"])
        # First repair should be no-op since everything is already correct
        changed = repair_graph(empty_graph)
        assert changed is False


# ---------------------------------------------------------------------------
# Report functions (capsys stdout capture)
# ---------------------------------------------------------------------------


class TestReportFunctions:
    def test_report_timeline(self, graph_with_experiments: MemoryGraph, capsys) -> None:
        report_timeline(graph_with_experiments.graph)
        captured = capsys.readouterr()
        assert "EXPERIMENT TIMELINE" in captured.out
        assert "exp_" in captured.out

    def test_report_ablation(self, graph_with_experiments: MemoryGraph, capsys) -> None:
        report_ablation(graph_with_experiments.graph)
        captured = capsys.readouterr()
        assert "ABLATION" in captured.out

    def test_report_saturated_columns(
        self, graph_with_experiments: MemoryGraph, capsys
    ) -> None:
        report_saturated_columns(graph_with_experiments.graph)
        captured = capsys.readouterr()
        assert "SATURATED" in captured.out

    def test_report_shap(self, graph_with_experiments: MemoryGraph, capsys) -> None:
        report_shap(graph_with_experiments.graph)
        captured = capsys.readouterr()
        assert "SHAP" in captured.out

    def test_report_hypotheses(
        self, graph_with_experiments: MemoryGraph, capsys
    ) -> None:
        report_hypotheses(graph_with_experiments)
        captured = capsys.readouterr()
        assert "HYPOTHESES" in captured.out
        assert "SUPPORTS" in captured.out
