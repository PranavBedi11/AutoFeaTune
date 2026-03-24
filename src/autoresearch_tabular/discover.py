"""discover.py — Data discovery: entity key detection, invariant scanning, residual analysis.

Populates DerivedColumn and EntityKey nodes in the memory graph.
Called via: uv run autoresearch discover

THIS FILE IS PART OF THE FIXED HARNESS. THE AGENT NEVER TOUCHES IT.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder

from autoresearch_tabular.config import load_config
from autoresearch_tabular.memory_graph import MemoryGraph, load_graph

__all__ = ["run_discovery"]

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_DERIVED_COLS = 50
MAX_ENTITY_KEYS = 20
MAX_FALLBACK_COLS = 30      # Cap input columns before pairing in fallback
CV_THRESHOLD = 0.1
MIN_GROUP_SIZE = 10
SAMPLE_THRESHOLD = 100_000
SAMPLE_FRAC = 0.10
RESIDUAL_TREES = 50

# Semantic keyword groups for program.md parsing
SEMANTIC_KEYWORDS: dict[str, list[str]] = {
    "temporal": ["seconds", "timestamp", "timedelta", "days", "datetime", "time", "date", "elapsed"],
    "monetary": ["usd", "amount", "price", "cost", "dollar", "value", "fee", "payment"],
    "count": ["count", "frequency", "number of", "how many"],
    "distance": ["distance", "dist", "miles", "km", "meters"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_id_derived(expr: str) -> str:
    """Stable node ID for a derived expression."""
    return f"derived_{hashlib.sha256(expr.encode()).hexdigest()[:8]}"


def _node_id_entity(columns: tuple[str, ...]) -> str:
    """Stable node ID for an entity key."""
    return f"entity_{hashlib.sha256(str(sorted(columns)).encode()).hexdigest()[:8]}"


# ---------------------------------------------------------------------------
# 2a. Extended column profiling
# ---------------------------------------------------------------------------

def profile_columns(X_train: pd.DataFrame, mg: MemoryGraph) -> None:
    """Update existing Column nodes with richer univariate stats."""
    G = mg.graph

    # Missingness pattern clustering
    miss_matrix = X_train.isna().astype(float)
    cols_with_missing = [c for c in X_train.columns if miss_matrix[c].sum() > 0]

    missingness_groups: dict[str, int] = {}
    if len(cols_with_missing) >= 2:
        miss_corr = miss_matrix[cols_with_missing].corr().fillna(0)
        dist = (1 - miss_corr.values) / 2  # Convert correlation to distance
        np.fill_diagonal(dist, 0)
        dist = np.clip(dist, 0, None)  # Ensure non-negative

        n_clusters = min(10, len(cols_with_missing))
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage="complete",
        )
        labels = clustering.fit_predict(dist)
        for col, label in zip(cols_with_missing, labels):
            missingness_groups[col] = int(label)

    for col in X_train.columns:
        col_id = f"col_{col}"
        if not G.has_node(col_id):
            continue

        updates: dict[str, Any] = {}

        if pd.api.types.is_numeric_dtype(X_train[col]) and not pd.api.types.is_bool_dtype(X_train[col]):
            valid = X_train[col].dropna()
            if len(valid) > 0:
                quantiles = valid.quantile([0.05, 0.25, 0.75, 0.95]).to_dict()
                updates["p5"] = float(quantiles.get(0.05, 0))
                updates["p25"] = float(quantiles.get(0.25, 0))
                updates["p75"] = float(quantiles.get(0.75, 0))
                updates["p95"] = float(quantiles.get(0.95, 0))
                updates["skewness"] = float(valid.skew())

        # Top-5 most frequent values
        top5 = X_train[col].value_counts().head(5)
        updates["top5_values"] = list(top5.index.astype(str))
        updates["top5_counts"] = [int(c) for c in top5.values]

        # Missingness group
        if col in missingness_groups:
            updates["missingness_group"] = missingness_groups[col]

        G.nodes[col_id].update(updates)

    mg.save()
    print(f"   Profiled {len(X_train.columns)} columns.")


# ---------------------------------------------------------------------------
# 2h. Main entry point
# ---------------------------------------------------------------------------

def run_discovery() -> dict[str, Any]:
    """Run the full discovery pipeline. Idempotent — safe to re-run."""
    t0 = time.time()

    print("Running data discovery ...")

    # Initialize data pipeline (same pattern as cmd_prepare)
    import autoresearch_tabular.prepare as prepare
    prepare._initialize()

    config = load_config()
    folds = prepare.get_folds()
    X_train, _, y_train, _ = folds[0]

    print(f"   X_train: {X_train.shape[0]:,} rows × {X_train.shape[1]} columns")

    # Load graph and clear previous discovery nodes
    mg = load_graph()
    n_cleared = mg.clear_discovery_nodes()
    if n_cleared:
        print(f"   Cleared {n_cleared} previous discovery nodes.")

    # Also clear query log for new session
    query_log = PROJECT_ROOT / "db" / "query_log.json"
    if query_log.exists():
        query_log.unlink()

    # 2a. Extended column profiling
    print("\n[1/5] Profiling columns ...")
    profile_columns(X_train, mg)

    elapsed = time.time() - t0
    print(f"\nDiscovery complete in {elapsed:.1f}s.")

    return {
        "elapsed_seconds": elapsed,
    }
