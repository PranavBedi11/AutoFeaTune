"""query.py — Safe statistical query tool with leakage protection and rate limiting.

Called via: uv run autoresearch query <query_type> [args]

Query types:
    within_group_variance --expr "..." --groupby col1 col2
    cardinality --cols col1 col2
    correlation --col_a X --col_b Y
    conditional_distribution --col X --groupby Y --n_groups 5

Safety: rejects queries referencing the target column.
Rate limit: max 20 queries per session (resets when ``autoresearch discover`` runs).

THIS FILE IS PART OF THE FIXED HARNESS. THE AGENT NEVER TOUCHES IT.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from autoresearch_tabular.config import load_config

__all__ = ["run_query"]

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent.parent
QUERY_LOG_PATH = PROJECT_ROOT / "db" / "query_log.json"
MAX_QUERIES_PER_SESSION = 20


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

def _check_leakage(columns: list[str], target: str) -> None:
    """Raise ValueError if any column matches the target."""
    for col in columns:
        if col == target:
            raise ValueError(
                f"LEAKAGE BLOCKED: column '{col}' is the target variable. "
                f"Queries referencing the target are not allowed."
            )


def _check_rate_limit() -> int:
    """Check rate limit. Returns remaining quota. Raises RuntimeError if exhausted."""
    entries: list[dict] = []
    if QUERY_LOG_PATH.exists():
        entries = json.loads(QUERY_LOG_PATH.read_text())

    used = len(entries)
    remaining = MAX_QUERIES_PER_SESSION - used

    if remaining <= 0:
        raise RuntimeError(
            f"RATE LIMIT: {MAX_QUERIES_PER_SESSION} queries per session exhausted. "
            f"Run 'uv run autoresearch discover' to reset."
        )

    return remaining


def _log_query(query_type: str) -> None:
    """Append a query record to the log."""
    entries: list[dict] = []
    if QUERY_LOG_PATH.exists():
        entries = json.loads(QUERY_LOG_PATH.read_text())

    entries.append({
        "timestamp": datetime.now().isoformat(),
        "query_type": query_type,
    })

    QUERY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUERY_LOG_PATH.write_text(json.dumps(entries, indent=2))


def _safe_eval(expr: str, X_train: pd.DataFrame) -> pd.Series:
    """Evaluate an expression safely using only column references."""
    # Build restricted namespace: only column Series, no builtins
    local_dict = {col: X_train[col] for col in X_train.columns}
    # Also add numpy functions commonly used in expressions
    local_dict["floor"] = np.floor
    local_dict["log"] = np.log
    local_dict["log1p"] = np.log1p
    local_dict["abs"] = np.abs
    local_dict["sqrt"] = np.sqrt

    return pd.eval(expr, local_dict=local_dict, global_dict={"__builtins__": {}})


# ---------------------------------------------------------------------------
# Query implementations
# ---------------------------------------------------------------------------

def _query_within_group_variance(
    X_train: pd.DataFrame,
    expr: str,
    groupby: list[str],
) -> dict[str, Any]:
    """Check if an expression is near-constant within groups."""
    series = _safe_eval(expr, X_train)

    tmp = pd.DataFrame({"_val": series, **{c: X_train[c] for c in groupby}})
    tmp = tmp.dropna(subset=["_val"])

    grouped = tmp.groupby(groupby)["_val"]
    stats = grouped.agg(["std", "mean", "count"])
    stats = stats[stats["count"] >= 10]

    if len(stats) == 0:
        return {"error": "No groups with >= 10 rows after filtering."}

    with np.errstate(divide="ignore", invalid="ignore"):
        cvs = stats["std"] / stats["mean"].abs()
    cvs = cvs.replace([np.inf, -np.inf], np.nan).dropna()

    if len(cvs) == 0:
        return {"error": "All CVs are undefined (zero-mean groups)."}

    # Sample a few invariant groups for inspection
    low_cv = cvs.nsmallest(5)
    sample_groups = []
    for idx in low_cv.index:
        if isinstance(idx, tuple):
            grp = dict(zip(groupby, idx))
        else:
            grp = {groupby[0]: idx}
        grp["cv"] = float(cvs.loc[idx])
        grp["mean"] = float(stats.loc[idx, "mean"])
        grp["std"] = float(stats.loc[idx, "std"])
        grp["count"] = int(stats.loc[idx, "count"])
        sample_groups.append(grp)

    return {
        "median_cv": float(cvs.median()),
        "mean_cv": float(cvs.mean()),
        "n_groups": int(len(stats)),
        "n_invariant_groups": int((cvs < 0.1).sum()),
        "sample_low_cv_groups": sample_groups,
    }


def _query_cardinality(
    X_train: pd.DataFrame,
    cols: list[str],
) -> dict[str, Any]:
    """Cardinality of individual columns and their combination."""
    result: dict[str, Any] = {}

    for col in cols:
        if col not in X_train.columns:
            result[col] = {"error": f"Column '{col}' not found"}
            continue
        result[col] = {
            "nunique": int(X_train[col].nunique()),
            "dtype": str(X_train[col].dtype),
            "n_missing": int(X_train[col].isna().sum()),
        }

    if len(cols) > 1:
        valid_cols = [c for c in cols if c in X_train.columns]
        if len(valid_cols) > 1:
            combined = X_train.groupby(valid_cols).ngroups
            result["combined"] = {"nunique": int(combined), "columns": valid_cols}

    return result


def _query_correlation(
    X_train: pd.DataFrame,
    col_a: str,
    col_b: str,
) -> dict[str, Any]:
    """Correlation between two columns."""
    for col in [col_a, col_b]:
        if col not in X_train.columns:
            return {"error": f"Column '{col}' not found"}

    a = X_train[col_a]
    b = X_train[col_b]
    valid = a.notna() & b.notna()

    return {
        "pearson": float(a.corr(b, method="pearson")),
        "spearman": float(a.corr(b, method="spearman")),
        "n_valid": int(valid.sum()),
    }


def _query_conditional_distribution(
    X_train: pd.DataFrame,
    col: str,
    groupby: str,
    n_groups: int = 5,
) -> dict[str, Any]:
    """Distribution of a column within the top N groups."""
    for c in [col, groupby]:
        if c not in X_train.columns:
            return {"error": f"Column '{c}' not found"}

    # Top N groups by size
    top_groups = X_train[groupby].value_counts().head(n_groups).index.tolist()

    result: dict[str, Any] = {"groups": []}
    for grp in top_groups:
        mask = X_train[groupby] == grp
        vals = X_train.loc[mask, col].dropna()
        if len(vals) == 0:
            continue

        entry: dict[str, Any] = {
            "group_value": str(grp),
            "count": int(len(vals)),
        }
        if pd.api.types.is_numeric_dtype(vals):
            entry.update({
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "p25": float(vals.quantile(0.25)),
                "median": float(vals.median()),
                "p75": float(vals.quantile(0.75)),
            })
        else:
            entry["top_values"] = vals.value_counts().head(5).to_dict()

        result["groups"].append(entry)

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_query(query_type: str, **kwargs: Any) -> dict[str, Any]:
    """Execute a statistical query on training data.

    Args:
        query_type: One of within_group_variance, cardinality, correlation, conditional_distribution.
        **kwargs: Query-specific arguments.

    Returns:
        Query result dict (also printed to stdout).
    """
    config = load_config()

    # Collect all referenced columns for leakage check
    referenced_cols: list[str] = []
    if "cols" in kwargs and kwargs["cols"]:
        referenced_cols.extend(kwargs["cols"])
    if "col_a" in kwargs and kwargs["col_a"]:
        referenced_cols.append(kwargs["col_a"])
    if "col_b" in kwargs and kwargs["col_b"]:
        referenced_cols.append(kwargs["col_b"])
    if "col" in kwargs and kwargs["col"]:
        referenced_cols.append(kwargs["col"])
    if "groupby" in kwargs and kwargs["groupby"]:
        gb = kwargs["groupby"]
        if isinstance(gb, list):
            referenced_cols.extend(gb)
        else:
            referenced_cols.append(gb)

    _check_leakage(referenced_cols, config.target)
    remaining = _check_rate_limit()

    # Load training data
    import autoresearch_tabular.prepare as prepare
    prepare._initialize()
    folds = prepare.get_folds()
    X_train, _, _, _ = folds[0]

    # Dispatch
    if query_type == "within_group_variance":
        result = _query_within_group_variance(X_train, kwargs["expr"], kwargs["groupby"])
    elif query_type == "cardinality":
        result = _query_cardinality(X_train, kwargs["cols"])
    elif query_type == "correlation":
        result = _query_correlation(X_train, kwargs["col_a"], kwargs["col_b"])
    elif query_type == "conditional_distribution":
        result = _query_conditional_distribution(
            X_train, kwargs["col"], kwargs["groupby"], kwargs.get("n_groups", 5),
        )
    else:
        result = {"error": f"Unknown query type: {query_type}"}

    # Log the query
    _log_query(query_type)
    remaining -= 1

    # Print results
    print(f"Query: {query_type}")
    print(f"Remaining quota: {remaining}/{MAX_QUERIES_PER_SESSION}")
    print()

    if "error" in result:
        print(f"ERROR: {result['error']}")
    else:
        for key, val in result.items():
            if isinstance(val, list):
                print(f"{key}:")
                for item in val:
                    print(f"  {item}")
            elif isinstance(val, dict):
                print(f"{key}:")
                for k, v in val.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {val}")

    return result
