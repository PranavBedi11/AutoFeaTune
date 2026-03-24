"""prepare.py — Fixed evaluation harness: data loading and splitting.

THIS FILE IS PART OF THE FIXED HARNESS. THE AGENT NEVER TOUCHES IT.

Responsibilities:
    1. Load and validate config.yaml via AgentConfig
    2. Load CSV dataset
    3. Drop exclude_columns and validate target
    4. Auto-detect or use explicit categorical/numeric columns
    5. Create train/val/test splits (temporal or random stratified)
    6. Populate Column nodes in memory graph on first run
    7. Export get_folds() and get_test_set()

Design:
    - Temporal splits when date_col is set (train on past, val on future)
    - Random stratified splits when date_col is null
    - Hold-out test set is NEVER scored by the agent
    - Module-level state is initialized once on first get_folds() call
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import json
from hashlib import sha256
from sklearn.model_selection import StratifiedKFold, train_test_split

from autoresearch_tabular.config import AgentConfig, load_config
from autoresearch_tabular.memory_graph import load_graph

__all__ = [
    "get_folds",
    "get_test_set",
    "load_config",
    "FoldData",
    "SplitResult",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
FoldData = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
SplitResult = tuple[list[FoldData], pd.DataFrame, pd.Series]

PROJECT_ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(config: AgentConfig) -> pd.DataFrame:
    """Load the CSV dataset and apply initial preprocessing.

    Steps:
        1. Read CSV from config.data_path
        2. Drop columns in config.exclude_columns
        3. Validate config.target exists

    Args:
        config: Validated AgentConfig.

    Returns:
        Raw DataFrame with excluded columns removed.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the target column is not present.
    """
    data_path = Path(config.data_path)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    # Drop excluded columns (silently skip missing ones)
    if config.exclude_columns:
        to_drop = [c for c in config.exclude_columns if c in df.columns]
        df = df.drop(columns=to_drop)

    # Validate target
    if config.target not in df.columns:
        raise ValueError(
            f"Target column '{config.target}' not found in dataset. "
            f"Available: {list(df.columns)}"
        )

    # Validate date_col if set
    if config.date_col and config.date_col not in df.columns:
        raise ValueError(
            f"date_col '{config.date_col}' not found in dataset. "
            f"Available: {list(df.columns)}"
        )

    # Validate categorical_columns if explicitly specified
    if config.categorical_columns:
        missing = [c for c in config.categorical_columns if c not in df.columns]
        if missing:
            logger.warning(
                "categorical_columns specified but not found in dataset: %s", missing
            )

    return df


def _dataset_signature(config: AgentConfig, data_path: Path, df: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    """Compute a dataset+config signature used to decide whether to reset memory graph."""
    try:
        st = data_path.stat()
        file_info = {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
    except Exception:
        file_info = {"size": None, "mtime_ns": None}

    meta: dict[str, Any] = {
        "data_path": str(data_path),
        "target": config.target,
        "metric": config.metric,
        "date_col": config.date_col,
        "exclude_columns": list(config.exclude_columns or []),
        "categorical_columns": list(config.categorical_columns or []),
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "file": file_info,
    }
    blob = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    sig = sha256(blob).hexdigest()
    return sig, meta


def detect_column_types(
    df: pd.DataFrame,
    target: str,
    categorical_columns: list[str],
) -> tuple[list[str], list[str]]:
    """Detect which columns are categorical and which are numeric.

    If categorical_columns is provided, uses that list directly.
    Otherwise auto-detects based on dtype and cardinality heuristics:
        - object/category/bool dtype → categorical
        - int/float with cardinality < 20 → categorical
        - everything else → numeric

    Args:
        df: DataFrame without the target column.
        target: Name of target column (excluded from detection).
        categorical_columns: Explicit list from config; empty for auto-detect.

    Returns:
        Tuple of (categorical_col_names, numeric_col_names).
    """
    feature_cols = [c for c in df.columns if c != target]

    if categorical_columns:
        cat_cols = [c for c in categorical_columns if c in feature_cols]
        num_cols = [c for c in feature_cols if c not in cat_cols]
        return cat_cols, num_cols

    cat_cols: list[str] = []
    num_cols: list[str] = []
    for col in feature_cols:
        dtype = df[col].dtype
        if dtype == "object" or dtype.name == "category" or dtype == "bool":
            cat_cols.append(col)
        elif pd.api.types.is_numeric_dtype(dtype):
            if df[col].nunique() < 20:
                cat_cols.append(col)
            else:
                num_cols.append(col)
        else:
            cat_cols.append(col)

    return cat_cols, num_cols


# ---------------------------------------------------------------------------
# Splitting strategies
# ---------------------------------------------------------------------------


def create_temporal_splits(
    df: pd.DataFrame,
    target: str,
    date_col: str,
    n_folds: int,
    random_seed: int,
) -> SplitResult:
    """Create temporal train/val splits using an ordered date column.

    Strategy: expanding window — each fold uses all data up to the
    validation boundary as training, the next chunk as validation.
    Last 20% of data (chronologically) is held out as the test set.

    Args:
        df: Full DataFrame including target.
        target: Name of target column.
        date_col: Name of the date/datetime column.
        n_folds: Number of folds.
        random_seed: Unused (present for API consistency).

    Returns:
        Tuple of (folds, X_test, y_test).
    """
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    n = len(df_sorted)
    test_start = int(n * 0.8)
    df_train_pool = df_sorted.iloc[:test_start].copy()
    df_test = df_sorted.iloc[test_start:].copy()

    feature_cols = [c for c in df.columns if c != target]
    X_test = df_test[feature_cols]
    y_test = df_test[target]

    n_train_pool = len(df_train_pool)
    chunk_size = n_train_pool // (n_folds + 1)

    folds: list[FoldData] = []
    for i in range(n_folds):
        train_end = chunk_size * (i + 1)
        val_start = train_end
        val_end = min(train_end + chunk_size, n_train_pool)
        if val_end <= val_start:
            break
        fold_train = df_train_pool.iloc[:train_end]
        fold_val = df_train_pool.iloc[val_start:val_end]
        folds.append(
            (
                fold_train[feature_cols],
                fold_val[feature_cols],
                fold_train[target],
                fold_val[target],
            )
        )

    return folds, X_test, y_test


def create_random_splits(
    df: pd.DataFrame,
    target: str,
    n_folds: int,
    random_seed: int,
) -> SplitResult:
    """Create random stratified train/val splits.

    Holds out 20% as test set, then creates n_folds stratified folds
    on the remaining 80%. Regression targets are binned into quantiles
    for stratification.

    Args:
        df: Full DataFrame including target.
        target: Name of target column.
        n_folds: Number of cross-validation folds.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (folds, X_test, y_test).
    """
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols]
    y = df[target]

    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        strat_col = pd.qcut(y, q=min(10, y.nunique()), labels=False, duplicates="drop")
    else:
        strat_col = y

    X_train_pool, X_test, y_train_pool, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=strat_col
    )

    if pd.api.types.is_numeric_dtype(y_train_pool) and y_train_pool.nunique() > 20:
        strat_train = pd.qcut(
            y_train_pool,
            q=min(10, y_train_pool.nunique()),
            labels=False,
            duplicates="drop",
        )
    else:
        strat_train = y_train_pool

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    folds: list[FoldData] = []
    for train_idx, val_idx in skf.split(X_train_pool, strat_train):
        folds.append(
            (
                X_train_pool.iloc[train_idx],
                X_train_pool.iloc[val_idx],
                y_train_pool.iloc[train_idx],
                y_train_pool.iloc[val_idx],
            )
        )

    return folds, X_test, y_test


# ---------------------------------------------------------------------------
# Module-level state (lazy initialization)
# ---------------------------------------------------------------------------

_folds: list[FoldData] | None = None
_X_test: pd.DataFrame | None = None
_y_test: pd.Series | None = None
_config: AgentConfig | None = None


def _initialize() -> None:
    """Load config, create splits, and populate memory graph. Called once."""
    global _folds, _X_test, _y_test, _config

    if _folds is not None:
        return

    logger.info("Initializing data pipeline ...")
    print("Initializing data pipeline ...")

    _config = load_config()
    logger.info("Config loaded: target='%s', metric='%s'", _config.target, _config.metric)
    print(f"   Config: target='{_config.target}', metric='{_config.metric}'")

    df = load_dataset(_config)
    logger.info("Dataset loaded: %d rows × %d columns", df.shape[0], df.shape[1])
    print(f"   Dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    target = _config.target
    cat_cols, num_cols = detect_column_types(
        df, target, _config.categorical_columns
    )
    print(f"   Column types: {len(cat_cols)} categorical, {len(num_cols)} numeric")

    date_col = _config.date_col
    if date_col and date_col in df.columns:
        print(f"   Using temporal splits on '{date_col}'")
        folds, X_test, y_test = create_temporal_splits(
            df, target, date_col, _config.n_folds, _config.random_seed
        )
    else:
        print("   Using random stratified splits")
        folds, X_test, y_test = create_random_splits(
            df, target, _config.n_folds, _config.random_seed
        )

    _folds = folds
    _X_test = X_test
    _y_test = y_test
    print(f"   Created {len(folds)} folds, test set: {len(X_test)} rows")

    # Populate Column nodes in memory graph
    mg = load_graph()
    data_path = Path(_config.data_path)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    sig, meta = _dataset_signature(_config, data_path, df)
    reset = mg.ensure_dataset_signature(sig, meta=meta, backup_on_change=True)
    if reset:
        logger.info("Dataset signature changed; reset memory graph.")
    feature_df = df[[c for c in df.columns if c != target]]
    mg.populate_source_columns(feature_df)

    print("   Data pipeline initialized.\n")


def get_folds() -> list[FoldData]:
    """Return the list of (X_train, X_val, y_train, y_val) fold tuples.

    Initializes data on first call.

    Returns:
        List of fold tuples.
    """
    _initialize()
    assert _folds is not None
    return _folds


def get_test_set() -> tuple[pd.DataFrame, pd.Series]:
    """Return the held-out test set.

    The agent passes X_test through engineer_features() but never scores it.
    This function is called by train.py only for the engineering passthrough.

    Returns:
        Tuple of (X_test, y_test).
    """
    _initialize()
    assert _X_test is not None and _y_test is not None
    return _X_test, _y_test


# ---------------------------------------------------------------------------
# CLI entry point (python -m autoresearch_tabular.prepare)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _initialize()
    print("Data pipeline ready.")
    cfg = _config
    assert cfg is not None
    folds_n = len(_folds) if _folds is not None else 0
    test_rows = len(_X_test) if _X_test is not None else 0
    print(f"Folds: {folds_n}, test rows: {test_rows}")
