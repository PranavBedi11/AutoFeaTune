"""train.py — Fixed evaluation harness: training and scoring.

THIS FILE IS PART OF THE FIXED HARNESS. THE AGENT NEVER TOUCHES IT.

Responsibilities:
    1. Import features.py and prepare.py
    2. For each fold: call engineer_features, apply guard, fit XGBoost, score
    3. Inf/NaN guard between features.py output and XGBoost input
    4. Compute mean and std of metric across folds
    5. Compute composite score (penalises feature explosion)
    6. Print results including "METRIC: <value>" for subprocess parsing
    7. Record experiment to memory graph (trusted write path)
    8. Append results to results.tsv

Design decisions:
    - Agent NEVER records its own scores — only train.py writes to memory graph
    - Composite score = cv_score - 0.001 * n_features (higher-is-better metrics)
    - XGBoost uses fixed hyperparameters; agent optimises features, not HP
    - Called as subprocess: uv run python -m autoresearch_tabular.train
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import ast

import autoresearch_tabular.prepare as prepare
from autoresearch_tabular.memory_graph import MemoryGraph, load_graph

__all__ = ["run_experiment"]

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_FILE = PROJECT_ROOT / "results.tsv"


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_metric(
    y_true: pd.Series,
    y_pred: np.ndarray,
    metric_name: str,
) -> float:
    """Compute the evaluation metric.

    Args:
        y_true: Ground truth target values.
        y_pred: Model predictions (probabilities for classification).
        metric_name: One of 'rmse', 'mae', 'auc', 'logloss', 'f1'.

    Returns:
        The computed metric value.

    Raises:
        ValueError: If metric_name is not recognised.
    """
    if metric_name == "rmse":
        from sklearn.metrics import root_mean_squared_error
        return float(root_mean_squared_error(y_true, y_pred))
    elif metric_name == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    elif metric_name == "auc":
        if len(np.unique(y_true)) > 2:
            return float(roc_auc_score(y_true, y_pred, multi_class="ovr", average="macro"))
        else:
            return float(roc_auc_score(y_true, y_pred))
    elif metric_name == "logloss":
        return float(log_loss(y_true, y_pred))
    elif metric_name == "f1":
        if len(np.unique(y_true)) > 2:
            return float(f1_score(y_true, y_pred, average="weighted"))
        else:
            return float(f1_score(y_true, y_pred))
    elif metric_name == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric_name!r}")


# ---------------------------------------------------------------------------
# Data safety guard
# ---------------------------------------------------------------------------


def guard_dataframe(
    df: pd.DataFrame,
    fold_idx: int,
    split_name: str,
    label_encoders: dict[str, dict[Any, int]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[Any, int]]]:
    """Apply safety guards to a DataFrame before XGBoost.

    Guards:
        1. Replace ±inf with NaN
        2. Ordinal-encode any remaining string/object/categorical columns
        3. Fill NaN with column median

    Args:
        df: DataFrame to guard.
        fold_idx: Fold index (for logging).
        split_name: 'train' or 'val' (for logging and fitting/applying).
        label_encoders: Fitted encoders from train split (pass on val/test).

    Returns:
        Cleaned DataFrame and updated label_encoders dict.
    """
    df = df.copy()
    label_encoders = label_encoders or {}

    # 1. Replace inf
    numeric_df = df.select_dtypes(include=[np.number])
    inf_cols = numeric_df.columns[np.isinf(numeric_df).any()]
    if len(inf_cols) > 0:
        warnings.warn(
            f"Fold {fold_idx} {split_name}: replacing inf in {len(inf_cols)} cols"
        )
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 2. Ordinal encode non-numeric columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        if split_name == "train":
            unique_vals = df[col].dropna().unique()
            label_encoders[col] = {val: i for i, val in enumerate(unique_vals)}
        mapping = label_encoders.get(col, {})
        df[col] = (
            df[col].map(lambda x: mapping.get(x, -1) if pd.notna(x) else np.nan)
            .astype("float32")
        )

    # 3. Fill NaN with median
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        df[nan_cols] = df[nan_cols].fillna(df[nan_cols].median())

    assert df.select_dtypes(exclude=[np.number]).empty, (
        f"Fold {fold_idx} {split_name}: non-numeric columns remain after guard!"
    )
    return df, label_encoders


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def run_experiment(experiment_description: str = "") -> dict[str, Any]:
    """Run a full experiment: feature engineering + XGBoost + scoring.

    Steps:
        1. Load folds from prepare.get_folds()
        2. Import features.engineer_features (fresh from disk each call)
        3. Validate engineer_features returns 3 DataFrames
        4. For each fold: engineer → guard → fit XGBoost → predict → score
        5. Compute mean/std across folds
        6. Compute composite score with feature count penalty
        7. Record experiment to memory graph (trusted write path)
        8. Append to results.tsv
        9. Print "METRIC: <value>" for subprocess parsing

    Args:
        experiment_description: Human-readable description of the attempt.

    Returns:
        Dict with: experiment_id, cv_score, cv_std, delta, n_features,
                   composite_score, kept, metric, description, run_time.
    """
    sep = "=" * 50
    print(f"\n{sep}\nRunning experiment: {experiment_description}\n{sep}")

    folds = prepare.get_folds()
    config = prepare.load_config()
    metric_name = config.metric
    is_higher_better = config.is_higher_better

    # Import features module fresh (subprocess → no reload needed)
    features_module = _import_features_module()

    num_folds = len(folds)
    scores: list[float] = []
    n_features = 0

    # Get test set for engineering passthrough (X_test is passed to engineer_features;
    # test_y is never used — agent does not score the test set)
    X_test, _ = prepare.get_test_set()

    # Encode categorical target if needed
    folds = list(folds)
    if folds:
        first_y = folds[0][2]
        if (
            first_y.dtype == "object"
            or first_y.dtype.name in {"category", "string", "str"}
            or first_y.dtype == "bool"
            or not pd.api.types.is_numeric_dtype(first_y)
        ):
            unique_labels = np.unique(first_y.dropna())
            unique_labels.sort()
            target_map = {val: i for i, val in enumerate(unique_labels)}
            for i in range(len(folds)):
                folds[i] = (
                    folds[i][0],
                    folds[i][1],
                    folds[i][2].map(target_map).astype(int),
                    folds[i][3].map(target_map).astype(int),
                )

    feature_columns: list[str] = []
    fold_shap_arrays: list[dict[str, float]] = []   # one dict per fold
    start_time = time.time()
    for fold_idx, (X_train, X_val, y_train, y_val) in enumerate(folds):
        print(f"  Fold {fold_idx + 1}/{num_folds}...")

        # Engineer features — y_train passed for target-aware encoding (no leakage)
        result = features_module.engineer_features(
            X_train.copy(), X_val.copy(), X_test.copy(), y_train=y_train
        )

        # Validate return type
        if not (isinstance(result, tuple) and len(result) == 3):
            raise ValueError(
                "engineer_features must return a tuple of 3 DataFrames, "
                f"got: {type(result)}"
            )
        X_train_eng, X_val_eng, _ = result
        if not (isinstance(X_train_eng, pd.DataFrame) and isinstance(X_val_eng, pd.DataFrame)):
            raise ValueError(
                "engineer_features must return DataFrames, "
                f"got types: {type(X_train_eng)}, {type(X_val_eng)}"
            )

        n_features = X_train_eng.shape[1]
        if not feature_columns:
            feature_columns = list(X_train_eng.columns)

        # Guard
        X_train_clean, encoders = guard_dataframe(X_train_eng, fold_idx, "train")
        X_val_clean, _ = guard_dataframe(X_val_eng, fold_idx, "val", label_encoders=encoders)

        # Fit XGBoost
        model = _fit_xgboost(X_train_clean, y_train, metric_name)

        # Predict
        if metric_name in {"auc", "logloss"} and hasattr(model, "predict_proba"):
            preds = model.predict_proba(X_val_clean)
            if preds.ndim == 2 and preds.shape[1] == 2:
                preds = preds[:, 1]
        else:
            preds = model.predict(X_val_clean)

        # Score
        score = compute_metric(y_val, preds, metric_name)
        scores.append(score)
        print(f"    {metric_name.upper()}: {score:.4f}")

        # SHAP — use XGBoost native pred_contribs (no extra dependency)
        fold_shap = _compute_fold_shap(model, X_val_clean)
        if fold_shap:
            fold_shap_arrays.append(fold_shap)

    run_time = time.time() - start_time

    cv_score = float(np.mean(scores))
    cv_std = float(np.std(scores))

    # Aggregate SHAP across folds: mean and std of per-fold mean-abs values
    feature_shap, feature_shap_std = _aggregate_fold_shap(fold_shap_arrays)

    # Composite score: penalise feature explosion
    feature_penalty = 0.001 * n_features
    composite_score = (
        cv_score - feature_penalty if is_higher_better else cv_score + feature_penalty
    )

    print(f"\nCV {metric_name.upper()}: {cv_score:.4f} ± {cv_std:.4f} (n_features={n_features})")
    print(f"Composite Score: {composite_score:.4f}")

    # Compare to best
    mg = load_graph()
    best_exp = mg.get_best_experiment(is_higher_better=is_higher_better)
    delta = 0.0
    kept = True

    if best_exp:
        best_comp = float(best_exp.get("composite_score", 0.0))
        if is_higher_better:
            delta = composite_score - best_comp
        else:
            delta = best_comp - composite_score
        threshold = config.min_delta * abs(best_comp)
        kept = delta > threshold
        print(f"Delta: {delta:+.4f} (threshold={threshold:.4f} = {config.min_delta:.1%} of best, kept={kept})")
    else:
        print("Delta: baseline (kept=True)")

    results: dict[str, Any] = {
        "cv_score": cv_score,
        "cv_std": cv_std,
        "delta": delta,
        "n_features": n_features,
        "composite_score": composite_score,
        "kept": kept,
        "description": experiment_description,
        "run_time": run_time,
        "metric": metric_name,
    }

    exp_id = mg.record_experiment(
        cv_score=cv_score,
        cv_std=cv_std,
        delta=delta,
        n_features=n_features,
        composite_score=composite_score,
        kept=kept,
        description=experiment_description,
        features_used=feature_columns,
        feature_shap=feature_shap,
        feature_shap_std=feature_shap_std,
    )
    results["experiment_id"] = exp_id

    _register_features(exp_id, feature_columns)
    _register_feature_set(exp_id, feature_columns)
    _register_correlations(feature_columns, X_train_eng)
    if kept:
        _register_outperforms(mg, exp_id, feature_columns, is_higher_better)
        mg.update_active_feature_statuses(feature_columns)
    _append_to_results_tsv(results)

    # Print parseable metric line for the agent to read
    print(f"METRIC: {composite_score:.6f}")

    return results


def _register_features(exp_id: int, feature_columns: list[str]) -> None:
    """Parse features.py with AST and register Feature + DERIVED_FROM nodes.

    Finds all DataFrame column assignments (df['col'] = ...) and the source
    columns referenced on the right-hand side. Runs inside the trusted harness
    so the graph is always up to date after each experiment.

    Args:
        exp_id: Experiment ID to attach features to.
        feature_columns: Actual column names present after engineering (from X_train_eng).
    """
    # NOTE: Project package is `autoresearch_tabular` (with underscore).
    # This path is used only for AST parsing to populate the memory graph.
    features_path = PROJECT_ROOT / "src" / "autoresearch_tabular" / "features.py"
    if not features_path.exists():
        return

    try:
        source = features_path.read_text()
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return

    valid_vars = {"df", "out", "result", "train", "val", "test", "X_train", "X_val", "X_test", "X", "data"}
    assigned: dict[str, list[str]] = {}

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            # Track simple dataflow within function scopes:
            # name -> set of source column names.
            self._scopes: list[dict[str, set[str]]] = [{}]

        def _scope(self) -> dict[str, set[str]]:
            return self._scopes[-1]

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._scopes.append({})
            self.generic_visit(node)
            self._scopes.pop()

        def visit_Assign(self, node: ast.Assign) -> None:
            targets: list[str] = []
            for t in node.targets:
                if (
                    isinstance(t, ast.Subscript)
                    and isinstance(t.value, ast.Name)
                    and t.value.id in valid_vars
                    and isinstance(t.slice, ast.Constant)
                ):
                    targets.append(str(t.slice.value))
            if targets:
                sources: set[str] = set()
                for child in ast.walk(node.value):
                    if (
                        isinstance(child, ast.Subscript)
                        and isinstance(child.value, ast.Name)
                        and child.value.id in valid_vars
                        and isinstance(child.slice, ast.Constant)
                    ):
                        sources.add(str(child.slice.value))
                    elif isinstance(child, ast.Name) and child.id in self._scope():
                        sources.update(self._scope()[child.id])
                for tgt in targets:
                    assigned[tgt] = list(sources)
            else:
                # Record intermediate variables so we can attribute sources through
                # simple chains like:
                #   clipped = df["AveOccup"].clip(...)
                #   out["log_AveOccup"] = np.log1p(clipped)
                rhs_sources: set[str] = set()
                for child in ast.walk(node.value):
                    if (
                        isinstance(child, ast.Subscript)
                        and isinstance(child.value, ast.Name)
                        and child.value.id in valid_vars
                        and isinstance(child.slice, ast.Constant)
                    ):
                        rhs_sources.add(str(child.slice.value))
                    elif isinstance(child, ast.Name) and child.id in self._scope():
                        rhs_sources.update(self._scope()[child.id])
                if rhs_sources:
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            self._scope()[t.id] = set(rhs_sources)
            self.generic_visit(node)

    _Visitor().visit(tree)

    # Only register features actually present in the engineered output
    feature_set = set(feature_columns)
    mg = load_graph()
    for name, sources in assigned.items():
        if name in feature_set:
            try:
                mg.register_feature(name, sources, exp_id, save=False)
            except Exception as e:
                logger.debug("Feature registration skipped for %s: %s", name, e)
    if assigned:
        mg.save()


def _register_feature_set(exp_id: int, feature_columns: list[str]) -> None:
    """Register Feature → Experiment edges for an experiment."""
    mg = load_graph()
    try:
        mg.register_feature_set(exp_id, feature_columns)
    except Exception:
        # Non-fatal: graph utility should not break training runs.
        return


def _register_correlations(
    feature_columns: list[str], X_train_eng: pd.DataFrame
) -> None:
    """Register CORRELATED_WITH edges for highly correlated feature pairs.

    Computes pairwise Pearson correlation on the last fold's engineered
    training data. Pairs with |r| > 0.8 get a CORRELATED_WITH edge.
    Non-fatal: graph errors never break training.
    """
    if len(feature_columns) < 2:
        return
    try:
        mg = load_graph()
        # Ensure the relationship type is registered.
        mg.register_relationship_type(
            rel_type="CORRELATED_WITH",
            description="Features with Pearson |r| > 0.8",
            source_type="Feature",
            target_type=["Feature"],
            category="statistical",
        )
        # Only correlate numeric columns present in feature_columns.
        numeric = X_train_eng[feature_columns].select_dtypes(include="number")
        if numeric.shape[1] < 2:
            return
        corr = numeric.corr()
        for i, col_a in enumerate(corr.columns):
            for col_b in corr.columns[i + 1 :]:
                r = corr.loc[col_a, col_b]
                if abs(r) > 0.8:
                    mg.add_edge_typed(
                        f"feat_{col_a}",
                        f"feat_{col_b}",
                        "CORRELATED_WITH",
                        save=False,
                        correlation=round(float(r), 4),
                    )
        mg.save()
    except Exception:
        return


def _register_outperforms(
    mg: MemoryGraph,
    exp_id: int,
    feature_columns: list[str],
    is_higher_better: bool,
) -> None:
    """Register OUTPERFORMS edges when a kept experiment beats prior kept ones.

    Unlike IMPROVED_OVER (purely chronological), OUTPERFORMS captures that
    specific overlapping features drove the improvement.
    Non-fatal: graph errors never break training.
    """
    try:
        mg.register_relationship_type(
            rel_type="OUTPERFORMS",
            description="Kept experiment with overlapping features scored better",
            source_type="Experiment",
            target_type=["Experiment"],
            category="improvement",
        )
        current_features = set(feature_columns)
        current_node = f"exp_{exp_id}"
        current_score = mg.graph.nodes.get(current_node, {}).get("composite_score", 0)

        for e in mg._nodes_of_type("Experiment"):
            if not e.get("kept") or e.get("exp_id") == exp_id:
                continue
            other_features = set(e.get("features_used", []))
            overlap = current_features & other_features
            if not overlap:
                continue
            other_score = e.get("composite_score", 0)
            other_node = f"exp_{e['exp_id']}"
            if is_higher_better and current_score > other_score:
                mg.add_edge_typed(
                    current_node, other_node, "OUTPERFORMS",
                    save=False, overlap=len(overlap),
                )
            elif not is_higher_better and current_score < other_score:
                mg.add_edge_typed(
                    current_node, other_node, "OUTPERFORMS",
                    save=False, overlap=len(overlap),
                )
        mg.save()
    except Exception:
        return


def _compute_fold_shap(model: Any, X_val_clean: pd.DataFrame) -> dict[str, float]:
    """Compute mean absolute SHAP values for one fold using XGBoost native contribs.

    Uses ``predict(pred_contribs=True)`` on the booster — no external ``shap``
    package required. Handles regression, binary classification, and multiclass.

    Returns an empty dict on any failure so a SHAP error never breaks training.

    Args:
        model: Fitted XGBRegressor or XGBClassifier.
        X_val_clean: Cleaned validation DataFrame for this fold.

    Returns:
        ``{feature_name: mean_abs_shap}`` — one entry per column in X_val_clean.
    """
    try:
        dmat = xgb.DMatrix(X_val_clean)
        contribs = model.get_booster().predict(dmat, pred_contribs=True)
        n_feat = X_val_clean.shape[1]

        if contribs.ndim == 2 and contribs.shape[1] == n_feat + 1:
            # Regression or binary classification: (n_samples, n_features + 1)
            mean_abs = np.abs(contribs[:, :n_feat]).mean(axis=0)
        elif contribs.ndim == 3 and contribs.shape[2] == n_feat + 1:
            # Multiclass natively 3D in newer XGBoost: (n_samples, n_classes, n_features + 1)
            mean_abs = np.abs(contribs[:, :, :n_feat]).mean(axis=(0, 1))
        elif contribs.ndim == 2 and contribs.shape[1] > n_feat + 1:
            # Multiclass older XGBoost: (n_samples, n_classes * (n_features + 1))
            n_classes = contribs.shape[1] // (n_feat + 1)
            reshaped = contribs.reshape(-1, n_classes, n_feat + 1)
            mean_abs = np.abs(reshaped[:, :, :n_feat]).mean(axis=(0, 1))
        else:
            return {}

        return {
            col: float(mean_abs[i])
            for i, col in enumerate(X_val_clean.columns)
        }
    except Exception as exc:
        logger.debug("SHAP computation skipped for this fold: %s", exc)
        return {}


def _aggregate_fold_shap(
    fold_shap_arrays: list[dict[str, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Aggregate per-fold SHAP dicts into mean and std across folds.

    Features not present in a fold's dict (e.g. fold computation failed) are
    treated as 0 for that fold so the mean is not inflated by missing data.

    Args:
        fold_shap_arrays: List of ``{feature: mean_abs_shap}`` dicts, one per fold.

    Returns:
        ``(feature_shap, feature_shap_std)`` — both dicts keyed by feature name.
        Returns two empty dicts if no fold data is available.
    """
    if not fold_shap_arrays:
        return {}, {}

    all_features: set[str] = set()
    for d in fold_shap_arrays:
        all_features.update(d.keys())

    feature_shap: dict[str, float] = {}
    feature_shap_std: dict[str, float] = {}
    for feat in sorted(all_features):
        vals = [d.get(feat, 0.0) for d in fold_shap_arrays]
        feature_shap[feat] = float(np.mean(vals))
        feature_shap_std[feat] = float(np.std(vals))

    return feature_shap, feature_shap_std


def _import_features_module() -> Any:
    """Import (or reimport) autoresearch_tabular.features from disk."""
    import autoresearch_tabular.features as features_module
    return importlib.reload(features_module)


def _fit_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    metric_name: str,
) -> Any:
    """Fit an XGBoost model with fixed hyperparameters.

    -------------------------------------------------------------------------
    WANT TO USE YOUR OWN MODEL? Replace the body of this function.

    You MUST ensure:
      1. Same signature: takes (X_train: DataFrame, y_train: Series,
         metric_name: str) and returns a fitted model object.
      2. predict() works: model.predict(X_val) must return a 1-D array of
         predictions for regression (rmse, mae) and binary classification (f1).
      3. predict_proba() works for probability metrics: if metric_name is
         'auc' or 'logloss', the harness calls model.predict_proba(X_val)
         and expects shape (n_samples, n_classes). If your model doesn't have
         predict_proba, change metric_name in config.yaml to 'f1' instead.
      4. Fit on X_train/y_train only — never touch X_val or X_test here.
      5. Input is a clean numeric DataFrame (no NaNs, no strings) — the
         guard step above this call already handles that.
      6. Keep it deterministic: set a fixed random seed so experiments are
         comparable across runs.
    -------------------------------------------------------------------------

    Args:
        X_train: Training features.
        y_train: Training target.
        metric_name: Determines regression vs. classification objective.

    Returns:
        Fitted model.
    """
    params: dict[str, Any] = {
        "random_state": 42,
        "n_jobs": 1,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "enable_categorical": False,
    }

    if metric_name in {"rmse", "mae"}:
        return xgb.XGBRegressor(**params).fit(X_train, y_train)

    if len(np.unique(y_train)) > 2:
        params["objective"] = "multi:softprob"
    else:
        params["objective"] = "binary:logistic"
    return xgb.XGBClassifier(**params).fit(X_train, y_train)


def _get_git_commit() -> str:
    """Return the short (7-char) git commit hash, or 'unknown' if not in a repo."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _append_to_results_tsv(results: dict[str, Any]) -> None:
    """Append experiment results to results.tsv (creates file with headers if needed).

    Args:
        results: Experiment results dict.
    """
    file_exists = RESULTS_FILE.exists()
    kept = results["kept"]
    row = {
        "commit": _get_git_commit(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_id": results.get("experiment_id", "?"),
        "status": "keep" if kept else "discard",
        "metric": results["metric"],
        "cv_score": f"{results['cv_score']:.4f}",
        "cv_std": f"{results['cv_std']:.4f}",
        "delta": f"{results['delta']:+.4f}",
        "n_features": results["n_features"],
        "composite_score": f"{results['composite_score']:.4f}",
        "run_time": f"{results['run_time']:.1f}s",
        "description": results.get("description", ""),
    }
    df_row = pd.DataFrame([row])
    df_row.to_csv(
        RESULTS_FILE, mode="a", header=not file_exists, index=False, sep="\t"
    )


# ---------------------------------------------------------------------------
# CLI entry point (uv run python -m autoresearch_tabular.train)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single feature engineering experiment.")
    parser.add_argument("--description", default="manual run", help="Experiment description")
    args = parser.parse_args()

    result = run_experiment(experiment_description=args.description)
    print(f"\nResult: experiment_id={result['experiment_id']}, kept={result['kept']}")