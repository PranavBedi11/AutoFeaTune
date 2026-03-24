"""cli.py — Entry point for autoresearch-tabular.

Subcommands:
    autoresearch demo             — zero-config demo on California Housing
    autoresearch setup <csv>      — set up a new project from your own CSV
    autoresearch init             — initialize project (git, memory graph, baseline commit)
    autoresearch prepare          — load data and populate Column nodes in memory graph
    autoresearch discover         — run data discovery (entity keys, invariant expressions)
    autoresearch query <type>     — safe statistical query tool with leakage protection

Usage:
    uv run autoresearch demo
    uv run autoresearch setup data/myfile.csv --target Price --metric rmse
    uv run autoresearch init
    uv run autoresearch prepare
    uv run autoresearch discover
    uv run autoresearch query cardinality --cols card1 addr1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

__all__ = ["main"]

PROJECT_ROOT = Path(__file__).parent.parent.parent
FEATURES_FILE = PROJECT_ROOT / "src" / "autoresearch_tabular" / "features.py"
FEATURES_GIT_PATH = "src/autoresearch_tabular/features.py"


def cmd_init(args: argparse.Namespace) -> None:
    """Initialize the project: memory graph + git + baseline commit."""
    print("Initializing autoresearch-tabular project ...")

    from autoresearch_tabular.memory_graph import load_graph

    graph_path = PROJECT_ROOT / "db" / "memory_graph.json"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    mg = load_graph(graph_path)
    mg.save()
    print(f"   Memory graph initialized: {graph_path}")

    git_dir = PROJECT_ROOT / ".git"
    if not git_dir.exists():
        subprocess.run(["git", "init"], cwd=PROJECT_ROOT, check=True, capture_output=True)
        print("   Created git repository.")
    else:
        print("   Git repository already exists.")

    gitignore = PROJECT_ROOT / ".gitignore"
    _write_gitignore(gitignore)
    print("   .gitignore updated.")

    if FEATURES_FILE.exists():
        subprocess.run(
            ["git", "add", FEATURES_GIT_PATH, ".gitignore"],
            cwd=PROJECT_ROOT, check=True, capture_output=True,
        )
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=PROJECT_ROOT, capture_output=True,
        )
        if result.returncode != 0:
            subprocess.run(
                ["git", "commit", "-m", "baseline: initial features.py (passthrough)"],
                cwd=PROJECT_ROOT, check=True, capture_output=True,
            )
            print("   Committed baseline features.py.")
        else:
            print("   Baseline already committed.")

    print("\nSetup complete. Next steps:")
    print("  1. Place your CSV in data/")
    print("  2. Edit config.yaml")
    print("  3. Edit program.md with domain knowledge")
    print("  4. Run: uv run autoresearch prepare")
    print("  5. Open AGENTS.md and start the AI agent loop")


def cmd_demo(args: argparse.Namespace) -> None:
    """Zero-config demo: download California Housing, write all files, init+prepare."""
    import pandas as pd
    from sklearn.datasets import fetch_california_housing

    print("autoresearch demo — California Housing")
    print("=" * 42)

    # 1. Download dataset
    print("\n[1/4] Downloading California Housing dataset from sklearn ...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame  # includes target column MedHouseVal
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / "california_housing.csv"
    df.to_csv(csv_path, index=False)
    print(f"      Saved {len(df):,} rows × {len(df.columns)} columns → {csv_path.relative_to(PROJECT_ROOT)}")

    # 2. Write config.yaml
    print("\n[2/4] Writing config.yaml ...")
    config_text = """\
data_path: data/california_housing.csv
target: MedHouseVal
metric: rmse
date_col: null
exclude_columns: []
categorical_columns: []
n_folds: 5
random_seed: 42
time_budget_minutes: 60
min_delta: 0.001
"""
    (PROJECT_ROOT / "config.yaml").write_text(config_text)
    print("      config.yaml written.")

    # 3. Write program.md
    print("\n[3/4] Writing program.md ...")
    _write_california_housing_program_md(PROJECT_ROOT / "program.md")
    print("      program.md written.")

    # 4. Init + prepare
    print("\n[4/4] Running init + prepare ...")
    cmd_init(args)
    cmd_prepare(args)

    print("\n" + "=" * 42)
    print("READY. California Housing demo is set up.")
    print()
    print("Next: open AGENTS.md and start the AI agent loop.")
    print("  uv run autoresearch status   — check loop state")
    print("  uv run python visualize.py   — interactive graph viewer")


def cmd_setup(args: argparse.Namespace) -> None:
    """Set up a new project from a user-provided CSV."""
    import pandas as pd

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"autoresearch setup — {csv_path.name}")
    print("=" * 42)

    # Load just the header + a few rows for inspection
    df = pd.read_csv(csv_path, nrows=200)
    cols = list(df.columns)
    print(f"\nDataset: {len(cols)} columns")
    for i, c in enumerate(cols):
        print(f"  [{i}] {c}  ({df[c].dtype})")

    # Resolve target
    target = args.target
    if not target:
        raw = input("\nWhich column do you want to predict? ").strip()
        if raw.isdigit():
            target = cols[int(raw)]
        else:
            target = raw
    if target not in cols:
        print(f"Error: column '{target}' not found in CSV.", file=sys.stderr)
        sys.exit(1)
    print(f"Target: {target}")

    # Resolve metric
    metric = args.metric
    if not metric:
        n_unique = df[target].nunique()
        if n_unique == 2:
            suggested = "auc"
            task = "binary classification"
        elif n_unique <= 20:
            suggested = "f1"
            task = "multi-class classification"
        else:
            suggested = "rmse"
            task = "regression"
        print(f"Detected task: {task} ({n_unique} unique target values)")
        print(f"Suggested metric: {suggested}  (rmse / mae / auc / f1 / logloss)")
        raw = input(f"Metric [{suggested}]: ").strip()
        metric = raw if raw else suggested
    print(f"Metric: {metric}")

    # Write config.yaml
    print("\nWriting config.yaml ...")
    config_text = f"""\
data_path: {csv_path}
target: {target}
metric: {metric}
date_col: null
exclude_columns: []
categorical_columns: []
n_folds: 5
random_seed: 42
time_budget_minutes: 60
min_delta: 0.001
"""
    (PROJECT_ROOT / "config.yaml").write_text(config_text)
    print("  config.yaml written.")

    # Write skeleton program.md
    program_md_path = PROJECT_ROOT / "program.md"
    if program_md_path.exists():
        overwrite = input("\nprogram.md already exists. Overwrite with skeleton? [y/N] ").strip().lower()
        if overwrite != "y":
            print("  Keeping existing program.md.")
        else:
            _write_skeleton_program_md(program_md_path, csv_path, target, metric, df)
            print("  program.md skeleton written — add your domain knowledge.")
    else:
        _write_skeleton_program_md(program_md_path, csv_path, target, metric, df)
        print("  program.md skeleton written — add your domain knowledge.")

    # Init + prepare
    print("\nRunning init + prepare ...")
    cmd_init(args)
    cmd_prepare(args)

    print("\n" + "=" * 42)
    print("READY.")
    print()
    print("Edit program.md to add domain knowledge, then start the AI agent loop.")
    print("  uv run autoresearch status   — check loop state")
    print("  uv run python visualize.py   — interactive graph viewer")


def _write_california_housing_program_md(path: Path) -> None:
    path.write_text("""\
# Program Brief — California Housing Price Prediction

## What are you predicting?

Median house value (`MedHouseVal`) for California census block groups (~1990 census).
- **Units:** $100,000s — so a value of 2.5 means $250,000
- **Task:** Regression
- **Range:** 0.15 – 5.0 (capped at 5.0 in the dataset)

## Dataset description

- **Source:** sklearn `fetch_california_housing` (1990 US Census)
- **Rows:** 20,640
- **Features:** 8 numeric, no categoricals, no missing values
- **Target:** MedHouseVal — median house value per block group

## Column descriptions

| Column | Description | Notes |
|---|---|---|
| `MedInc` | Median income in block group | Units: tens of thousands of dollars. Strongest single predictor. Right-skewed. |
| `HouseAge` | Median house age in block group | Capped at 52. Weakly predictive on its own. |
| `AveRooms` | Average rooms per household | Can be extremely high for small populations — outliers exist. |
| `AveBedrms` | Average bedrooms per household | Should be < AveRooms always. Ratio AveBedrms/AveRooms is a quality signal. |
| `Population` | Block group population | Very right-skewed. Not directly predictive but useful in ratios. |
| `AveOccup` | Average occupants per household | High values (overcrowding) signal lower-value areas. Has extreme outliers. |
| `Latitude` | Block group latitude | Encodes geography — Bay Area / coastal areas are higher value. |
| `Longitude` | Block group longitude | Encodes geography — inland/desert areas are lower value. |

## Domain knowledge

- **MedInc is dominant.** It explains the largest share of variance.
- **Geography matters enormously.** Lat/Lon alone don't express geography well as raw floats — binning or KMeans clustering unlocks major signal.
- **Overcrowding is a strong negative signal.** Clip `AveOccup` at 99th percentile before any transform.
- **Rooms-per-bedroom is a quality proxy.** `AveRooms / AveBedrms` signals luxury vs cramped housing.
- **The 5.0 cap on MedHouseVal.** ~4% of rows are capped — features that push predictions above 5.0 don't help.

## Suggested transformations

- `log_MedInc` — log1p transform on MedInc (reduces right skew)
- `log_AveOccup` — clip at 99th pct then log1p
- `geo_cell` — quantile bin Latitude (6 bins) × Longitude (6 bins) into a 36-cell grid
- `geo_medinc` — mean MedInc per geo_cell (fit on train)
- `relative_income` — MedInc / geo_medinc
- KMeans clustering on (Latitude, Longitude)

## Things to avoid

- Do not use raw Latitude/Longitude without any geographic grouping
- Do not use AveOccup raw without clipping extreme outliers
- Do not create more than ~20 features (composite score penalty: +0.001 × n_features)
- Do not leak target: all geo aggregations must be fit on X_train only
""")


def _write_skeleton_program_md(
    path: Path,
    csv_path: Path,
    target: str,
    metric: str,
    df: "pd.DataFrame",  # type: ignore[name-defined]
) -> None:
    col_rows = "\n".join(
        f"| `{c}` | {df[c].dtype} | <!-- describe this column --> |"
        for c in df.columns
        if c != target
    )
    path.write_text(f"""\
# Program Brief — {csv_path.stem}

## What are you predicting?

- **Target column:** `{target}`
- **Metric:** {metric}
- **Task:** <!-- regression / binary classification / multi-class -->

## Dataset description

- **File:** {csv_path}
- **Rows:** <!-- fill in -->
- **Features:** {len(df.columns) - 1} columns

## Column descriptions

| Column | Type | Notes |
|---|---|---|
{col_rows}

## Domain knowledge

<!-- Add what you know about this domain, key predictors, known patterns -->

## Suggested transformations

<!-- Add ideas for feature engineering -->

## Things to avoid

<!-- Add known pitfalls, leakage risks, useless transforms -->
""")


def cmd_status(args: argparse.Namespace) -> None:
    """Print loop status: STOP/EXPLORE/CONTINUE + totals."""
    import yaml
    from autoresearch_tabular.memory_graph import load_graph
    from autoresearch_tabular.inspect_graph import (
        should_stop,
        get_diminishing_returns_signal,
        consecutive_failures,
    )

    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
    mg = load_graph()
    G = mg.graph

    stop, reason = should_stop(
        G,
        time_budget_minutes=cfg.get("time_budget_minutes", 480),
        max_consecutive_failures=cfg.get("max_consecutive_failures", 999),
    )
    print(reason)

    exps = mg.get_experiment_history(n=9999)
    total = len(exps)
    print(f"TOTAL_EXPERIMENTS: {total}")

    consecutive = consecutive_failures(G)

    stagnation = get_diminishing_returns_signal(G, min_delta=cfg.get("min_delta", 0.001))
    if not stop and (consecutive >= 10 or (total > 0 and total % 15 == 0) or stagnation > 0.7):
        print(f"EXPLORE: forced exploration triggered (consecutive={consecutive}, total={total}, stagnation={stagnation:.2f})")


def cmd_inspect(args: argparse.Namespace) -> None:
    """Delegate to inspect_graph, forwarding all flags."""
    import sys
    from autoresearch_tabular import inspect_graph

    # Reconstruct argv so inspect_graph.main() sees the flags
    flags = []
    if args.exp:
        flags += ["--exp", str(args.exp)]
    if args.col:
        flags += ["--col", args.col]
    if args.central:
        flags.append("--central")
    if args.ablation:
        flags.append("--ablation")
    if args.longest_path:
        flags.append("--longest-path")
    if args.saturated:
        flags.append("--saturated")
    if args.rates:
        flags.append("--rates")
    if args.load_bearing:
        flags.append("--load-bearing")
    if args.untried:
        flags.append("--untried")
    if args.shap:
        flags.append("--shap")
    if args.hypotheses:
        flags.append("--hypotheses")
    if args.correlations:
        flags.append("--correlations")
    if args.edges:
        flags.append("--edges")
    if args.failed:
        flags.append("--failed")
    if args.context:
        flags.append("--context")
    if args.coverage:
        flags.append("--coverage")
    if args.discovery:
        flags.append("--discovery")

    old_argv = sys.argv
    sys.argv = ["inspect_graph"] + flags
    try:
        inspect_graph.main()
    finally:
        sys.argv = old_argv


def cmd_prepare(args: argparse.Namespace) -> None:
    """Load data, create splits, and populate Column nodes in memory graph."""
    print("Preparing data pipeline ...")
    import autoresearch_tabular.prepare as prepare
    prepare._initialize()
    print("Data pipeline ready.")


def cmd_discover(args: argparse.Namespace) -> None:
    """Run data discovery pipeline."""
    from autoresearch_tabular.discover import run_discovery
    run_discovery()


def cmd_query(args: argparse.Namespace) -> None:
    """Run a statistical query on training data."""
    from autoresearch_tabular.query import run_query
    kwargs = vars(args).copy()
    query_type = kwargs.pop("query_type")
    kwargs.pop("command", None)
    run_query(query_type, **kwargs)


def _write_gitignore(gitignore: Path) -> None:
    content = """\
# Database and generated files
db/
assets/
run.log

# Dataset files (user-provided, often large)
data/

# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/
*.egg-info/
.pytest_cache/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
    gitignore.write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="autoresearch",
        description="Feature engineering agent for tabular ML.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("demo", help="Zero-config demo on California Housing (downloads data automatically)")

    setup_p = subparsers.add_parser("setup", help="Set up a new project from your own CSV")
    setup_p.add_argument("csv", help="Path to your CSV dataset")
    setup_p.add_argument("--target", default="", help="Target column name (prompted if omitted)")
    setup_p.add_argument("--metric", default="", help="Metric: rmse | mae | auc | f1 | logloss (auto-detected if omitted)")

    subparsers.add_parser("init", help="Initialize project (git, memory graph, baseline commit)")
    subparsers.add_parser("prepare", help="Load data and populate Column nodes")
    subparsers.add_parser("discover", help="Run data discovery: entity keys, invariant expressions, residual analysis")
    subparsers.add_parser("status", help="Print loop status (STOP/EXPLORE/CONTINUE)")

    # Query subcommand with nested sub-subparsers
    query_p = subparsers.add_parser("query", help="Safe statistical query tool with leakage protection")
    query_sub = query_p.add_subparsers(dest="query_type", required=True)

    wgv_p = query_sub.add_parser("within_group_variance", help="Check if expression is constant within groups")
    wgv_p.add_argument("--expr", required=True, help="Expression to evaluate (e.g., 'TransactionDT/86400 - D1')")
    wgv_p.add_argument("--groupby", nargs="+", required=True, help="Columns to group by")

    card_p = query_sub.add_parser("cardinality", help="Cardinality of columns or column combinations")
    card_p.add_argument("--cols", nargs="+", required=True, help="Column names")

    corr_p = query_sub.add_parser("correlation", help="Correlation between two columns")
    corr_p.add_argument("--col_a", required=True, help="First column")
    corr_p.add_argument("--col_b", required=True, help="Second column")

    cond_p = query_sub.add_parser("conditional_distribution", help="Distribution of column within groups")
    cond_p.add_argument("--col", required=True, help="Column to analyze")
    cond_p.add_argument("--groupby", required=True, help="Column to group by")
    cond_p.add_argument("--n_groups", type=int, default=5, help="Number of top groups to show")

    inspect_p = subparsers.add_parser("inspect", help="Query the memory graph")
    inspect_p.add_argument("--exp", type=int, default=None, help="Deep-dive on experiment ID")
    inspect_p.add_argument("--col", type=str, default=None, help="Deep-dive on column name")
    inspect_p.add_argument("--central", action="store_true", help="Feature centrality ranking")
    inspect_p.add_argument("--ablation", action="store_true", help="Ablation signal report")
    inspect_p.add_argument("--longest-path", action="store_true", dest="longest_path", help="Longest improvement path")
    inspect_p.add_argument("--saturated", action="store_true", help="Columns with exhausted signal")
    inspect_p.add_argument("--rates", action="store_true", help="Keep rate per transform family")
    inspect_p.add_argument("--load-bearing", action="store_true", dest="load_bearing", help="Features in every kept experiment")
    inspect_p.add_argument("--untried", action="store_true", help="Untried (column, transform_family) pairs")
    inspect_p.add_argument("--shap", action="store_true", help="SHAP importance: current best + cross-experiment consensus")
    inspect_p.add_argument("--hypotheses", action="store_true", help="Active hypotheses grouped by SUPPORTS/CONTRADICTS")
    inspect_p.add_argument("--correlations", action="store_true", help="Highly correlated feature pairs (Pearson |r| > 0.8)")
    inspect_p.add_argument("--edges", action="store_true", help="List all registered relationship types with edge counts")
    inspect_p.add_argument("--failed",   action="store_true", help="Reverted experiment descriptions (do not repeat)")
    inspect_p.add_argument("--context",  action="store_true", help="Compressed full history (MemGPT-style, for long runs)")
    inspect_p.add_argument("--coverage", action="store_true", help="Columns tried vs untouched, features active vs total")
    inspect_p.add_argument("--discovery", action="store_true", help="Discovery results: entity keys, invariant expressions")

    args = parser.parse_args()

    if args.command == "demo":
        cmd_demo(args)
    elif args.command == "setup":
        cmd_setup(args)
    elif args.command == "init":
        cmd_init(args)
    elif args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "discover":
        cmd_discover(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "inspect":
        cmd_inspect(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
