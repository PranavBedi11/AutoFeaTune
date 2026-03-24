"""inspect_graph — Terminal graph traversal + analytics for the autoresearch memory graph.

Usage:
    uv run autoresearch inspect                    # full report
    uv run autoresearch inspect --exp 9            # deep-dive on one experiment
    uv run autoresearch inspect --col <name>       # all paths through a column
    uv run autoresearch inspect --central          # feature centrality ranking
    uv run autoresearch inspect --ablation         # ablation: which features matter most
    uv run autoresearch inspect --longest-path     # longest improvement chain
    uv run autoresearch inspect --saturated        # columns with exhausted signal
    uv run autoresearch inspect --rates            # keep rate per transform family
    uv run autoresearch inspect --load-bearing     # features present in every kept experiment
    uv run autoresearch inspect --untried          # (column, transform_family) pairs not yet tried
    uv run autoresearch inspect --hypotheses       # active hypotheses grouped by SUPPORTS/CONTRADICTS
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any

import networkx as nx

from autoresearch_tabular.memory_graph import MemoryGraph, load_graph

D = "─" * 60

# ---------------------------------------------------------------------------
# Shared helpers for standalone analytics functions
# ---------------------------------------------------------------------------

# Keyword list for classifying experiment descriptions into transform families.
# First match wins — more specific keywords first.
FAMILY_KEYWORDS: list[tuple[str, str]] = [
    ("target_enc", "target_encoding"),
    ("onehot", "onehot"),
    ("frequency", "frequency_encoding"),
    ("ordinal", "ordinal"),
    ("kmeans", "clustering"),
    ("cluster", "clustering"),
    ("quantile", "binning"),
    ("bin", "binning"),
    ("log", "log"),
    ("sqrt", "sqrt"),
    ("clip", "clip"),
    ("power", "power"),
    ("aggregate", "geo_aggregate"),
    ("_mean", "geo_aggregate"),
    ("_std", "geo_aggregate"),
    ("ratio", "ratio"),
    ("product", "product"),
    ("interaction", "interaction"),
    ("difference", "difference"),
    ("drop", "feature_drop"),
    ("add", "feature_add"),
]


def _nodes_of_type(G: nx.DiGraph, node_type: str) -> list[dict[str, Any]]:
    """Return attribute dicts for all nodes with the given node_type."""
    return [
        d
        for _, d in G.nodes(data=True)
        if d.get("node_type") == node_type
    ]


def _classify_family(description: str) -> str:
    """Classify an experiment description into a transform family."""
    op_text = description.lower()
    for part in description.split(";"):
        stripped = part.strip()
        if stripped.lower().startswith("op="):
            op_text = stripped[3:].lower()
            break
    for keyword, fam in FAMILY_KEYWORDS:
        if keyword in op_text:
            return fam
    return "other"


# ---------------------------------------------------------------------------
# Standalone analytics functions (extracted from MemoryGraph)
# ---------------------------------------------------------------------------


def get_saturated_columns(
    G: nx.DiGraph,
    min_experiments: int = 3,
    delta_threshold: float = 0.001,
) -> list[dict[str, Any]]:
    """Return columns whose marginal gain has flattened across experiments."""
    col_deltas: dict[str, list[float]] = {}

    for col_data in _nodes_of_type(G, "Column"):
        col_name = col_data["name"]
        col_id = f"col_{col_name}"

        derived: set[str] = set()
        for feat_data in _nodes_of_type(G, "Feature"):
            feat_id = f"feat_{feat_data['name']}"
            if G.has_edge(feat_id, col_id):
                derived.add(feat_data["name"])

        if not derived:
            continue

        col_deltas[col_name] = []
        for exp_data in _nodes_of_type(G, "Experiment"):
            if set(exp_data.get("features_used", [])) & derived:
                col_deltas[col_name].append(abs(exp_data.get("delta", 0.0)))

    results = []
    for col_name, deltas in col_deltas.items():
        if len(deltas) < min_experiments:
            continue
        mean_d = sum(deltas) / len(deltas)
        if mean_d < delta_threshold:
            results.append(
                {
                    "column": col_name,
                    "n_experiments": len(deltas),
                    "mean_delta": round(mean_d, 6),
                    "max_delta": round(max(deltas), 6),
                }
            )

    return sorted(results, key=lambda x: x["mean_delta"])


def get_transform_success_rates(G: nx.DiGraph) -> dict[str, dict[str, Any]]:
    """Return keep rate broken down by transform family."""
    counts: dict[str, dict[str, Any]] = {}

    for exp_data in _nodes_of_type(G, "Experiment"):
        desc = exp_data.get("description", "")
        kept = exp_data.get("kept", False)
        family = _classify_family(desc)

        if family not in counts:
            counts[family] = {"total": 0, "kept": 0, "rate": 0.0}
        counts[family]["total"] += 1
        if kept:
            counts[family]["kept"] += 1

    for fam, c in counts.items():
        t = c["total"]
        c["rate"] = round(c["kept"] / t, 3) if t > 0 else 0.0

    return dict(sorted(counts.items(), key=lambda kv: kv[1]["rate"], reverse=True))


def get_load_bearing_features(G: nx.DiGraph) -> list[str]:
    """Return features present in every kept experiment's feature set."""
    kept_exps = [
        e for e in _nodes_of_type(G, "Experiment")
        if e.get("kept", False) and e.get("features_used")
    ]
    if not kept_exps:
        return []

    load_bearing: set[str] = set(kept_exps[0]["features_used"])
    for exp in kept_exps[1:]:
        load_bearing &= set(exp["features_used"])

    return sorted(load_bearing)


def get_untried_column_transform_pairs(G: nx.DiGraph) -> list[dict[str, Any]]:
    """Return (column, transform_family) pairs not yet explored."""
    all_families = {fam for _, fam in FAMILY_KEYWORDS}

    tried: set[tuple[str, str]] = set()
    for exp_data in _nodes_of_type(G, "Experiment"):
        desc = exp_data.get("description", "")

        cols_in_desc: list[str] = []
        for part in desc.split(";"):
            stripped = part.strip()
            if stripped.lower().startswith("col="):
                raw = stripped[4:].strip()
                cols_in_desc = [c.strip() for c in raw.split(",") if c.strip()]
                break

        family = _classify_family(desc)

        for col in cols_in_desc:
            tried.add((col, family))

    all_cols = {d["name"] for d in _nodes_of_type(G, "Column")}

    results = []
    for col in sorted(all_cols):
        for fam in sorted(all_families):
            if (col, fam) not in tried:
                results.append({"column": col, "transform_family": fam})

    return results


def get_shap_ranking(
    G: nx.DiGraph,
    experiment_id: int | None = None,
    is_higher_better: bool = True,
) -> list[dict[str, Any]]:
    """Return features sorted by mean-abs-SHAP for one experiment."""
    if experiment_id is not None:
        exp = G.nodes.get(f"exp_{experiment_id}", {})
    else:
        kept = [d for d in _nodes_of_type(G, "Experiment") if d.get("kept", False)]
        if not kept:
            return []
        if is_higher_better:
            best = max(kept, key=lambda x: x.get("composite_score", 0.0))
        else:
            best = min(kept, key=lambda x: x.get("composite_score", float("inf")))
        if best and best.get("feature_shap"):
            exp = best
        else:
            all_exps = sorted(
                _nodes_of_type(G, "Experiment"),
                key=lambda x: x.get("exp_id", 0),
                reverse=True,
            )
            exp = next((e for e in all_exps if e.get("feature_shap")), {})

    shap = exp.get("feature_shap", {})
    shap_std = exp.get("feature_shap_std", {})
    if not shap:
        return []

    return sorted(
        [
            {
                "feature": f,
                "mean_shap": round(float(v), 6),
                "shap_std": round(float(shap_std.get(f, 0.0)), 6),
            }
            for f, v in shap.items()
        ],
        key=lambda x: x["mean_shap"],
        reverse=True,
    )


def get_shap_consensus(G: nx.DiGraph) -> list[dict[str, Any]]:
    """Return features ranked by consistent importance across kept experiments."""
    kept_with_shap = [
        e for e in _nodes_of_type(G, "Experiment")
        if e.get("kept") and e.get("feature_shap")
    ]
    if not kept_with_shap:
        return []

    all_features: set[str] = set()
    for e in kept_with_shap:
        all_features.update(e["feature_shap"].keys())

    exp_data: list[tuple[dict[str, int], dict[str, float], int]] = []
    for e in kept_with_shap:
        shap = e["feature_shap"]
        n = len(all_features)
        ranked = sorted(shap.keys(), key=lambda f: shap[f], reverse=True)
        rank_map = {f: i + 1 for i, f in enumerate(ranked)}
        exp_data.append((rank_map, shap, n))

    n_exps = len(exp_data)
    results = []
    for feat in sorted(all_features):
        ranks = [r.get(feat, n + 1) for r, _, n in exp_data]
        shap_vals = [s.get(feat, 0.0) for _, s, _ in exp_data]
        results.append(
            {
                "feature": feat,
                "mean_rank": round(sum(ranks) / n_exps, 2),
                "mean_shap": round(sum(shap_vals) / n_exps, 6),
                "times_in_top3": sum(1 for r in ranks if r <= 3),
                "n_experiments": n_exps,
            }
        )

    return sorted(results, key=lambda x: x["mean_rank"])


def get_diminishing_returns_signal(
    G: nx.DiGraph,
    min_delta: float = 0.001,
    is_higher_better: bool = True,
) -> float:
    """Return a 0–1 score indicating how much returns are diminishing."""
    exps = _nodes_of_type(G, "Experiment")
    exps.sort(key=lambda x: x.get("exp_id", 0), reverse=True)
    exps = exps[:10]

    real_exps = [
        e for e in exps
        if not (e.get("delta", 0.0) == 0.0 and e.get("kept", False))
    ]

    if len(real_exps) < 5:
        return 0.0

    recent_deltas = [abs(e.get("delta", 0.0)) for e in real_exps[:5]]
    mean_delta = sum(recent_deltas) / len(recent_deltas)

    if mean_delta == 0.0:
        return 1.0

    kept = [d for d in _nodes_of_type(G, "Experiment") if d.get("kept", False)]
    if kept:
        if is_higher_better:
            best_exp = max(kept, key=lambda x: x.get("composite_score", 0.0))
        else:
            best_exp = min(kept, key=lambda x: x.get("composite_score", float("inf")))
        best_comp = abs(float(best_exp.get("composite_score", 1.0)))
    else:
        best_comp = 1.0

    scale = (35 / 3) * min_delta * best_comp

    return min(1.0, 1.0 - mean_delta / (mean_delta + scale))


def get_best_features_for_column(G: nx.DiGraph, column_name: str) -> list[dict[str, Any]]:
    """Return features derived from a column that appeared in kept experiments."""
    col_id = f"col_{column_name}"
    if not G.has_node(col_id):
        return []

    results = []
    for feat_data in _nodes_of_type(G, "Feature"):
        feat_id = f"feat_{feat_data['name']}"
        if G.has_edge(feat_id, col_id):
            results.append(feat_data)
    return results


# ---------------------------------------------------------------------------
# should_stop — split into focused helpers (Change #4)
# ---------------------------------------------------------------------------

def _session_elapsed_minutes(G: nx.DiGraph) -> tuple[float, str, str]:
    """Return (elapsed_minutes, elapsed_str, remaining_str_placeholder).

    Detects session gaps (>60 min) and resets the timer for new sessions.
    """
    SESSION_GAP_MINUTES = 60
    exps = sorted(
        _nodes_of_type(G, "Experiment"),
        key=lambda x: x.get("exp_id", 0),
        reverse=True,
    )
    if not exps:
        return 0.0, "0m", ""

    last_ts = datetime.fromisoformat(exps[0]["timestamp"])
    gap = (datetime.now() - last_ts).total_seconds() / 60
    if gap > SESSION_GAP_MINUTES:
        return 0.0, "0m (new session)", ""

    # Walk backwards to find session start
    session_exps = sorted(exps, key=lambda x: x.get("exp_id", 0))
    session_start_ts = datetime.fromisoformat(session_exps[-1]["timestamp"])
    for i in range(len(session_exps) - 2, -1, -1):
        t_curr = datetime.fromisoformat(session_exps[i]["timestamp"])
        t_next = datetime.fromisoformat(session_exps[i + 1]["timestamp"])
        if (t_next - t_curr).total_seconds() / 60 > SESSION_GAP_MINUTES:
            break
        session_start_ts = t_curr
    elapsed = (datetime.now() - session_start_ts).total_seconds() / 60
    return elapsed, f"{elapsed:.0f}m", ""


def consecutive_failures(G: nx.DiGraph) -> int:
    """Return count of consecutive kept=False experiments from most recent."""
    exps = sorted(
        _nodes_of_type(G, "Experiment"),
        key=lambda x: x.get("exp_id", 0),
        reverse=True,
    )
    consecutive = 0
    for e in exps:
        if not e.get("kept", False):
            consecutive += 1
        else:
            break
    return consecutive


def should_stop(
    G: nx.DiGraph,
    time_budget_minutes: int = 480,
    max_consecutive_failures: int = 5,
) -> tuple[bool, str]:
    """Return (stop, reason) so the AI agent gets an unambiguous signal."""
    total = len(_nodes_of_type(G, "Experiment"))

    elapsed, elapsed_str, _ = _session_elapsed_minutes(G)
    remaining = max(time_budget_minutes - elapsed, 0)
    remaining_str = f"{remaining:.0f}m"

    if elapsed_str == "0m (new session)":
        remaining_str = f"{time_budget_minutes}m"
    elif total > 0 and elapsed >= time_budget_minutes:
        return True, (
            f"STOP: time budget reached ({elapsed:.0f}/{time_budget_minutes} minutes)."
        )

    consecutive = consecutive_failures(G)
    if consecutive >= max_consecutive_failures:
        return True, (
            f"STOP: {consecutive} consecutive experiments with kept=False "
            f"(max allowed={max_consecutive_failures})."
        )

    return False, (
        f"CONTINUE: {total} experiments, {elapsed_str}/{time_budget_minutes}m elapsed, "
        f"{remaining_str} remaining, {consecutive} consecutive failures."
    )


def get_compressed_history(G: nx.DiGraph, mg: MemoryGraph, recent_n: int = 5) -> str:
    """Return a graph-traversal-based history string that scales to any run length."""
    exps = sorted(
        _nodes_of_type(G, "Experiment"),
        key=lambda x: x.get("exp_id", 0),
        reverse=True,
    )
    total = len(exps)
    recent = exps[:recent_n]
    older = exps[recent_n:]

    lines: list[str] = []

    lines.append(f"── RECENT {len(recent)} EXPERIMENTS (graph detail) ──")
    for e in recent:
        exp_id = e.get("exp_id", 0)
        exp_node = f"exp_{exp_id}"
        s = "KEPT" if e.get("kept") else "REV "

        lines.append(
            f"  [{s}] exp_{exp_id:>3}  "
            f"score={e.get('composite_score', 0):.4f}  "
            f"Δ={e.get('delta', 0):+.4f}  "
            f"feat={e.get('n_features', 0):>3}  "
            f"{str(e.get('description', ''))[:60]}"
        )

        improvement_types = mg.get_rel_types_for_category("improvement")
        improved_over = [
            dst for _, dst, d in G.out_edges(exp_node, data=True)
            if d.get("rel") in improvement_types
        ]
        if improved_over:
            prev = improved_over[0]
            prev_data = G.nodes.get(prev, {})
            lines.append(
                f"    ↳ improved over exp_{prev_data.get('exp_id', '?')}  "
                f"(was {prev_data.get('composite_score', 0):.4f})"
            )

        features_used: list[str] = e.get("features_used", [])
        if features_used:
            lineage_parts: list[str] = []
            lineage_types = mg.get_rel_types_for_category("lineage")
            for feat_name in features_used[:8]:
                feat_node = f"feat_{feat_name}"
                if not G.has_node(feat_node):
                    continue
                sources = [
                    G.nodes[dst].get("name", dst)
                    for _, dst, d in G.out_edges(feat_node, data=True)
                    if d.get("rel") in lineage_types
                    and G.nodes.get(dst, {}).get("node_type") == "Column"
                ]
                if sources:
                    lineage_parts.append(f"{feat_name}←[{','.join(sources)}]")
                else:
                    lineage_parts.append(feat_name)
            if lineage_parts:
                lines.append(f"    features: {', '.join(lineage_parts)}")

    if older:
        kept_older = [e for e in older if e.get("kept")]
        rev_older = [e for e in older if not e.get("kept")]
        try:
            from autoresearch_tabular.prepare import load_config
            _ihb = load_config().is_higher_better
        except Exception:
            _ihb = True
        best_older = (
            (max if _ihb else min)(kept_older, key=lambda x: x.get("composite_score", 0.0))
            if kept_older else None
        )

        lines.append(
            f"\n── OLDER HISTORY (exp 1–{total - len(recent)}, compressed) ──"
        )
        lines.append(f"  {len(kept_older)} kept / {len(rev_older)} reverted")

        if best_older:
            lines.append(
                f"  Best: exp_{best_older.get('exp_id')}  "
                f"score={best_older.get('composite_score', 0):.4f}  "
                f"{str(best_older.get('description', ''))[:60]}"
            )

        improvement_spine: list[str] = []
        for e in reversed(kept_older):
            improvement_spine.append(
                f"exp_{e.get('exp_id')}({e.get('composite_score', 0):.4f})"
            )
        if improvement_spine:
            lines.append(f"  Improvement spine: {' → '.join(improvement_spine)}")

        old_fails = list({
            e.get("description", "")[:60]
            for e in rev_older
            if e.get("description")
        })
        if old_fails:
            lines.append("  Reverted (older, deduplicated):")
            for p in old_fails[:6]:
                lines.append(f"    – {p}")

    try:
        from autoresearch_tabular.prepare import load_config
        _is_higher_better = load_config().is_higher_better
    except Exception:
        _is_higher_better = True
    best = mg.get_best_experiment(is_higher_better=_is_higher_better)
    best_score = best.get("composite_score", 0) if best else 0
    lines.append(f"\n── TOTALS: {total} experiments  best={best_score:.4f} ──")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report display functions
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{D}\n  {title}\n{D}")


def _exp_line(d: dict) -> str:
    s = "KEPT" if d.get("kept") else "REV "
    return (
        f"[{s}] exp_{d.get('exp_id','?'):<3}  "
        f"score={d.get('composite_score',0):.4f}  "
        f"cv={d.get('cv_score',0):.4f}±{d.get('cv_std',0):.4f}  "
        f"feat={d.get('n_features',0):>3}  "
        f"{str(d.get('description',''))[:65]}"
    )


# ---------------------------------------------------------------------------
# Graph repair — backfill missing exp_id attrs + IMPROVED_OVER edges
# ---------------------------------------------------------------------------

def repair_graph(mg) -> bool:
    G = mg.graph
    changed = False

    for node_id, data in G.nodes(data=True):
        if data.get("node_type") == "Experiment" and "exp_id" not in data:
            try:
                G.nodes[node_id]["exp_id"] = int(node_id.split("_", 1)[1])
                changed = True
            except (IndexError, ValueError):
                pass

    kept = sorted(
        [d for _, d in G.nodes(data=True) if d.get("node_type") == "Experiment" and d.get("kept")],
        key=lambda x: x.get("exp_id", 0),
    )
    for i in range(1, len(kept)):
        curr, prev = f"exp_{kept[i]['exp_id']}", f"exp_{kept[i-1]['exp_id']}"
        if G.has_node(curr) and G.has_node(prev) and not G.has_edge(curr, prev):
            G.add_edge(curr, prev, rel="IMPROVED_OVER")
            changed = True

    # Backfill USED_IN edges from features_used attributes
    exp_nodes = [
        (node_id, dict(data))
        for node_id, data in G.nodes(data=True)
        if data.get("node_type") == "Experiment" and data.get("features_used")
    ]
    for node_id, data in exp_nodes:
        for feat_name in data["features_used"]:
            feat_node = f"feat_{feat_name}"
            if not G.has_node(feat_node):
                G.add_node(
                    feat_node,
                    node_type="Feature",
                    name=feat_name,
                    status="active",
                    created_at_experiment=data.get("exp_id", 0),
                )
                changed = True
            if not G.has_edge(feat_node, node_id):
                G.add_edge(feat_node, node_id, rel="USED_IN")
                changed = True

    if changed:
        mg.save()
    return changed


# ---------------------------------------------------------------------------
# Full report sections
# ---------------------------------------------------------------------------

def report_timeline(G: nx.DiGraph) -> None:
    section("EXPERIMENT TIMELINE")
    exps = sorted(
        [d for _, d in G.nodes(data=True) if d.get("node_type") == "Experiment"],
        key=lambda x: x.get("exp_id", 0),
    )
    for e in exps:
        print(" ", _exp_line(e))


def report_improvement_chain(G: nx.DiGraph, mg: MemoryGraph | None = None) -> None:
    section("IMPROVEMENT CHAIN  (IMPROVED_OVER edge traversal)")
    improvement_types = mg.get_rel_types_for_category("improvement") if mg else {"IMPROVED_OVER"}
    kept = sorted(
        [d for _, d in G.nodes(data=True) if d.get("node_type") == "Experiment" and d.get("kept")],
        key=lambda x: x.get("exp_id", 0),
    )
    if not kept:
        print("  No kept experiments yet.")
        return

    chain = []
    current = f"exp_{kept[-1]['exp_id']}"
    while current:
        chain.append(current)
        nxt = next(
            (dst for _, dst, d in G.out_edges(current, data=True) if d.get("rel") in improvement_types),
            None,
        )
        current = nxt

    print(f"  Chain length: {len(chain)}")
    for i, node in enumerate(chain):
        prefix = "  BEST →" if i == 0 else "       ↑"
        print(f"{prefix} {_exp_line(G.nodes[node])}")


def report_longest_path(G: nx.DiGraph) -> None:
    section("LONGEST PATH IN EXPERIMENT DAG  (nx.dag_longest_path)")
    exp_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "Experiment"]
    sub = G.subgraph(exp_nodes)
    try:
        path = nx.dag_longest_path(sub)
        print(f"  Length: {len(path)} nodes")
        for node in path:
            print(f"    {_exp_line(G.nodes[node])}")
    except nx.NetworkXUnfeasible:
        print("  Graph has cycles — cannot compute longest path.")


def report_column_coverage(G: nx.DiGraph, mg: MemoryGraph | None = None) -> None:
    section("COLUMN COVERAGE")
    lineage_types = mg.get_rel_types_for_category("lineage") if mg else {"DERIVED_FROM"}
    cols = {d["name"]: nid for nid, d in G.nodes(data=True) if d.get("node_type") == "Column"}
    covered: dict[str, list[str]] = {name: [] for name in cols}
    for feat_id, feat_data in G.nodes(data=True):
        if feat_data.get("node_type") != "Feature":
            continue
        feat_name = feat_data.get("name")
        if feat_name in covered:
            covered[feat_name].append(feat_name)
        for _, dst, ed in G.out_edges(feat_id, data=True):
            if ed.get("rel") in lineage_types:
                col_name = G.nodes.get(dst, {}).get("name")
                if col_name in covered:
                    covered[col_name].append(feat_data.get("name", feat_id))

    tried = {c: fs for c, fs in covered.items() if fs}
    untried = [c for c in covered if not covered[c]]

    print(f"  Tried ({len(tried)}/{len(cols)}):")
    for col, feats in sorted(tried.items()):
        print(f"    {col:20} → {', '.join(feats)}")
    print(f"\n  Untried ({len(untried)}):")
    for col in sorted(untried):
        d = G.nodes.get(f"col_{col}", {})
        print(f"    {col:20}  mean={d.get('mean',0):.3g}  std={d.get('std',0):.3g}")


def report_feature_lineage(G: nx.DiGraph, mg: MemoryGraph | None = None) -> None:
    section("FEATURE LINEAGE  (nx.ancestors — full upstream tree)")
    lineage_types = mg.get_rel_types_for_category("lineage") if mg else {"DERIVED_FROM"}
    feats = [d for _, d in G.nodes(data=True) if d.get("node_type") == "Feature"]
    if not feats:
        print("  No Feature nodes yet. (Run a new experiment to populate.)")
        return
    for f in sorted(feats, key=lambda x: x.get("name", "")):
        feat_id = f"feat_{f['name']}"
        direct = [
            G.nodes[dst].get("name", dst)
            for _, dst, ed in G.out_edges(feat_id, data=True)
            if ed.get("rel") in lineage_types
        ]
        all_col_ancestors = sorted(
            G.nodes[n].get("name", n)
            for n in nx.descendants(G, feat_id)
            if G.nodes.get(n, {}).get("node_type") == "Column"
        )
        print(f"  {f['name']:30}  direct={direct}  all_cols={all_col_ancestors}")


def report_reachable_from_column(G: nx.DiGraph, mg: MemoryGraph | None = None) -> None:
    section("REACHABLE FROM EACH COLUMN  (nx.single_source_shortest_path)")
    lineage_types = mg.get_rel_types_for_category("lineage") if mg else {"DERIVED_FROM"}
    membership_types = mg.get_rel_types_for_category("membership") if mg else {"USED_IN"}
    cols = [(nid, d) for nid, d in G.nodes(data=True) if d.get("node_type") == "Column"]
    for col_id, col_data in sorted(cols, key=lambda x: x[1].get("name", "")):
        feat_nodes: set[str] = set(
            src
            for src, _, ed in G.in_edges(col_id, data=True)
            if ed.get("rel") in lineage_types and G.nodes.get(src, {}).get("node_type") == "Feature"
        )
        # With direct Feature→Experiment edges, traverse directly
        exp_nodes: set[str] = set()
        for feat in feat_nodes:
            for _, dst, ed in G.out_edges(feat, data=True):
                if ed.get("rel") in membership_types and G.nodes.get(dst, {}).get("node_type") == "Experiment":
                    exp_nodes.add(dst)
        if feat_nodes or exp_nodes:
            print(f"  {col_data['name']:20}  features={len(feat_nodes)}  experiments={len(exp_nodes)}")


def report_centrality(G: nx.DiGraph) -> None:
    section("FEATURE CENTRALITY  (nx.betweenness_centrality — most connected features)")
    feats = [nid for nid, d in G.nodes(data=True) if d.get("node_type") == "Feature"]
    if not feats:
        print("  No Feature nodes yet.")
        return
    centrality = nx.betweenness_centrality(G)
    feat_central = sorted(
        [(nid, centrality[nid]) for nid in feats if nid in centrality],
        key=lambda x: x[1], reverse=True,
    )
    print(f"  {'Feature':<30}  Centrality")
    for nid, score in feat_central:
        name = G.nodes[nid].get("name", nid)
        print(f"  {name:<30}  {score:.4f}")


def report_ablation(G: nx.DiGraph) -> None:
    """Which features appear in kept vs reverted experiments — implicit ablation signal."""
    section("ABLATION SIGNAL  (feature presence in KEPT vs REVERTED experiments)")
    exps = [d for _, d in G.nodes(data=True) if d.get("node_type") == "Experiment"]
    kept_sets = [set(e.get("features_used", [])) for e in exps if e.get("kept")]
    rev_sets = [set(e.get("features_used", [])) for e in exps if not e.get("kept")]

    if not kept_sets:
        print("  No kept experiments yet.")
        return
    if len(kept_sets) < 3 or len(rev_sets) < 3:
        print(f"  Note: low sample size (kept={len(kept_sets)}, reverted={len(rev_sets)}); interpret signals cautiously.")

    all_feats = set().union(*kept_sets, *rev_sets) if (kept_sets or rev_sets) else set()
    rows = []
    for feat in sorted(all_feats):
        in_kept = sum(1 for s in kept_sets if feat in s)
        in_rev = sum(1 for s in rev_sets if feat in s)
        rows.append((feat, in_kept, in_rev))

    rows.sort(key=lambda x: (-x[1], x[2]))
    print(f"  {'Feature':<30}  kept  rev   signal")
    for feat, ik, ir in rows:
        signal = "✓ always kept" if ik > 0 and ir == 0 else ("✗ never kept" if ik == 0 else "~ mixed")
        print(f"  {feat:<30}  {ik:>4}  {ir:>4}   {signal}")


def report_saturated_columns(G: nx.DiGraph) -> None:
    section("SATURATED COLUMNS  (exhausted signal — mean |delta| near zero)")
    results = get_saturated_columns(G)
    if not results:
        print("  No saturated columns detected (need ≥3 experiments per column).")
        return
    print(f"  {'Column':<25}  n_exp  mean_delta  max_delta")
    for r in results:
        print(
            f"  {r['column']:<25}  {r['n_experiments']:>5}  "
            f"{r['mean_delta']:.6f}  {r['max_delta']:.6f}"
        )
    print(f"\n  → Avoid re-engineering these columns. Try a different one.")


def report_transform_rates(G: nx.DiGraph) -> None:
    section("TRANSFORM SUCCESS RATES  (keep rate by transform family)")
    rates = get_transform_success_rates(G)
    if not rates:
        print("  No experiments recorded yet.")
        return
    print(f"  {'Family':<22}  total  kept   rate")
    for family, stats in rates.items():
        bar = "█" * int(stats["rate"] * 10)
        print(
            f"  {family:<22}  {stats['total']:>5}  {stats['kept']:>4}  "
            f"{stats['rate']:.2f}  {bar}"
        )
    best = next(iter(rates))
    worst = list(rates.keys())[-1]
    print(f"\n  → Prefer '{best}' transforms ({rates[best]['rate']:.0%} keep rate).")
    if rates[worst]["rate"] < 0.3 and rates[worst]["total"] >= 2:
        print(f"  → Deprioritise '{worst}' transforms ({rates[worst]['rate']:.0%} keep rate).")


def report_load_bearing(G: nx.DiGraph) -> None:
    section("LOAD-BEARING FEATURES  (present in every kept experiment)")
    features = get_load_bearing_features(G)
    if not features:
        print("  No kept experiments yet, or none recorded features_used.")
        return
    for f in features:
        print(f"  {f}")
    print(f"\n  → Do not prune these without strong evidence. ({len(features)} features)")


def report_untried_pairs(G: nx.DiGraph) -> None:
    section("UNTRIED (column, transform_family) PAIRS")
    pairs = get_untried_column_transform_pairs(G)
    if not pairs:
        print("  All combinations have been tried, or no columns registered.")
        return
    by_col: dict[str, list[str]] = {}
    for p in pairs:
        by_col.setdefault(p["column"], []).append(p["transform_family"])
    for col, families in sorted(by_col.items()):
        print(f"  {col:<25}  {', '.join(sorted(families))}")
    print(f"\n  → {len(pairs)} untried combinations across {len(by_col)} columns.")


def report_shap(G: nx.DiGraph) -> None:
    section("SHAP IMPORTANCE  (mean absolute SHAP, averaged across folds)")

    ranking = get_shap_ranking(G)
    consensus = get_shap_consensus(G)

    if not ranking:
        print("  No SHAP data recorded yet. Run at least one experiment.")
        return

    print("  Current best experiment — features by mean |SHAP|:")
    print(f"  {'Feature':<30}  mean_shap   shap_std   stability")
    for r in ranking:
        if r["mean_shap"] > 0:
            cv = r["shap_std"] / r["mean_shap"]
            stability = "stable" if cv < 0.2 else ("variable" if cv < 0.5 else "unstable")
        else:
            stability = "zero"
        bar = "█" * min(int(r["mean_shap"] * 200), 20)
        print(
            f"  {r['feature']:<30}  {r['mean_shap']:.6f}  "
            f"{r['shap_std']:.6f}   {stability:<8}  {bar}"
        )

    if not consensus:
        return

    print(f"\n  Cross-experiment consensus ({consensus[0]['n_experiments']} kept experiments):")
    print(f"  {'Feature':<30}  mean_rank  mean_shap  top3_count")
    for r in consensus:
        print(
            f"  {r['feature']:<30}  {r['mean_rank']:>9.1f}  "
            f"{r['mean_shap']:.6f}  {r['times_in_top3']:>3}/{r['n_experiments']}"
        )

    zero = [r["feature"] for r in ranking if r["mean_shap"] == 0.0]
    if zero:
        print(f"\n  Zero-SHAP features (not used in any tree split): {zero}")
        print("  → These add feature penalty with no model benefit — consider dropping.")

    unstable = [
        r["feature"] for r in ranking
        if r["mean_shap"] > 0 and (r["shap_std"] / r["mean_shap"]) >= 0.5
    ]
    if unstable:
        print(f"\n  High-variance features (SHAP unstable across folds): {unstable}")
        print("  → Useful in some folds but not others — may be overfitting.")


def report_failed(G: nx.DiGraph) -> None:
    """Standalone wrapper for failed patterns."""
    report_failed_patterns(G)


def report_context(mg: MemoryGraph) -> None:
    section("COMPRESSED HISTORY  (MemGPT-style lossless graph traversal)")
    print(get_compressed_history(mg.graph, mg, recent_n=5))


def report_coverage(mg: MemoryGraph) -> None:
    section("FEATURE COVERAGE  (columns tried vs untouched, features active vs total)")
    diff = mg.get_feature_set_diff()
    print(f"  Features tried (ever): {diff['n_features_tried']}")
    print(f"  Features active now:   {diff['n_features_active']}")
    print(f"\n  Columns tried ({len(diff['columns_tried'])}):")
    for col in diff['columns_tried']:
        print(f"    {col}")
    if diff['columns_untried']:
        print(f"\n  Columns UNTOUCHED ({len(diff['columns_untried'])}) — no feature ever derived:")
        for col in diff['columns_untried']:
            print(f"    {col}")
    else:
        print("\n  All columns have been explored at least once.")


def report_failed_patterns(G: nx.DiGraph) -> None:
    section("FAILED PATTERNS  (reverted — do not repeat)")
    reverted = sorted(
        [d for _, d in G.nodes(data=True) if d.get("node_type") == "Experiment" and not d.get("kept")],
        key=lambda x: x.get("exp_id", 0),
    )
    seen: set[str] = set()
    for e in reverted:
        desc = e.get("description", "")[:80]
        if desc and desc not in seen:
            seen.add(desc)
            print(f"  exp_{e.get('exp_id','?'):<3}  {desc}")


# ---------------------------------------------------------------------------
# Deep-dives
# ---------------------------------------------------------------------------

def report_single_exp(G: nx.DiGraph, exp_id: int, mg: MemoryGraph | None = None) -> None:
    node = f"exp_{exp_id}"
    if not G.has_node(node):
        print(f"  exp_{exp_id} not found.")
        return
    d = G.nodes[node]
    section(f"DEEP DIVE: exp_{exp_id}")
    print(f"  {_exp_line(d)}")

    improvement_types = mg.get_rel_types_for_category("improvement") if mg else {"IMPROVED_OVER"}
    improved_over = [dst for _, dst, ed in G.out_edges(node, data=True) if ed.get("rel") in improvement_types]
    if improved_over:
        print(f"\n  Improved over: {_exp_line(G.nodes[improved_over[0]])}")

    beaten_by = [src for src, _, ed in G.in_edges(node, data=True) if ed.get("rel") in improvement_types]
    if beaten_by:
        print(f"  Later beaten by: {_exp_line(G.nodes[beaten_by[0]])}")

    all_anc = nx.ancestors(G, node)
    col_anc = sorted(G.nodes[n].get("name", n) for n in all_anc if G.nodes.get(n, {}).get("node_type") == "Column")
    feat_anc = sorted(G.nodes[n].get("name", n) for n in all_anc if G.nodes.get(n, {}).get("node_type") == "Feature")

    print(f"\n  features_used ({len(d.get('features_used',[]))}): {d.get('features_used', [])}")
    print(f"  feature nodes in graph: {feat_anc}")
    print(f"  column ancestors (any depth): {col_anc}")

    if col_anc:
        print(f"\n  All paths (exp → feature → column):")
        for col in col_anc:
            col_node = f"col_{col}"
            for path in nx.all_simple_paths(G, source=node, target=col_node, cutoff=5):
                labels = [G.nodes[n].get("name", n) or n for n in path]
                print(f"    {' → '.join(labels)}")

    desc = nx.descendants(G, node)
    later_exps = sorted(
        [G.nodes[n] for n in desc if G.nodes.get(n, {}).get("node_type") == "Experiment"],
        key=lambda x: x.get("exp_id", 0),
    )
    if later_exps:
        print(f"\n  Downstream experiments ({len(later_exps)}):")
        for e in later_exps:
            print(f"    {_exp_line(e)}")


def report_single_col(G: nx.DiGraph, col_name: str, mg: MemoryGraph | None = None) -> None:
    col_node = f"col_{col_name}"
    if not G.has_node(col_node):
        print(f"  Column '{col_name}' not found.")
        return
    d = G.nodes[col_node]
    section(f"COLUMN DEEP DIVE: {col_name}")
    print(f"  dtype={d.get('dtype')}  mean={d.get('mean',0):.4g}  std={d.get('std',0):.4g}  card={d.get('cardinality')}")

    lineage_types = mg.get_rel_types_for_category("lineage") if mg else {"DERIVED_FROM"}
    derived = [(src, G.nodes[src]) for src, _, ed in G.in_edges(col_node, data=True) if ed.get("rel") in lineage_types]
    print(f"\n  Features derived directly ({len(derived)}):")
    for _, fd in derived:
        print(f"    {fd.get('name', '?')}")

    all_desc = nx.descendants(G, col_node)
    exp_desc = sorted(
        [G.nodes[n] for n in all_desc if G.nodes.get(n, {}).get("node_type") == "Experiment"],
        key=lambda x: x.get("exp_id", 0),
    )
    print(f"\n  Experiments that used this column ({len(exp_desc)}):")
    for e in exp_desc:
        print(f"    {_exp_line(e)}")

    print(f"\n  Reachability by hop (nx.single_source_shortest_path, cutoff=4):")
    paths = nx.single_source_shortest_path(G, col_node, cutoff=4)
    by_depth: dict[int, list[str]] = {}
    for target, path in paths.items():
        depth = len(path) - 1
        if depth > 0:
            by_depth.setdefault(depth, []).append(G.nodes[target].get("name", target))
    for depth in sorted(by_depth):
        print(f"    hop {depth}: {sorted(by_depth[depth])}")


# ---------------------------------------------------------------------------
# Hypotheses
# ---------------------------------------------------------------------------

def report_hypotheses(mg) -> None:
    """Print active hypotheses: pending predictions, calibration, and resolved outcomes."""
    section("HYPOTHESES")
    hyps = mg.get_active_hypotheses()
    if not hyps:
        print("  (none — write a hypothesis before your next experiment with mg.add_hypothesis())")
        return

    pending = [h for h in hyps if not h.get("validated", False) and h.get("edge_type") is None]
    supported = [h for h in hyps if h.get("edge_type") == "SUPPORTS"]
    contradicted = [h for h in hyps if h.get("edge_type") == "CONTRADICTS"]

    n_resolved = len(supported) + len(contradicted)
    if n_resolved > 0:
        accuracy = len(supported) / n_resolved
        print(f"\n  Calibration: {len(supported)}/{n_resolved} predictions correct ({accuracy:.0%})")

    if pending:
        print(f"\n  PENDING ({len(pending)}) — awaiting experiment outcome:")
        for h in pending:
            direction = h.get("predicted_direction", "?")
            pred_delta = h.get("predicted_delta")
            delta_str = f"  predicted_delta={pred_delta:+.4f}" if pred_delta is not None else ""
            print(f"    hyp_{h['hyp_id']}  predict={direction}{delta_str}  {h['text']}")

    if supported:
        print(f"\n  SUPPORTS ({len(supported)}) — prediction was correct:")
        for h in supported:
            exp_ref = h.get("created_at_experiment", "?")
            direction = h.get("predicted_direction", "?")
            actual = h.get("actual_delta")
            actual_str = f"  actual={actual:+.4f}" if actual is not None else ""
            print(f"    hyp_{h['hyp_id']}  exp_{exp_ref}  predict={direction}{actual_str}  {h['text']}")

    if contradicted:
        print(f"\n  CONTRADICTS ({len(contradicted)}) — prediction was wrong:")
        for h in contradicted:
            exp_ref = h.get("created_at_experiment", "?")
            direction = h.get("predicted_direction", "?")
            actual = h.get("actual_delta")
            actual_str = f"  actual={actual:+.4f}" if actual is not None else ""
            print(f"    hyp_{h['hyp_id']}  exp_{exp_ref}  predict={direction}{actual_str}  {h['text']}")


# ---------------------------------------------------------------------------
# Relationship registry report
# ---------------------------------------------------------------------------

def report_correlations(mg: MemoryGraph) -> None:
    """Print highly correlated feature pairs (Pearson |r| > 0.8)."""
    section("CORRELATED FEATURES  (|r| > 0.8)")
    edges = mg.get_edges_by_type("CORRELATED_WITH")
    if not edges:
        print("  (none recorded yet)")
        return
    rows = []
    for src, tgt, data in edges:
        a = src.removeprefix("feat_")
        b = tgt.removeprefix("feat_")
        r = data.get("correlation", 0.0)
        rows.append((abs(r), a, b, r))
    rows.sort(reverse=True)
    print(f"  {'Feature A':<30}  {'Feature B':<30}  {'r':>7}")
    for _, a, b, r in rows:
        print(f"  {a:<30}  {b:<30}  {r:+.4f}")


def report_edge_types(mg: MemoryGraph) -> None:
    """List all registered relationship types with edge counts."""
    section("RELATIONSHIP TYPES  (registry)")
    registry = mg.get_relationship_types()
    if not registry:
        print("  No relationship types registered.")
        return
    print(f"  {'Type':<25}  {'Category':<14}  {'Edges':>5}  {'Builtin':>7}  Description")
    for entry in registry:
        count = len(mg.get_edges_by_type(entry["rel_type"]))
        builtin = "yes" if entry.get("builtin") else "no"
        desc = entry.get("description", "")[:45]
        print(
            f"  {entry['rel_type']:<25}  {entry.get('category', ''):<14}  "
            f"{count:>5}  {builtin:>7}  {desc}"
        )


# ---------------------------------------------------------------------------
# Discovery report
# ---------------------------------------------------------------------------

def report_discovery(mg: MemoryGraph) -> None:
    """Report discovery results: entity keys, invariant expressions, residual ICC."""
    section("DISCOVERY RESULTS  (entity keys, invariant expressions)")
    summary = mg.get_discovery_summary()

    if summary["n_entity_keys"] == 0:
        print("  No discovery data. Run: uv run autoresearch discover")
        return

    print(f"  Entity keys: {summary['n_entity_keys']}    "
          f"Derived columns: {summary['n_derived_columns']}    "
          f"Invariant pairs: {summary['n_invariant_pairs']}")

    # Entity keys ranked by residual ICC
    entities = sorted(
        summary["entity_keys"],
        key=lambda e: e.get("residual_icc") or 0,
        reverse=True,
    )
    print(f"\n  {'Entity Key':<40}  {'Cardinality':>12}  {'Residual ICC':>12}")
    for ek in entities:
        cols_str = ", ".join(ek.get("columns", []))
        icc = ek.get("residual_icc")
        icc_str = f"{icc:.4f}" if icc is not None else "—"
        print(f"  {cols_str:<40}  {ek.get('cardinality', 0):>12,}  {icc_str:>12}")

    # Invariant pairs
    invariant_edges = mg.get_edges_by_type("INVARIANT_WITHIN")
    if invariant_edges:
        print(f"\n  Invariant (expression, entity key) pairs:")
        print(f"  {'Expression':<50}  {'Entity Key':<30}  {'Median CV':>10}")
        inv_sorted = sorted(invariant_edges, key=lambda e: e[2].get("median_cv", 1.0))
        for src, tgt, data in inv_sorted[:20]:
            src_data = mg.graph.nodes.get(src, {})
            tgt_data = mg.graph.nodes.get(tgt, {})
            expr = src_data.get("expr", src_data.get("name", src))
            ek_cols = ", ".join(tgt_data.get("columns", []))
            cv = data.get("median_cv", 0)
            print(f"  {expr:<50}  {ek_cols:<30}  {cv:>10.4f}")

        print(f"\n  → Expressions with low CV are near-constant within entity groups.")
        print(f"  → Use these as groupby keys for per-entity aggregation features.")
    else:
        print("\n  No invariant (expression, entity key) pairs found.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, help="Deep-dive on experiment ID")
    parser.add_argument("--col", type=str, help="Deep-dive on column name")
    parser.add_argument("--central", action="store_true", help="Feature centrality ranking")
    parser.add_argument("--ablation", action="store_true", help="Ablation signal report")
    parser.add_argument("--longest-path", action="store_true", help="Longest improvement path")
    parser.add_argument("--saturated", action="store_true", help="Columns with exhausted signal")
    parser.add_argument("--rates", action="store_true", help="Keep rate per transform family")
    parser.add_argument("--load-bearing", action="store_true", help="Features in every kept experiment")
    parser.add_argument("--untried", action="store_true", help="Untried (column, transform_family) pairs")
    parser.add_argument("--shap", action="store_true", help="SHAP importance: current best + cross-experiment consensus")
    parser.add_argument("--hypotheses", action="store_true", help="Active hypotheses grouped by SUPPORTS/CONTRADICTS")
    parser.add_argument("--correlations", action="store_true", help="Highly correlated feature pairs (Pearson |r| > 0.8)")
    parser.add_argument("--edges", action="store_true", help="List all registered relationship types with edge counts")
    parser.add_argument("--failed",   action="store_true", help="Reverted experiment descriptions (do not repeat)")
    parser.add_argument("--context",  action="store_true", help="Compressed full history (MemGPT-style, for long runs)")
    parser.add_argument("--coverage", action="store_true", help="Columns tried vs untouched, features active vs total")
    parser.add_argument("--discovery", action="store_true", help="Discovery results: entity keys, invariant expressions")
    args = parser.parse_args()

    mg = load_graph()
    if repair_graph(mg):
        print("(Graph repaired: backfilled exp_id attrs and IMPROVED_OVER edges)\n")
    G = mg.graph

    if args.exp:
        report_single_exp(G, args.exp, mg=mg)
    elif args.col:
        report_single_col(G, args.col, mg=mg)
    elif args.central:
        report_centrality(G)
    elif args.ablation:
        report_ablation(G)
    elif args.longest_path:
        report_longest_path(G)
    elif args.saturated:
        report_saturated_columns(G)
    elif args.rates:
        report_transform_rates(G)
    elif args.load_bearing:
        report_load_bearing(G)
    elif args.untried:
        report_untried_pairs(G)
    elif args.shap:
        report_shap(G)
    elif args.hypotheses:
        report_hypotheses(mg)
    elif args.correlations:
        report_correlations(mg)
    elif args.edges:
        report_edge_types(mg)
    elif args.failed:
        report_failed(G)
    elif args.context:
        report_context(mg)
    elif args.coverage:
        report_coverage(mg)
    elif args.discovery:
        report_discovery(mg)
    else:
        report_timeline(G)
        report_improvement_chain(G, mg=mg)
        report_longest_path(G)
        report_column_coverage(G, mg=mg)
        report_feature_lineage(G, mg=mg)
        report_reachable_from_column(G, mg=mg)
        report_centrality(G)
        report_ablation(G)
        report_failed_patterns(G)
        report_saturated_columns(G)
        report_transform_rates(G)
        report_load_bearing(G)
        report_shap(G)
        report_hypotheses(mg)
        report_correlations(mg)
        report_edge_types(mg)
        report_discovery(mg)

    print(f"\n{D}")
    print(f"  Graph: {G.number_of_nodes()} nodes  {G.number_of_edges()} edges")
    print(D)


if __name__ == "__main__":
    main()
