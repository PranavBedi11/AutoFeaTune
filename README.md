<div align="center">

# autoresearch-tabular

**An AI agent that engineers features while you sleep.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

</div>

---

Point it at a CSV. Write two sentences about what you're predicting. Walk away.

The agent reads your data, forms a hypothesis, writes a feature transform, trains XGBoost, and decides whether to keep it — all in a loop. A persistent memory graph accumulates every experiment, every SHAP score, every failed attempt. The next session picks up exactly where the last one left off, armed with everything it learned.

The loop only moves forward. If a feature doesn't improve the composite score, the git ratchet snaps back to the last known-good state and the agent tries something else.

---

## What it looks like

**Experiment progress — scores, deltas, SHAP importance, live hypothesis tracking:**

![Experiment Progress](static/images/expprogress.png)

**Memory graph — every feature, experiment, hypothesis, and column relationship:**

![Memory Graph](static/images/Graph.png)

---

## How it works

```
You write:
  config.yaml   ← dataset path, target column, metric
  program.md    ← what you're predicting, domain knowledge, ideas

The agent loops:
  1. Read memory graph — what worked, what failed, what's untried
  2. Form a hypothesis — write it to the graph before touching any code
  3. Edit features.py — one targeted transform at a time
  4. Train — XGBoost 5-fold CV, composite score = accuracy − 0.001 × n_features
  5. Ratchet — kept? commit. dropped? revert. repeat.

Stops when:
  • Time budget exceeded
  • N consecutive failures with no improvement path
  • You manually intervene
```

The agent never cheats. Statistical queries are rate-limited and run only on training data. Target-aware features use out-of-fold encoding so training rows never see their own labels.

---

## Why this is different

**Memory compounds across sessions.** The graph isn't reset between runs. It tracks which transform families win on your dataset, which columns are exhausted, which hypotheses were wrong. Session 3 is smarter than session 1.

**One file, total control.** The agent only edits `features.py`. The training harness is fixed. You always know exactly what the agent changed and why.

**Git ratchet = no regression, ever.** `HEAD` always points to the best experiment so far. If a session crashes mid-run, you're still on solid ground.

**SHAP-informed pruning.** After every kept experiment, SHAP values are written to the memory graph. The agent uses them to identify dead features, high-variance features, and untapped columns — not gut instinct.

**Discovery phase finds hidden structure.** Before the loop starts, the system scans for invariant expressions, entity keys, and column relationships that raw feature engineering would miss entirely.

---

## Quickstart

> Requires Python 3.12+, [uv](https://docs.astral.sh/uv/getting-started/installation/), and an AI agent (Claude Code, OpenAI Codex, or similar).

```bash
# 1. Clone
git clone https://github.com/ashsanjay2005/autoresearch-tabular.git
cd autoresearch-tabular
uv sync

# 2. Zero-config demo — downloads California Housing, wires everything up
uv run autoresearch demo

# 3. Initialize and prepare
uv run autoresearch init
uv run autoresearch prepare

# 4. Discover hidden structure (takes 10–20 min on large datasets — let it finish)
uv run autoresearch discover

# 5. Hand off to the agent
claude AGENTS.md   # or whichever agent you use
```

The agent reads `AGENTS.md` and runs until the stop condition is hit. You can leave it overnight.

---

## Bring your own dataset

```bash
# Interactive setup
uv run autoresearch setup data/my_data.csv

# Or edit config.yaml directly
data_path: data/my_data.csv
target: price
metric: rmse

# Then fill in program.md — this is the most important thing you do
# The more domain knowledge you put here, the better the agent performs
```

**`program.md` matters.** It's where you tell the agent what signals you expect, what to avoid, and what to try. A blank `program.md` works, but a detailed one is the difference between good and great results.

---

## Supported metrics

| Metric | Direction | Task |
|--------|-----------|------|
| `rmse` | lower is better | regression |
| `mae` | lower is better | regression |
| `accuracy` | higher is better | classification |
| `auc` | higher is better | binary classification |
| `f1` | higher is better | imbalanced classification |
| `logloss` | lower is better | classification |

---

## Commands

```bash
uv run autoresearch demo              # zero-config California Housing demo
uv run autoresearch setup <csv>       # interactive project setup
uv run autoresearch init              # git init, memory graph, baseline commit
uv run autoresearch prepare           # load data, splits, Column nodes
uv run autoresearch discover          # entity keys, invariant expressions
uv run autoresearch status            # STOP / EXPLORE / CONTINUE signal
uv run autoresearch query <type>      # leakage-safe statistical queries
uv run autoresearch inspect --timeline    # experiment history
uv run autoresearch inspect --shap        # SHAP feature importance
uv run autoresearch inspect --hypotheses  # hypothesis calibration
uv run autoresearch inspect --rates       # best-performing transform families
uv run autoresearch inspect --failed      # exact reverted descriptions (never repeat these)
uv run autoresearch inspect --context     # compressed full history

uv run python visualize.py            # interactive graph dashboard → http://localhost:8050
```

---

## Architecture

```
autoresearch-tabular/
├── src/autoresearch_tabular/
│   ├── features.py        ← the only file the agent edits
│   ├── train.py           ← XGBoost 5-fold CV harness (fixed)
│   ├── prepare.py         ← data loading, splits, Column nodes (fixed)
│   ├── memory_graph.py    ← NetworkX graph, experiment + hypothesis tracking
│   ├── inspect_graph.py   ← analytics queries and CLI reports
│   ├── discover.py        ← entity key detection, invariant scanning
│   ├── query.py           ← leakage-safe statistical queries
│   ├── config.py          ← Pydantic config model
│   └── cli.py             ← CLI entry point
├── AGENTS.md              ← full loop instructions for the agent
├── visualize.py           ← Dash + Cytoscape memory graph viewer
├── config.yaml            ← dataset, target, metric
└── program.md             ← domain brief (you write this)
```

The constraint that the agent only edits `features.py` is intentional. It makes every experiment auditable, every regression traceable, and every session reproducible.

---

## Scoring

```
Higher-is-better (accuracy, auc, f1):
  composite = cv_score − 0.001 × n_features

Lower-is-better (rmse, mae, logloss):
  composite = cv_score + 0.001 × n_features
```

This forces the agent to justify every feature it adds. A transform that improves accuracy by 0.05% but adds 10 features nets zero gain. The agent has to find features that are genuinely predictive, not just correlated noise.

---

## Development

```bash
uv sync
uv run pytest -v
```

The one rule: **the agent only edits `features.py`**. `train.py` and `prepare.py` are fixed — touching them breaks the git ratchet.

---

## Contributing

PRs welcome. Open an issue first for anything beyond a small bug fix. Keep changes focused — one thing per PR.

---

## License

MIT
