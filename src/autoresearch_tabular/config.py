"""config.py — Pydantic configuration model for autoresearch-tabular.

Replaces raw dict from config.yaml with a validated, typed model.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

__all__ = ["AgentConfig", "load_config"]

_VALID_METRICS = {"rmse", "mae", "auc", "logloss", "f1", "accuracy"}


class AgentConfig(BaseModel):
    """Validated configuration for a single dataset run."""

    # Required — must be set in config.yaml
    data_path: str
    target: str
    metric: str

    # Optional — sensible defaults if omitted from config.yaml
    date_col: str | None = None
    exclude_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    n_folds: int = 5
    random_seed: int = 42
    time_budget_minutes: int = 60
    min_delta: float = 0.001

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        if v not in _VALID_METRICS:
            raise ValueError(f"metric must be one of {_VALID_METRICS}, got '{v}'")
        return v

    @field_validator("n_folds")
    @classmethod
    def validate_n_folds(cls, v: int) -> int:
        if v < 2:
            raise ValueError("n_folds must be >= 2")
        return v

    @property
    def is_higher_better(self) -> bool:
        """Return True for metrics where higher values are better."""
        return self.metric in {"auc", "f1", "accuracy"}

    @property
    def is_classification(self) -> bool:
        """Return True for classification metrics."""
        return self.metric in {"auc", "f1", "logloss", "accuracy"}


def load_config(config_path: Path | None = None) -> AgentConfig:
    """Load and validate config.yaml, returning an AgentConfig.

    Args:
        config_path: Path to config.yaml. Defaults to PROJECT_ROOT/config.yaml.

    Returns:
        Validated AgentConfig instance.

    Raises:
        FileNotFoundError: If config.yaml does not exist.
        ValidationError: If required fields are missing or invalid.
    """
    if config_path is None:
        # Resolve relative to project root (3 levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return AgentConfig(**raw)
