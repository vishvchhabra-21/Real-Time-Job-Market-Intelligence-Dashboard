from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = WORKSPACE_ROOT / "config" / "settings.yaml"

REQUIRED_CONFIG_PATHS: tuple[tuple[str, ...], ...] = (
    ("ingestion", "schedule_interval_hours"),
    ("ingestion", "queries"),
    ("ingestion", "description_max_chars"),
    ("ml", "skill_keywords"),
    ("ml", "kmeans_min_clusters"),
    ("ml", "kmeans_max_clusters"),
    ("ml", "trend_history_days"),
    ("ml", "forecast_horizon_days"),
    ("dashboard", "cache_ttl_seconds"),
    ("dashboard", "top_skills_display"),
    ("dashboard", "top_jobs_resume_display"),
    ("dashboard", "posting_freshness_max_days"),
)


def _is_empty(value: Any) -> bool:
    return value in (None, "", [], {}, ())


def _require_path(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Missing required config key: {'.'.join(path)}")
        current = current[key]

    if _is_empty(current):
        raise KeyError(f"Missing required config key: {'.'.join(path)}")

    return current


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise TypeError("settings.yaml must contain a mapping at the top level.")

    for path in REQUIRED_CONFIG_PATHS:
        _require_path(loaded, path)

    return loaded
