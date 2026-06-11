from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from config.loader import load_config

try:
    from .skill_extractor import SKILL_CATALOG, extract_skills
except ImportError:  # pragma: no cover - supports direct script execution
    from skill_extractor import SKILL_CATALOG, extract_skills


DB_PATH = WORKSPACE_ROOT / "data" / "jobs.db"
CONFIG = load_config()
HISTORY_WINDOW_DAYS = int(CONFIG["ml"]["trend_history_days"])
FORECAST_HORIZON_DAYS = int(CONFIG["ml"]["forecast_horizon_days"])


def load_job_records(db_path: str | Path = DB_PATH, limit: int | None = None) -> list[dict[str, Any]]:
    db_file = Path(db_path)
    query = "SELECT id, description, skills, posted_at, fetched_at FROM jobs ORDER BY fetched_at DESC"
    params: tuple[Any, ...] = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (limit,)

    with sqlite3.connect(db_file) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def get_verification_records() -> list[dict[str, Any]]:
    try:
        records = load_job_records(limit=100)
    except sqlite3.Error:
        records = []

    if records:
        return records

    return [
        {
            "id": "mock-1",
            "description": "Python SQL Pandas job building data products.",
            "skills": "Python, SQL, Pandas",
            "posted_at": "2026-06-04T00:00:00+00:00",
            "fetched_at": "2026-06-04T00:30:00+00:00",
        },
        {
            "id": "mock-2",
            "description": "PyTorch NLP LLM RAG systems.",
            "skills": "PyTorch, NLP, LLM, RAG",
            "posted_at": "2026-06-05T00:00:00+00:00",
            "fetched_at": "2026-06-05T00:30:00+00:00",
        },
        {
            "id": "mock-3",
            "description": "Python SQL dashboards Power BI.",
            "skills": "Python, SQL, Power BI",
            "posted_at": "2026-06-06T00:00:00+00:00",
            "fetched_at": "2026-06-06T00:30:00+00:00",
        },
        {
            "id": "mock-4",
            "description": "Docker FastAPI MLOps deployment.",
            "skills": "Docker, FastAPI, MLOps",
            "posted_at": "2026-06-07T00:00:00+00:00",
            "fetched_at": "2026-06-07T00:30:00+00:00",
        },
        {
            "id": "mock-5",
            "description": "Python SQL Spark Kafka streaming.",
            "skills": "Python, SQL, Spark, Kafka",
            "posted_at": "2026-06-08T00:00:00+00:00",
            "fetched_at": "2026-06-08T00:30:00+00:00",
        },
    ]


def _record_skills(record: dict[str, Any]) -> list[str]:
    stored_skills = record.get("skills")
    if isinstance(stored_skills, str) and stored_skills.strip():
        return [skill.strip() for skill in stored_skills.split(",") if skill.strip()]
    return extract_skills(str(record.get("description") or ""))


def build_skill_timeseries(
    records: list[dict[str, Any]],
    *,
    history_days: int | None = None,
    target_skills: list[str] | None = None,
) -> pd.DataFrame:
    history_days = HISTORY_WINDOW_DAYS if history_days is None else history_days
    if not records:
        columns = target_skills or []
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(records)
    timestamp_source = frame["posted_at"].where(frame["posted_at"].notna(), frame["fetched_at"])
    frame["event_date"] = pd.to_datetime(timestamp_source, utc=True, errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["event_date"]).copy()
    if frame.empty:
        columns = target_skills or []
        return pd.DataFrame(columns=columns)

    frame["skill_list"] = frame.apply(_record_skills, axis=1)
    if target_skills:
        allowed = set(target_skills)
        frame["skill_list"] = frame["skill_list"].apply(lambda skills: [skill for skill in skills if skill in allowed])

    last_date = frame["event_date"].max()
    start_date = last_date - pd.Timedelta(days=history_days - 1)
    frame = frame[frame["event_date"] >= start_date].copy()

    exploded = frame.explode("skill_list")
    exploded = exploded.dropna(subset=["skill_list"])

    full_index = pd.date_range(start=start_date, end=last_date, freq="D", tz="UTC")
    skills = target_skills or sorted(exploded["skill_list"].unique().tolist())
    if not skills:
        skills = list(SKILL_CATALOG.keys())

    pivot = pd.DataFrame(0, index=full_index, columns=skills, dtype=float)
    if not exploded.empty:
        grouped = (
            exploded.groupby(["event_date", "skill_list"])
            .size()
            .rename("count")
            .reset_index()
        )
        for _, row in grouped.iterrows():
            skill = row["skill_list"]
            if skill in pivot.columns:
                pivot.loc[row["event_date"], skill] = float(row["count"])

    pivot.index.name = "date"
    return pivot


def fit_ols_forecast(series: pd.Series, forecast_days: int | None = None) -> pd.DataFrame:
    forecast_days = FORECAST_HORIZON_DAYS if forecast_days is None else forecast_days
    if series.empty:
        return pd.DataFrame(columns=["date", "actual", "prediction", "is_forecast"])

    y = series.astype(float).to_numpy()
    x = np.arange(len(y), dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)

    future_x = np.arange(len(y) + forecast_days, dtype=float)
    future_design = np.column_stack([np.ones(len(future_x)), future_x])
    predictions = future_design @ coefficients
    predictions = np.clip(predictions, a_min=0.0, a_max=None)

    forecast_index = pd.date_range(
        start=series.index[0],
        periods=len(future_x),
        freq="D",
        tz="UTC",
    )

    history_cutoff = len(y)
    return pd.DataFrame(
        {
            "date": forecast_index,
            "actual": np.concatenate([y, np.full(forecast_days, np.nan)]),
            "prediction": predictions,
            "is_forecast": [False] * history_cutoff + [True] * forecast_days,
        }
    )


def forecast_skill_trends(
    records: list[dict[str, Any]],
    *,
    history_days: int | None = None,
    forecast_days: int | None = None,
    target_skills: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    timeseries = build_skill_timeseries(
        records,
        history_days=history_days,
        target_skills=target_skills,
    )

    forecasts: dict[str, pd.DataFrame] = {}
    for skill in timeseries.columns:
        forecasts[skill] = fit_ols_forecast(timeseries[skill], forecast_days=forecast_days)
    return forecasts


if __name__ == "__main__":
    verification_records = get_verification_records()
    selected_skills = ["Python", "SQL", "LLM"]
    timeseries = build_skill_timeseries(
        verification_records,
        history_days=HISTORY_WINDOW_DAYS,
        target_skills=selected_skills,
    )
    forecasts = forecast_skill_trends(
        verification_records,
        history_days=HISTORY_WINDOW_DAYS,
        forecast_days=FORECAST_HORIZON_DAYS,
        target_skills=selected_skills,
    )

    assert timeseries.shape[0] == 7
    assert set(timeseries.columns) == set(selected_skills)
    assert all(len(frame) == 14 for frame in forecasts.values())
    assert all((frame["prediction"] >= 0).all() for frame in forecasts.values())

    print("Trend forecast verification passed.")
    print(f"Timeseries shape: {timeseries.shape}")
    for skill, forecast in forecasts.items():
        latest_prediction = forecast.loc[forecast["is_forecast"], "prediction"].iloc[0]
        print(f"{skill}: first forecasted value = {latest_prediction:.2f}")
