from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import requests

from dashboard.app import call_market_chat
from ingestion.cleaner import normalize_location
from ingestion.scheduler import ensure_database, insert_jobs
from ml.clustering import cluster_job_descriptions
from ml.resume_scorer import score_resume_against_jobs
from ml.resume_scorer import get_verification_records as get_resume_records
from ml.resume_scorer import (
    clean_resume_text,
    education_level_from_text,
    extract_experience_years,
    extract_projects_count,
    extract_text_from_upload,
    predict_resume_category,
    predict_resume_quality,
)
from ml.skill_extractor import extract_skills, extract_skills_from_records
from ml.trend_forecast import build_skill_timeseries, fit_ols_forecast


def _count_rows(db_path: Path) -> int:
    with sqlite3.connect(db_path) as connection:
        row = connection.execute("SELECT COUNT(*) FROM jobs").fetchone()
    return int(row[0] if row else 0)


def test_pipeline_inserts_and_deduplicates(tmp_path: Path) -> None:
    db_path = ensure_database(tmp_path / "jobs.db")

    records = [
        {
            "id": "job-1",
            "title": "Data Scientist",
            "company": "Alpha",
            "location": "Delhi",
            "salary": None,
            "description": "Python SQL analytics role.",
            "skills": "Python, SQL",
            "posted_at": "2026-06-10T00:00:00+00:00",
            "fetched_at": "2026-06-10T01:00:00+00:00",
        },
        {
            "id": "job-2",
            "title": "ML Engineer",
            "company": "Beta",
            "location": "Bangalore",
            "salary": None,
            "description": "PyTorch FastAPI deployment role.",
            "skills": "PyTorch, FastAPI",
            "posted_at": "2026-06-10T00:00:00+00:00",
            "fetched_at": "2026-06-10T01:00:00+00:00",
        },
        {
            "id": "job-3",
            "title": "LLM Engineer",
            "company": "Gamma",
            "location": "Remote",
            "salary": None,
            "description": "LangChain RAG systems role.",
            "skills": "LangChain, RAG",
            "posted_at": "2026-06-10T00:00:00+00:00",
            "fetched_at": "2026-06-10T01:00:00+00:00",
        },
    ]

    inserted, ignored = insert_jobs(records, db_path=db_path)
    assert inserted == 3
    assert ignored == 0

    duplicate_attempt = [records[0], records[1]]
    inserted_again, ignored_again = insert_jobs(duplicate_attempt, db_path=db_path)
    assert inserted_again == 0
    assert ignored_again == 2
    assert _count_rows(db_path) == 3


@pytest.mark.parametrize(
    "raw_location",
    ["NCR", "New Delhi", "Delhi/NCR", "New Delhi, India"],
)
def test_location_normalizer(raw_location: str) -> None:
    assert normalize_location(raw_location) == "Delhi"


def test_skill_extractor_on_real_description() -> None:
    jd = "We need a Python developer with SQL and LangChain experience for internal AI workflows."
    extracted = extract_skills(jd)

    assert "Python" in extracted
    assert "SQL" in extracted
    assert "LangChain" in extracted


def test_resume_scorer_range() -> None:
    resume = "Python SQL ML engineer with FastAPI and Docker project delivery experience."
    scored = score_resume_against_jobs(resume, get_resume_records(), top_n=20)

    assert scored
    assert all(0.0 <= item["similarity"] <= 1.0 for item in scored)
    assert all(0.0 <= item["score_pct"] <= 100.0 for item in scored)


def test_empty_db_graceful_degradation() -> None:
    empty_df = pd.DataFrame()
    empty_records = empty_df.to_dict(orient="records")

    skill_records = extract_skills_from_records(empty_records)
    clustered_records, artifacts = cluster_job_descriptions(empty_records)
    timeseries = build_skill_timeseries(empty_records, history_days=7, target_skills=["Python"])
    forecast = fit_ols_forecast(pd.Series(dtype=float), forecast_days=7)
    resume_scores = score_resume_against_jobs("Python SQL resume text", empty_records)

    assert skill_records == []
    assert clustered_records == []
    assert artifacts.labels == []
    assert artifacts.cluster_count == 0
    assert artifacts.matrix_shape == (0, 0)
    assert timeseries.empty
    assert forecast.empty
    assert resume_scores == []


def test_clean_resume_text() -> None:
    raw = "Visit http://example.com! Python/SQL & ML-Engineer  (RAG)."
    cleaned = clean_resume_text(raw)

    assert "http" not in cleaned
    assert cleaned == cleaned.lower()
    assert "  " not in cleaned
    assert "python" in cleaned and "sql" in cleaned


def test_extract_text_from_upload_txt() -> None:
    text = extract_text_from_upload(b"Python SQL ML engineer resume", "resume.txt")
    assert "Python SQL ML engineer resume" in text


def test_extract_text_from_upload_unsupported_type() -> None:
    with pytest.raises(ValueError):
        extract_text_from_upload(b"data", "resume.csv")


def test_predict_resume_category_graceful() -> None:
    resume = "Python SQL ML engineer with FastAPI and Docker project delivery experience."
    result = predict_resume_category(resume)

    assert result is None or {"category", "confidence"}.issubset(result)
    if result is not None:
        assert 0.0 <= result["confidence"] <= 100.0


def test_extract_experience_years() -> None:
    assert extract_experience_years("5 years of experience in Python") == 5.0
    assert extract_experience_years("Worked for 2 yrs as a 10+ year veteran") == 10.0
    assert extract_experience_years("No experience figures here") == 0.0


def test_extract_projects_count() -> None:
    text = "Project Alpha, Project Beta, and a capstone project."
    assert extract_projects_count(text) == 3.0
    assert extract_projects_count("No mentions here.") == 0.0


def test_education_level_from_text() -> None:
    assert education_level_from_text("PhD in Computer Science") == 4.0
    assert education_level_from_text("Bachelor of Technology") == 2.0
    assert education_level_from_text("High school diploma") == 1.0


def test_predict_resume_quality_graceful() -> None:
    resume = "Python SQL ML engineer with FastAPI and Docker project delivery experience."
    result = predict_resume_quality(resume)

    assert result is None or {"ai_score", "hire_likelihood"}.issubset(result)
    if result is not None:
        assert 0.0 <= result["ai_score"] <= 100.0
        assert 0.0 <= result["hire_likelihood"] <= 100.0


def test_groq_timeout_fallback() -> None:
    context = {
        "summary_text": "Active filtered listings: 3 | Top skills: Python (2), SQL (2) | Top cities: Delhi (2) | Top companies: Alpha (1)",
        "record_count": 3,
        "top_skills": [("Python", 2), ("SQL", 2)],
        "top_cities": [("Delhi", 2)],
        "top_companies": [("Alpha", 1)],
    }

    with patch("dashboard.app.get_credential", return_value="fake-groq-key"), patch(
        "dashboard.app.requests.post", side_effect=requests.Timeout("request timed out")
    ):
        answer, source = call_market_chat(
            question="What skills are missing for Delhi roles?",
            context=context,
            history=[],
        )

    assert source == "local fallback"
    assert isinstance(answer, str)
    assert answer.strip()
