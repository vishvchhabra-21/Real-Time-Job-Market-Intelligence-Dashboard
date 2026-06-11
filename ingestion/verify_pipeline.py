from __future__ import annotations

import sqlite3
from pathlib import Path

try:
    from .cleaner import MAX_DESCRIPTION_LENGTH, build_canonical_job, normalize_location
    from .scheduler import DB_PATH, ensure_database, insert_jobs
except ImportError:  # pragma: no cover - supports direct script execution
    from cleaner import MAX_DESCRIPTION_LENGTH, build_canonical_job, normalize_location
    from scheduler import DB_PATH, ensure_database, insert_jobs


MOCK_JSEARCH_RECORD = {
    "job_id": "verify-jsearch-001",
    "job_title": "Data Scientist",
    "employer_name": "Acme Analytics",
    "job_location": "New Delhi / NCR",
    "job_salary_currency": "INR",
    "job_min_salary": 1200000,
    "job_max_salary": 1800000,
    "job_salary_period": "yearly",
    "job_description": (
        "<div><p>Build production ML pipelines with Python, SQL, and FastAPI.</p>"
        "<p>Remote collaboration, feature engineering, and dashboard support.</p></div>"
    ),
    "job_posted_at_datetime_utc": "2026-06-10T12:00:00Z",
    "job_required_skills": ["Python", "SQL", "FastAPI"],
    "_source": "jsearch",
}


def cleanup_record(job_id: str, db_path: str | Path = DB_PATH) -> None:
    db_file = ensure_database(db_path)
    with sqlite3.connect(db_file) as connection:
        connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        connection.commit()


def fetch_row_count(job_id: str, db_path: str | Path = DB_PATH) -> int:
    db_file = ensure_database(db_path)
    with sqlite3.connect(db_file) as connection:
        row = connection.execute("SELECT COUNT(*) FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return int(row[0])


def main() -> None:
    canonical = build_canonical_job(
        MOCK_JSEARCH_RECORD,
        source=MOCK_JSEARCH_RECORD["_source"],
        fetched_at="2026-06-11T00:00:00Z",
    )

    assert canonical["location"] == "Delhi"
    assert normalize_location("NCR") == "Delhi"
    assert normalize_location("New Delhi") == "Delhi"
    assert normalize_location("Noida, Uttar Pradesh") == "Noida"
    assert len(canonical["description"] or "") <= MAX_DESCRIPTION_LENGTH
    assert "<" not in (canonical["description"] or "")
    assert canonical["salary"] == "INR 1200000 - 1800000 yearly"

    cleanup_record(canonical["id"])

    try:
        first_inserted, first_ignored = insert_jobs([canonical], db_path=DB_PATH)
        second_inserted, second_ignored = insert_jobs([canonical], db_path=DB_PATH)
        row_count = fetch_row_count(canonical["id"], db_path=DB_PATH)

        assert first_inserted == 1
        assert first_ignored == 0
        assert second_inserted == 0
        assert second_ignored == 1
        assert row_count == 1
    finally:
        cleanup_record(canonical["id"])

    print("Verification passed: schema, normalization, truncation, and duplicate bypass are working.")


if __name__ == "__main__":
    main()
