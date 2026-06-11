from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

try:
    from .cleaner import build_canonical_job, clean_jobs
    from .fetcher import IndeedJobSpyClient, JSearchClient, JobQuery, fetch_jobs
except ImportError:  # pragma: no cover - supports direct script execution
    from cleaner import build_canonical_job, clean_jobs
    from fetcher import IndeedJobSpyClient, JSearchClient, JobQuery, fetch_jobs


LOGGER = logging.getLogger(__name__)
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from config.loader import load_config


DATA_DIR = WORKSPACE_ROOT / "data"
DB_PATH = DATA_DIR / "jobs.db"
CONFIG = load_config()
DEFAULT_QUERY_PAIRS = [
    ("data scientist", "Delhi"),
    ("ML engineer", "Bangalore"),
    ("AI engineer", "remote"),
    ("data analyst", "Noida"),
]
DEFAULT_QUERY_MATRIX = [JobQuery(search_term=search_term, location=location) for search_term, location in DEFAULT_QUERY_PAIRS]
SCHEDULE_INTERVAL_HOURS = int(CONFIG["ingestion"]["schedule_interval_hours"])

CREATE_JOBS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    company TEXT,
    location TEXT,
    salary TEXT,
    description TEXT,
    skills TEXT,
    posted_at TEXT,
    fetched_at TEXT,
    source TEXT
)
"""

INSERT_JOB_SQL = """
INSERT OR IGNORE INTO jobs (
    id,
    title,
    company,
    location,
    salary,
    description,
    skills,
    posted_at,
    fetched_at,
    source
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

@dataclass
class PipelineRunStats:
    queries_processed: int = 0
    raw_records_seen: int = 0
    records_cleaned: int = 0
    records_inserted: int = 0
    records_ignored: int = 0
    source_failures: list[str] = field(default_factory=list)
    clean_failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def configure_logging(level: int = logging.INFO) -> None:
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _table_columns(connection: sqlite3.Connection, table_name: str) -> list[str]:
    return [row[1] for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()]


def _copy_legacy_jobs_table(connection: sqlite3.Connection, legacy_table_name: str = "jobs_legacy") -> None:
    legacy_columns = _table_columns(connection, legacy_table_name)
    if not legacy_columns:
        connection.execute(f"DROP TABLE IF EXISTS {legacy_table_name}")
        return

    source_expression = "COALESCE(source, 'jsearch')" if "source" in legacy_columns else "'jsearch'"
    connection.execute(
        f"""
        INSERT OR IGNORE INTO jobs (
            id,
            title,
            company,
            location,
            salary,
            description,
            skills,
            posted_at,
            fetched_at,
            source
        )
        SELECT
            id,
            title,
            company,
            location,
            salary,
            description,
            skills,
            posted_at,
            fetched_at,
            {source_expression}
        FROM {legacy_table_name}
        """
    )
    connection.execute(f"DROP TABLE IF EXISTS {legacy_table_name}")


def ensure_database(db_path: str | Path = DB_PATH) -> Path:
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_file) as connection:
        connection.execute(CREATE_JOBS_TABLE_SQL)
        if _table_exists(connection, "jobs_legacy"):
            _copy_legacy_jobs_table(connection)

        columns = _table_columns(connection, "jobs")
        if columns and "source" not in columns:
            connection.execute("ALTER TABLE jobs RENAME TO jobs_legacy")
            connection.execute(CREATE_JOBS_TABLE_SQL.replace("IF NOT EXISTS ", ""))
            _copy_legacy_jobs_table(connection)
        connection.execute(
            """
            UPDATE jobs
            SET source = COALESCE(NULLIF(source, ''), 'jsearch')
            WHERE source IS NULL OR source = ''
            """
        )
        connection.commit()
    return db_file


def save_to_db(records: Iterable[dict[str, object]], db_path: str | Path = DB_PATH) -> int:
    inserted, _ignored = insert_jobs(records, db_path=db_path)
    return inserted


def insert_jobs(records: Iterable[dict[str, object]], db_path: str | Path = DB_PATH) -> tuple[int, int]:
    db_file = ensure_database(db_path)
    inserted = 0
    ignored = 0

    with sqlite3.connect(db_file) as connection:
        for record in records:
            cursor = connection.execute(
                INSERT_JOB_SQL,
                (
                    record["id"],
                    record["title"],
                    record["company"],
                    record["location"],
                    record["salary"],
                    record["description"],
                    record["skills"],
                    record["posted_at"],
                    record["fetched_at"],
                    record.get("source") or "jsearch",
                ),
            )
            if cursor.rowcount == 1:
                inserted += 1
            else:
                ignored += 1
        connection.commit()

    return inserted, ignored


def _fetch_for_query(
    query: JobQuery,
    *,
    jsearch_client: JSearchClient | None,
    indeed_client: IndeedJobSpyClient | None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    if jsearch_client is not None:
        try:
            records.extend(jsearch_client.fetch_jobs(query))
        except Exception as exc:
            raise RuntimeError(f"JSearch failed for '{query.as_search_phrase()}': {exc}") from exc

    if indeed_client is not None:
        try:
            records.extend(indeed_client.fetch_jobs(query))
        except Exception as exc:
            raise RuntimeError(f"Indeed failed for '{query.as_search_phrase()}': {exc}") from exc

    return records


def run_pipeline_cycle(
    *,
    db_path: str | Path = DB_PATH,
    query_matrix: list[JobQuery] | None = None,
    enable_jsearch: bool = True,
    enable_indeed: bool = True,
) -> PipelineRunStats:
    configure_logging()
    stats = PipelineRunStats()
    db_file = ensure_database(db_path)
    queries = query_matrix or DEFAULT_QUERY_MATRIX

    jsearch_client = JSearchClient() if enable_jsearch else None
    if jsearch_client is not None and not jsearch_client.is_configured():
        LOGGER.warning("Search credential missing; JSearch source disabled for this run.")
        jsearch_client = None

    indeed_client = IndeedJobSpyClient() if enable_indeed else None
    if indeed_client is not None and not indeed_client.is_available():
        LOGGER.warning("python-jobspy unavailable; Indeed source disabled for this run.")
        indeed_client = None

    LOGGER.info("Starting ingestion cycle against %s", db_file)

    for query in queries:
        stats.queries_processed += 1
        LOGGER.info("Fetching query matrix item: %s", query.as_search_phrase())

        raw_records: list[dict[str, object]] = []

        if jsearch_client is not None:
            try:
                raw_records.extend(jsearch_client.fetch_jobs(query))
            except Exception as exc:
                message = f"JSearch failed for '{query.as_search_phrase()}': {exc}"
                LOGGER.warning(message)
                stats.source_failures.append(message)

        if indeed_client is not None:
            try:
                raw_records.extend(indeed_client.fetch_jobs(query))
            except Exception as exc:
                message = f"Indeed failed for '{query.as_search_phrase()}': {exc}"
                LOGGER.warning(message)
                stats.source_failures.append(message)

        stats.raw_records_seen += len(raw_records)

        try:
            cleaned_records = clean_jobs(raw_records)
        except Exception as exc:
            message = f"Cleaning failed for query '{query.as_search_phrase()}': {exc}"
            LOGGER.warning(message)
            stats.clean_failures.append(message)
            cleaned_records = []

        stats.records_cleaned += len(cleaned_records)
        inserted, ignored = insert_jobs(cleaned_records, db_path=db_file)
        stats.records_inserted += inserted
        stats.records_ignored += ignored
        LOGGER.info(
            "Completed query '%s' | raw=%s cleaned=%s inserted=%s ignored=%s",
            query.as_search_phrase(),
            len(raw_records),
            len(cleaned_records),
            inserted,
            ignored,
        )

    LOGGER.info("Ingestion cycle finished: %s", stats.to_dict())
    return stats


def run_all_queries(query_pairs: list[tuple[str, str]] | None = None) -> None:
    queries = query_pairs or DEFAULT_QUERY_PAIRS
    for search_term, location in queries:
        try:
            raw_jobs = fetch_jobs(search_term, location)
            cleaned_jobs = clean_jobs(raw_jobs)
            count = save_to_db(cleaned_jobs)
            print(f"[OK] {search_term} / {location}: {count} new records saved")
        except Exception as exc:
            print(f"[FAIL] {search_term} / {location}: {exc}")


def build_scheduler(
    *,
    db_path: str | Path = DB_PATH,
    query_matrix: list[JobQuery] | None = None,
):
    from apscheduler.schedulers.blocking import BlockingScheduler

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        run_all_queries,
        trigger="interval",
        hours=SCHEDULE_INTERVAL_HOURS,
        kwargs={
            "query_pairs": DEFAULT_QUERY_PAIRS,
        },
        id="job-market-ingestion",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=900,
    )
    return scheduler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the job market ingestion scheduler.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single ingestion cycle and exit without starting the blocking scheduler.",
    )
    args = parser.parse_args()

    configure_logging()

    if args.once:
        run_all_queries()
        return

    print("=== Running initial fetch ===")
    run_all_queries()
    print("=== Scheduler starting (every 4 hours) ===")
    scheduler = build_scheduler()
    scheduler.start()


if __name__ == "__main__":
    main()
