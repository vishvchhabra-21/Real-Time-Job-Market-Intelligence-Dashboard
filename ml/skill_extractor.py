from __future__ import annotations

import re
import sqlite3
from collections import Counter
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any

try:
    import spacy
    from spacy.language import Language
    from spacy.matcher import PhraseMatcher

    SPACY_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on local environment
    spacy = None
    Language = Any  # type: ignore[assignment]
    PhraseMatcher = Any  # type: ignore[assignment]
    SPACY_AVAILABLE = False


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from config.loader import load_config


DB_PATH = WORKSPACE_ROOT / "data" / "jobs.db"


CONFIG = load_config()

DISPLAY_NAME_OVERRIDES: dict[str, str] = {
    "python": "Python",
    "sql": "SQL",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "scikit-learn": "Scikit-learn",
    "langchain": "LangChain",
    "docker": "Docker",
    "mlops": "MLOps",
    "llm": "LLM",
    "rag": "RAG",
    "xgboost": "XGBoost",
    "nlp": "NLP",
    "pandas": "Pandas",
    "power bi": "Power BI",
    "tableau": "Tableau",
    "spark": "Spark",
    "kafka": "Kafka",
    "fastapi": "FastAPI",
    "numpy": "NumPy",
    "matplotlib": "Matplotlib",
    "seaborn": "Seaborn",
    "streamlit": "Streamlit",
    "flask": "Flask",
    "django": "Django",
    "java": "Java",
    "javascript": "JavaScript",
    "aws": "AWS",
    "azure": "Azure",
    "gcp": "GCP",
    "kubernetes": "Kubernetes",
    "airflow": "Airflow",
    "dbt": "dbt",
    "snowflake": "Snowflake",
    "databricks": "Databricks",
    "hadoop": "Hadoop",
    "hive": "Hive",
    "etl": "ETL",
    "excel": "Excel",
    "git": "Git",
    "linux": "Linux",
    "mongodb": "MongoDB",
    "postgresql": "PostgreSQL",
    "mysql": "MySQL",
    "redis": "Redis",
    "elasticsearch": "Elasticsearch",
    "transformers": "Transformers",
    "hugging face": "Hugging Face",
    "openai": "OpenAI",
    "computer vision": "Computer Vision",
    "opencv": "OpenCV",
    "deep learning": "Deep Learning",
    "machine learning": "Machine Learning",
    "statistics": "Statistics",
    "a/b testing": "A/B Testing",
    "a b testing": "A/B Testing",
    "time series": "Time Series",
    "generative ai": "Generative AI",
    "prompt engineering": "Prompt Engineering",
    "vector database": "Vector Database",
    "fine-tuning": "Fine-tuning",
    "bert": "BERT",
    "keras": "Keras",
    "lightgbm": "LightGBM",
    "looker": "Looker",
    "rest api": "REST API",
    "microservices": "Microservices",
    "scala": "Scala",
}


def _display_skill_name(keyword: str) -> str:
    normalized = " ".join(keyword.split()).lower()
    return DISPLAY_NAME_OVERRIDES.get(normalized, keyword.strip().title())


def _skill_aliases(keyword: str) -> list[str]:
    normalized = normalize_text(keyword)
    compact = normalized.replace(" ", "")
    variants = {
        keyword.strip().lower(),
        normalized,
        compact,
        normalized.replace("-", " "),
        normalized.replace("/", " "),
    }
    return [alias for alias in sorted(variants) if alias]


def build_skill_catalog(skill_keywords: list[str] | None = None) -> dict[str, list[str]]:
    keywords = skill_keywords or list(CONFIG["ml"]["skill_keywords"])
    catalog: dict[str, list[str]] = {}
    for keyword in keywords:
        canonical_source = " ".join(str(keyword).split())
        if not canonical_source:
            continue
        canonical = _display_skill_name(canonical_source)
        catalog[canonical] = _skill_aliases(canonical_source)
    return catalog
def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    cleaned = text.lower()
    cleaned = cleaned.replace("/", " ")
    cleaned = re.sub(r"[^a-z0-9+\-#.\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


SKILL_CATALOG = build_skill_catalog()


def create_nlp() -> Language | None:
    if not SPACY_AVAILABLE:
        return None
    return spacy.blank("en")


def build_skill_matcher(
    nlp: Language | None = None,
    skill_catalog: dict[str, list[str]] | None = None,
) -> tuple[Language | None, PhraseMatcher | dict[str, set[str]]]:
    catalog = skill_catalog or SKILL_CATALOG
    nlp = nlp or create_nlp()
    if SPACY_AVAILABLE and nlp is not None:
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        for canonical, aliases in catalog.items():
            patterns = [nlp.make_doc(alias) for alias in {canonical.lower(), *aliases}]
            matcher.add(canonical, patterns)
        return nlp, matcher

    fallback_catalog = {
        canonical: {normalize_text(canonical), *(normalize_text(alias) for alias in aliases)}
        for canonical, aliases in catalog.items()
    }
    return None, fallback_catalog


@lru_cache(maxsize=1)
def get_default_matcher() -> tuple[Language | None, Any]:
    """Build the default skill matcher once per process.

    Constructing a spaCy PhraseMatcher over the full catalog is expensive, so
    callers that pass no explicit matcher share this cached instance instead of
    rebuilding it for every record.
    """
    return build_skill_matcher()


def _fallback_alias_matches(alias: str, normalized: str) -> bool:
    """Whole-word fallback match so 'git' never fires inside 'digital'."""
    if not alias:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized) is not None


def extract_skills(
    text: str | None,
    *,
    nlp: Language | None = None,
    matcher: PhraseMatcher | dict[str, set[str]] | None = None,
    skill_catalog: dict[str, list[str]] | None = None,
) -> list[str]:
    catalog = skill_catalog or SKILL_CATALOG
    if matcher is None:
        if skill_catalog is None and nlp is None:
            nlp, matcher = get_default_matcher()
        else:
            nlp, matcher = build_skill_matcher(nlp=nlp, skill_catalog=catalog)

    normalized = normalize_text(text)
    if not normalized:
        return []

    seen: set[str] = set()
    if SPACY_AVAILABLE and nlp is not None and not isinstance(matcher, dict):
        doc = nlp(normalized)
        for match_id, _, _ in matcher(doc):
            skill_name = nlp.vocab.strings[match_id]
            seen.add(skill_name)
    else:
        fallback_catalog = matcher if isinstance(matcher, dict) else build_skill_matcher(skill_catalog=catalog)[1]
        for canonical, aliases in fallback_catalog.items():
            if any(_fallback_alias_matches(alias, normalized) for alias in aliases):
                seen.add(canonical)

    ordered_skills = [skill for skill in catalog if skill in seen]
    return ordered_skills if ordered_skills else sorted(seen)


def extract_skills_from_records(
    records: list[dict[str, Any]],
    *,
    text_key: str = "description",
    skill_catalog: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    nlp, matcher = build_skill_matcher(skill_catalog=skill_catalog)
    enriched_records: list[dict[str, Any]] = []
    for record in records:
        text = str(record.get(text_key) or "")
        skills = extract_skills(text, nlp=nlp, matcher=matcher, skill_catalog=skill_catalog)
        enriched_record = dict(record)
        enriched_record["extracted_skills"] = skills
        enriched_records.append(enriched_record)
    return enriched_records


def aggregate_skill_counts(records: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        for skill in record.get("extracted_skills", []):
            counts[skill] += 1
    return counts


def load_job_records(db_path: str | Path = DB_PATH, limit: int | None = None) -> list[dict[str, Any]]:
    db_file = Path(db_path)
    query = "SELECT id, title, company, location, description, skills, posted_at FROM jobs ORDER BY fetched_at DESC"
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
        records = load_job_records(limit=25)
    except sqlite3.Error:
        records = []

    if records:
        return records

    return [
        {
            "id": "mock-1",
            "title": "Machine Learning Engineer",
            "company": "Mock AI Labs",
            "location": "Delhi",
            "description": "Build Python, SQL, PyTorch and FastAPI services for NLP and LLM products.",
            "skills": None,
            "posted_at": "2026-06-10T00:00:00+00:00",
        },
        {
            "id": "mock-2",
            "title": "Analytics Engineer",
            "company": "Mock Data Works",
            "location": "Bangalore",
            "description": "Create dashboards with Power BI, Tableau, SQL and Pandas for hiring intelligence.",
            "skills": None,
            "posted_at": "2026-06-09T00:00:00+00:00",
        },
    ]


if __name__ == "__main__":
    verification_records = get_verification_records()
    enriched = extract_skills_from_records(verification_records)
    counts = aggregate_skill_counts(enriched)

    assert len(enriched) == len(verification_records)
    assert all(isinstance(record["extracted_skills"], list) for record in enriched)
    assert any(record["extracted_skills"] for record in enriched)

    print("Skill extractor verification passed.")
    print(f"Records evaluated: {len(enriched)}")
    print(f"Tracked vocabulary size: {len(SKILL_CATALOG)}")
    print(f"Top extracted skills: {counts.most_common(5)}")
