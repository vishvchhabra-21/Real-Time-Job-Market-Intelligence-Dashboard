from __future__ import annotations

import re
import sqlite3
import sys
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from config.loader import load_config
from ml.skill_extractor import extract_skills, get_default_matcher


DB_PATH = WORKSPACE_ROOT / "data" / "jobs.db"
CONFIG = load_config()
DEFAULT_TOP_N = int(CONFIG["dashboard"]["top_jobs_resume_display"])

MODELS_DIR = WORKSPACE_ROOT / "ml" / "models"
CATEGORY_MODEL_PATH = MODELS_DIR / "resume_category_model.joblib"
CATEGORY_BOOST_POINTS = 8.0

# Raw TF-IDF cosine similarity between a short resume and a long, differently
# worded job description rarely climbs above ~0.30-0.35 even for an excellent
# match, so reporting cosine * 100 directly makes every result look "Borderline".
# The displayed Fit Score therefore blends two interpretable signals:
#   * skill coverage  - the fraction of the job's required skills the resume has
#   * text similarity - cosine normalised against a realistic strong-match ceiling
# `similarity` keeps the raw cosine value for transparency; `score_pct` is the
# calibrated, human-meaningful fit percentage.
COSINE_STRONG_MATCH = 0.35
SKILL_COVERAGE_WEIGHT = 0.55
TEXT_SIMILARITY_WEIGHT = 0.45
MIN_SKILLS_FOR_COVERAGE = 3

# Maps a predicted Kaggle resume category to title keywords worth a small
# ranking boost. Categories with no plausible match in this job market are
# omitted, so they only ever appear as informational labels.
CATEGORY_TITLE_HINTS: dict[str, tuple[str, ...]] = {
    "Data Science": ("data scientist", "data science", "machine learning", "ai engineer", "ml engineer", "deep learning"),
    "Python Developer": ("python developer", "backend", "software engineer"),
    "DevOps Engineer": ("devops", "mlops", "site reliability", "platform engineer"),
    "Database": ("database", "data engineer"),
    "ETL Developer": ("etl", "data engineer"),
    "Hadoop": ("data engineer", "big data"),
    "Business Analyst": ("business analyst", "business intelligence", "bi analyst"),
    "Testing": ("test", "quality", "qa"),
    "Automation Testing": ("test", "quality", "qa", "automation"),
    "Network Security Engineer": ("security",),
    "Java Developer": ("java", "backend", "software engineer"),
    "DotNet Developer": (".net", "dotnet", "backend", "software engineer"),
}

QUALITY_MODEL_PATH = MODELS_DIR / "resume_quality_model.joblib"

_URL_PATTERN = re.compile(r"http\S+|www\S+")
_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_EXPERIENCE_PATTERN = re.compile(r"(\d{1,2})\+?\s*(?:years?|yrs?)", re.IGNORECASE)
_PROJECT_PATTERN = re.compile(r"\bproject", re.IGNORECASE)

# Ordered keyword -> level lookup shared by training and inference so that both
# the dataset's "Education" column values (e.g. "B.Tech", "PhD") and free-text
# resume mentions (e.g. "Bachelor of Technology") map to the same scale.
EDUCATION_LEVELS: tuple[tuple[str, float], ...] = (
    ("phd", 4.0),
    ("doctorate", 4.0),
    ("m.tech", 3.0),
    ("mtech", 3.0),
    ("m.sc", 3.0),
    ("msc", 3.0),
    ("mba", 3.0),
    ("master", 3.0),
    ("mca", 2.5),
    ("b.tech", 2.0),
    ("btech", 2.0),
    ("b.e", 2.0),
    ("b.sc", 2.0),
    ("bsc", 2.0),
    ("bca", 2.0),
    ("bachelor", 2.0),
)


def normalize_text(text: str | None) -> str:
    return " ".join((text or "").split())


def clean_resume_text(text: str | None) -> str:
    """Lowercase and strip URLs/punctuation to match the trained vectorizer's preprocessing."""
    lowered = (text or "").lower()
    no_urls = _URL_PATTERN.sub(" ", lowered)
    alnum_only = _NON_ALNUM_PATTERN.sub(" ", no_urls)
    return _WHITESPACE_PATTERN.sub(" ", alnum_only).strip()


def extract_experience_years(text: str | None) -> float:
    """Return the largest 'N years/yrs' figure mentioned, or 0.0 if none is found."""
    matches = _EXPERIENCE_PATTERN.findall(text or "")
    return float(max((int(value) for value in matches), default=0))


def extract_projects_count(text: str | None) -> float:
    """Approximate a project count from how often 'project' is mentioned, capped at 10."""
    return float(min(len(_PROJECT_PATTERN.findall(text or "")), 10))


def education_level_from_text(text: str | None) -> float:
    """Map the highest education keyword found to an ordinal level (1.0 = unknown/lowest)."""
    lowered = (text or "").lower()
    for keyword, level in EDUCATION_LEVELS:
        if keyword in lowered:
            return level
    return 1.0


def extract_text_from_upload(file_bytes: bytes, filename: str) -> str:
    """Extract text from an in-memory resume upload (PDF, DOCX, or TXT).

    The file is never written to disk; everything happens on the in-memory
    ``BytesIO`` buffer per the project's no-persistence privacy requirement.
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".txt":
        return file_bytes.decode("utf-8", errors="ignore")

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".docx":
        from docx import Document

        document = Document(BytesIO(file_bytes))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    raise ValueError(f"Unsupported resume file type: {suffix}")


@lru_cache(maxsize=1)
def _load_category_model() -> dict[str, Any] | None:
    if not CATEGORY_MODEL_PATH.exists():
        return None
    try:
        bundle = joblib.load(CATEGORY_MODEL_PATH)
    except Exception:
        return None
    if not {"vectorizer", "classifier", "label_encoder"}.issubset(bundle):
        return None
    return bundle


def predict_resume_category(resume_text: str) -> dict[str, Any] | None:
    """Predict a resume's job category using the Kaggle-trained classifier, if available."""
    bundle = _load_category_model()
    if bundle is None:
        return None

    cleaned = clean_resume_text(resume_text)
    if not cleaned:
        return None

    features = bundle["vectorizer"].transform([cleaned])
    probabilities = bundle["classifier"].predict_proba(features)[0]
    best_index = int(np.argmax(probabilities))
    category = str(bundle["label_encoder"].inverse_transform([best_index])[0])
    confidence = round(float(probabilities[best_index]) * 100.0, 1)
    return {"category": category, "confidence": confidence}


@lru_cache(maxsize=1)
def _load_quality_model() -> dict[str, Any] | None:
    if not QUALITY_MODEL_PATH.exists():
        return None
    try:
        bundle = joblib.load(QUALITY_MODEL_PATH)
    except Exception:
        return None
    if not {"vectorizer", "scaler", "regressor", "classifier"}.issubset(bundle):
        return None
    return bundle


def predict_resume_quality(resume_text: str) -> dict[str, Any] | None:
    """Predict an AI-style quality score and hire likelihood, if the quality model is available.

    Trained on the "AI-Powered Resume Screening Dataset 2025" (Skills,
    Certifications, Experience, Education, Projects Count -> AI Score and
    Recruiter Decision).
    """
    bundle = _load_quality_model()
    if bundle is None:
        return None

    cleaned = clean_resume_text(resume_text)
    if not cleaned:
        return None

    text_features = bundle["vectorizer"].transform([cleaned])
    numeric_features = bundle["scaler"].transform(
        [
            [
                extract_experience_years(resume_text),
                extract_projects_count(resume_text),
                education_level_from_text(resume_text),
            ]
        ]
    )
    features = hstack([text_features, csr_matrix(numeric_features)]).tocsr()

    ai_score = float(np.clip(bundle["regressor"].predict(features)[0], 0.0, 100.0))
    hire_probability = float(bundle["classifier"].predict_proba(features)[0][1]) * 100.0
    return {"ai_score": round(ai_score, 1), "hire_likelihood": round(hire_probability, 1)}


def load_job_records(db_path: str | Path = DB_PATH, limit: int | None = None) -> list[dict[str, Any]]:
    db_file = Path(db_path)
    query = "SELECT id, title, company, location, description, posted_at FROM jobs ORDER BY fetched_at DESC"
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
        records = load_job_records(limit=50)
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
            "description": "Build Python, SQL, PyTorch and FastAPI services for NLP products.",
            "posted_at": "2026-06-10T00:00:00+00:00",
        },
        {
            "id": "mock-2",
            "title": "BI Analyst",
            "company": "Mock Insights",
            "location": "Noida",
            "description": "Create SQL, Power BI and Tableau dashboards for executive reporting.",
            "posted_at": "2026-06-10T00:00:00+00:00",
        },
        {
            "id": "mock-3",
            "title": "LLM Engineer",
            "company": "Mock GenAI",
            "location": "Remote",
            "description": "Develop LangChain, RAG and LLM systems with Docker and Python.",
            "posted_at": "2026-06-10T00:00:00+00:00",
        },
    ]


def _job_skill_set(record: dict[str, Any], *, nlp: Any = None, matcher: Any = None) -> set[str]:
    """Resolve a job's required skills, preferring pre-computed lists over text extraction."""
    stored_list = record.get("skill_list")
    if isinstance(stored_list, list) and stored_list:
        return {str(skill).strip() for skill in stored_list if str(skill).strip()}

    stored_csv = record.get("skills")
    if isinstance(stored_csv, str) and stored_csv.strip():
        return {part.strip() for part in stored_csv.split(",") if part.strip()}

    description = str(record.get("description") or record.get("title") or "")
    return set(extract_skills(description, nlp=nlp, matcher=matcher))


def calibrate_fit_score(cosine: float, resume_skills: set[str], job_skills: set[str]) -> float:
    """Blend skill coverage and normalised text similarity into a 0-100 fit percentage."""
    text_component = min(1.0, max(0.0, cosine) / COSINE_STRONG_MATCH)
    if job_skills:
        # Judge coverage against a denominator floor so a job listing just one
        # or two skills the resume happens to have cannot reach 100% coverage and
        # outrank a richer, genuinely well-matched listing. Jobs with 3+ listed
        # skills use their true coverage.
        matched = len(resume_skills & job_skills)
        coverage = matched / max(len(job_skills), MIN_SKILLS_FOR_COVERAGE)
        fit = SKILL_COVERAGE_WEIGHT * coverage + TEXT_SIMILARITY_WEIGHT * text_component
    else:
        # No extractable skills to compare against: fall back to text signal only.
        fit = text_component
    return round(min(100.0, fit * 100.0), 2)


def build_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2),
        max_features=max_features,
    )


def score_resume_against_jobs(
    resume_text: str,
    job_records: list[dict[str, Any]],
    *,
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    top_n = DEFAULT_TOP_N if top_n is None else top_n
    if not normalize_text(resume_text):
        raise ValueError("Resume text must not be empty.")
    if not job_records:
        return []

    documents = [normalize_text(str(job.get("description") or job.get("title") or "")) for job in job_records]
    vectorizer = build_vectorizer()
    matrix = vectorizer.fit_transform(documents + [resume_text])

    job_matrix = matrix[:-1]
    resume_vector = matrix[-1]
    similarities = cosine_similarity(resume_vector, job_matrix).ravel()
    similarities = np.clip(similarities, a_min=0.0, a_max=1.0)

    nlp, matcher = get_default_matcher()
    resume_skills = set(extract_skills(resume_text, nlp=nlp, matcher=matcher))

    scored_records: list[dict[str, Any]] = []
    for record, score in zip(job_records, similarities, strict=False):
        job_skills = _job_skill_set(record, nlp=nlp, matcher=matcher)
        fit_pct = calibrate_fit_score(float(score), resume_skills, job_skills)
        scored_records.append(
            {
                "id": record.get("id"),
                "title": record.get("title"),
                "company": record.get("company"),
                "location": record.get("location"),
                "posted_at": record.get("posted_at"),
                "similarity": round(float(score), 6),
                "score_pct": fit_pct,
            }
        )

    category_info = predict_resume_category(resume_text)
    if category_info is not None:
        hints = CATEGORY_TITLE_HINTS.get(category_info["category"], ())
        if hints:
            for record in scored_records:
                title = (record.get("title") or "").lower()
                if any(hint in title for hint in hints):
                    record["score_pct"] = round(min(100.0, record["score_pct"] + CATEGORY_BOOST_POINTS), 2)

    scored_records.sort(key=lambda item: item["score_pct"], reverse=True)
    return scored_records[:top_n]


if __name__ == "__main__":
    verification_records = get_verification_records()
    verification_resume = """
    Python SQL machine learning engineer with PyTorch, FastAPI, NLP and Docker experience.
    Built production APIs, trained models, and deployed analytics services.
    """
    scored = score_resume_against_jobs(verification_resume, verification_records, top_n=3)

    assert scored
    assert len(scored) <= 3
    assert all(0.0 <= item["score_pct"] <= 100.0 for item in scored)
    assert scored[0]["score_pct"] >= scored[-1]["score_pct"]

    print("Resume scorer verification passed.")
    print(f"Jobs evaluated: {len(verification_records)}")
    print(f"Top match: {scored[0]['title']} at {scored[0]['score_pct']}%")
