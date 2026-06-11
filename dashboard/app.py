from __future__ import annotations

import html
import math
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ingestion.fetcher import get_credential
from config.loader import load_config
from ml.resume_scorer import (
    extract_text_from_upload,
    predict_resume_category,
    predict_resume_quality,
    score_resume_against_jobs,
)
from ml.skill_extractor import SKILL_CATALOG, extract_skills, get_default_matcher
from ml.trend_forecast import HISTORY_WINDOW_DAYS, build_skill_timeseries, fit_ols_forecast


APP_TITLE = "Real-Time Job Market Intelligence Dashboard"
APP_SUBTITLE = "Streamlit architecture shell for live listings, skill intelligence, resume matching, and GenAI guidance."
APP_WIRE_TITLE = "📊 REAL-TIME DATA SCIENCE & AI JOB MARKET INTELLIGENCE DASHBOARD (INDIA)"
DB_PATH = ROOT_DIR / "data" / "jobs.db"


CONFIG = load_config()
DASHBOARD_CONFIG = CONFIG["dashboard"]
ML_CONFIG = CONFIG["ml"]
CACHE_TTL_SECONDS = int(DASHBOARD_CONFIG["cache_ttl_seconds"])
TOP_SKILLS_DISPLAY = int(DASHBOARD_CONFIG["top_skills_display"])
TOP_JOBS_RESUME_DISPLAY = int(DASHBOARD_CONFIG["top_jobs_resume_display"])
POSTING_FRESHNESS_MAX_DAYS = int(DASHBOARD_CONFIG["posting_freshness_max_days"])
FORECAST_HORIZON_DAYS = int(ML_CONFIG["forecast_horizon_days"])
CHAT_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
CHAT_MODEL = "llama-3.3-70b-versatile"
SESSION_MESSAGES_KEY = "job_market_chat_messages"
SESSION_CONTEXT_KEY = "job_market_chat_context"
SESSION_RESULTS_KEY = "resume_results"
SESSION_RESUME_TEXT_KEY = "resume_text_input"
SESSION_RESUME_UPLOAD_KEY = "resume_upload_signature"
SESSION_FILTER_KEY = "job_market_filter_signature"
SESSION_CONTROL_KEY = "job_market_control_signature"
SESSION_PROFILE_KEY = "candidate_profile"
MAX_CHAT_TURNS = 6
MAX_CONTEXT_CHARACTERS = 1600

# Design tokens shared between the CSS theme and the Plotly charts.
# Amber = the candidate ("you"); teal = the market. The duotone is semantic,
# not decorative: every amber element on the page traces back to the profile.
COLOR_SIGNAL = "#FFB224"
COLOR_RADAR = "#3FC9B9"
COLOR_DIM = "#8C9AB2"
COLOR_PAPER = "#E9EEF8"
PLOTLY_FONT = dict(family="IBM Plex Mono, monospace", color=COLOR_DIM, size=12)

DOMAIN_KEYWORDS: dict[str, set[str]] = {
    "GenAI / LLM": {
        "llm", "llms", "rag", "langchain", "prompt", "genai", "vector", "embedding",
        "generative ai", "hugging face", "transformers", "openai", "fine-tuning", "bert",
    },
    "Machine Learning": {
        "machine learning", "ml", "pytorch", "tensorflow", "scikit-learn", "xgboost",
        "deep learning", "computer vision", "opencv", "keras", "lightgbm", "time series",
    },
    "Data Engineering": {
        "spark", "kafka", "airflow", "etl", "dbt", "hadoop", "warehouse",
        "snowflake", "databricks", "hive", "data engineer", "data pipeline",
    },
    "Analytics / BI": {
        "sql", "power bi", "tableau", "pandas", "excel", "dashboard",
        "looker", "statistics", "a/b testing", "business analyst", "business intelligence",
    },
    "Backend / Platform": {
        "fastapi", "docker", "api", "microservice", "kubernetes", "aws",
        "azure", "gcp", "flask", "django", "rest api", "linux",
    },
    "General Tech": set(),
}

LOCAL_SUMMARY_INTRO = (
    "I am working from the currently filtered job snapshot only. "
    "I will not retain personal profile data beyond this session."
)

FIT_SCORE_SUCCESS_THRESHOLD = 70.0
FIT_SCORE_WARNING_THRESHOLD = 40.0


def get_fit_score_bucket(score_pct: float) -> str:
    if score_pct >= FIT_SCORE_SUCCESS_THRESHOLD:
        return "success"
    if score_pct >= FIT_SCORE_WARNING_THRESHOLD:
        return "warning"
    return "error"


def render_fit_results_table(result_frame: pd.DataFrame) -> None:
    st.dataframe(
        result_frame,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Fit Score": st.column_config.ProgressColumn(
                "Fit Score",
                help="Calibrated blend of skill coverage and TF-IDF text similarity",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
        },
    )


def configure_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,100..900&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        :root {
            --ink: #070B14;
            --panel: #0C1322;
            --panel-2: #111A2E;
            --line: #1D2940;
            --line-2: #2C3D60;
            --paper: #E9EEF8;
            --dim: #8C9AB2;
            --signal: #FFB224;
            --signal-soft: rgba(255, 178, 36, 0.13);
            --radar: #3FC9B9;
            --radar-soft: rgba(63, 201, 185, 0.12);
            --alert: #F0647A;
            --font-display: 'Archivo', 'IBM Plex Sans', sans-serif;
            --font-body: 'IBM Plex Sans', sans-serif;
            --font-mono: 'IBM Plex Mono', monospace;
            --shadow-md: 0 10px 28px rgba(2, 6, 16, 0.5);
        }

        html, body, .stApp {
            background: var(--ink);
            color: var(--paper);
            font-family: var(--font-body);
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 1280px;
            padding-top: 1.2rem;
            padding-bottom: 7rem;
        }

        h1, h2, h3 {
            font-family: var(--font-display);
            letter-spacing: -0.02em;
            color: var(--paper);
        }

        h3 {
            font-size: 1.12rem;
            font-weight: 650;
        }

        hr {
            border-color: var(--line);
        }

        /* ---- eyebrows / utility labels ---- */
        .eyebrow, .section-eyebrow {
            font-family: var(--font-mono);
            font-size: 0.72rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--dim);
        }

        .section-eyebrow {
            margin: 0.4rem 0 0.2rem;
        }

        /* ---- hero ---- */
        .hero {
            padding: 0.8rem 0 0.2rem;
        }

        .hero-eyebrow {
            font-family: var(--font-mono);
            font-size: 0.72rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--dim);
        }

        .live-dot {
            display: inline-block;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--radar);
            margin-right: 0.5rem;
            animation: pulse 2.2s ease infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }

        .hero h1 {
            margin: 0.4rem 0 0.3rem;
            font-size: clamp(1.7rem, 3vw, 2.4rem);
            font-weight: 740;
            font-stretch: 115%;
            font-variation-settings: 'wdth' 115;
            line-height: 1.04;
        }

        .hero h1 .accent, .gate-title .accent {
            color: var(--signal);
        }

        .hero p {
            margin: 0;
            color: var(--dim);
            max-width: 60rem;
            font-size: 0.95rem;
        }

        /* ---- skill ticker (signature) ---- */
        .ticker-wrap {
            overflow: hidden;
            border-top: 1px solid var(--line);
            border-bottom: 1px solid var(--line);
            margin: 0.9rem 0 1.0rem;
            -webkit-mask-image: linear-gradient(90deg, transparent, #000 6%, #000 94%, transparent);
            mask-image: linear-gradient(90deg, transparent, #000 6%, #000 94%, transparent);
        }

        .ticker {
            display: inline-block;
            white-space: nowrap;
            padding: 0.5rem 0;
            animation: ticker-scroll 55s linear infinite;
            will-change: transform;
        }

        .tick {
            display: inline-block;
            margin-right: 2.4rem;
            font-family: var(--font-mono);
            font-size: 0.78rem;
            letter-spacing: 0.08em;
        }

        .tick-skill {
            color: var(--paper);
            margin-right: 0.55rem;
        }

        .tick-count {
            color: var(--radar);
        }

        @keyframes ticker-scroll {
            from { transform: translateX(0); }
            to { transform: translateX(-50%); }
        }

        @media (prefers-reduced-motion: reduce) {
            .ticker { animation: none; }
            .live-dot { animation: none; }
        }

        /* ---- gate (locked onboarding) ---- */
        .gate-hero {
            padding: 2.2rem 0 0.4rem;
        }

        .gate-title {
            font-family: var(--font-display);
            font-size: clamp(2.1rem, 4.6vw, 3.3rem);
            font-weight: 780;
            font-stretch: 118%;
            font-variation-settings: 'wdth' 118;
            line-height: 1.02;
            letter-spacing: -0.02em;
            margin: 0.6rem 0 0.8rem;
        }

        .gate-sub {
            color: var(--dim);
            max-width: 46rem;
            font-size: 1.0rem;
            margin: 0;
        }

        .gate-stats {
            display: flex;
            gap: 2.6rem;
            flex-wrap: wrap;
            margin: 1.5rem 0 0.3rem;
        }

        .gate-stat .n {
            display: block;
            font-family: var(--font-mono);
            font-size: 1.45rem;
            color: var(--paper);
        }

        .gate-stat .l {
            font-family: var(--font-mono);
            font-size: 0.7rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--dim);
        }

        .path-label {
            font-family: var(--font-mono);
            font-size: 0.7rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--signal);
            margin: 0.2rem 0 0.1rem;
        }

        .path-title {
            font-family: var(--font-display);
            font-size: 1.15rem;
            font-weight: 650;
            margin-bottom: 0.1rem;
        }

        .path-hint {
            color: var(--dim);
            font-size: 0.86rem;
            margin: 0 0 0.4rem;
        }

        /* ---- profile strip ---- */
        .profile-strip {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.3rem 0.6rem;
            padding: 0.6rem 0.9rem;
            background: var(--panel);
            border: 1px solid var(--line);
            border-left: 3px solid var(--signal);
            border-radius: 12px;
        }

        .profile-flag {
            font-family: var(--font-mono);
            font-size: 0.68rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--signal);
        }

        .profile-meta {
            font-family: var(--font-mono);
            font-size: 0.75rem;
            color: var(--dim);
            margin-right: 0.3rem;
        }

        /* ---- chips ---- */
        .context-chip {
            display: inline-block;
            padding: 0.18rem 0.55rem;
            margin: 0 0.3rem 0.3rem 0;
            border-radius: 999px;
            background: var(--panel-2);
            border: 1px solid var(--line-2);
            color: var(--paper);
            font-family: var(--font-mono);
            font-size: 0.72rem;
        }

        .context-chip.you {
            border-color: rgba(255, 178, 36, 0.5);
            background: var(--signal-soft);
            color: #FFD27A;
        }

        .context-chip.gap-chip {
            border-color: rgba(240, 100, 122, 0.45);
            background: rgba(240, 100, 122, 0.1);
            color: #F8A2B1;
        }

        .hint-copy {
            color: var(--dim);
            font-size: 0.92rem;
        }

        /* ---- metrics ---- */
        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-left: 3px solid var(--radar);
            border-radius: 14px;
            padding: 0.95rem 1.05rem;
        }

        div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
            font-family: var(--font-mono);
            font-size: 0.7rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--dim) !important;
        }

        div[data-testid="stMetricValue"] {
            font-family: var(--font-mono);
        }

        /* ---- tabs: quiet instrument rail ---- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1.6rem;
            border-bottom: 1px solid var(--line);
        }

        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border: none;
            padding: 0.55rem 0.1rem;
        }

        .stTabs [data-baseweb="tab"] p {
            font-family: var(--font-mono);
            font-size: 0.8rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--dim);
        }

        .stTabs [data-baseweb="tab"]:hover p {
            color: var(--paper);
        }

        .stTabs [aria-selected="true"] p {
            color: var(--signal) !important;
            font-weight: 600;
        }

        .stTabs [data-baseweb="tab-highlight"] {
            background: var(--signal);
        }

        .stTabs [data-baseweb="tab-border"] {
            background: var(--line);
        }

        /* ---- sidebar ---- */
        section[data-testid="stSidebar"] {
            background: var(--panel);
            border-right: 1px solid var(--line);
        }

        .side-title {
            font-family: var(--font-display);
            font-size: 1.05rem;
            font-weight: 650;
            margin-bottom: 0.2rem;
        }

        .side-label {
            font-family: var(--font-mono);
            font-size: 0.68rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--dim);
            margin: 1.0rem 0 0.2rem;
        }

        .side-fact {
            font-family: var(--font-mono);
            font-size: 0.74rem;
            color: var(--dim);
            margin: 0.15rem 0;
        }

        /* ---- containers, tables, chat ---- */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--panel);
            border: 1px solid var(--line) !important;
            border-radius: 16px;
            box-shadow: var(--shadow-md);
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 12px;
            overflow: hidden;
        }

        .stChatMessage {
            border: 1px solid var(--line);
            border-radius: 14px;
            background: var(--panel);
        }

        div[data-testid="stChatInput"] textarea {
            border-radius: 12px;
            font-family: var(--font-body);
        }

        .stAlert, div[data-testid="stAlert"] {
            border-radius: 12px;
        }

        /* ---- inputs ---- */
        .stTextArea textarea, .stTextInput input {
            background: var(--panel-2);
            border-radius: 10px;
            font-family: var(--font-body);
        }

        div[data-baseweb="select"] > div {
            background: var(--panel-2);
            border-color: var(--line-2);
            border-radius: 10px;
        }

        span[data-baseweb="tag"] {
            background: var(--signal-soft);
            color: #FFD27A;
            font-family: var(--font-mono);
            border-radius: 8px;
        }

        [data-testid="stFileUploaderDropzone"],
        section[data-testid="stFileUploaderDropzone"] {
            background: var(--panel-2);
            border: 1px dashed var(--line-2);
            border-radius: 12px;
        }

        /* ---- buttons ---- */
        .stButton > button, .stDownloadButton > button {
            font-family: var(--font-mono);
            font-size: 0.82rem;
            letter-spacing: 0.06em;
            border-radius: 10px;
            border: 1px solid var(--line-2);
            background: transparent;
            color: var(--paper);
            transition: border-color 0.15s ease, color 0.15s ease, background 0.15s ease;
        }

        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: var(--radar);
            color: var(--radar);
            background: var(--radar-soft);
        }

        .stButton > button[kind="primary"] {
            background: var(--signal);
            border-color: var(--signal);
            color: #161005;
            font-weight: 600;
        }

        .stButton > button[kind="primary"]:hover {
            background: #FFC14F;
            border-color: #FFC14F;
            color: #161005;
        }

        button:focus-visible {
            outline: 2px solid var(--signal);
            outline-offset: 2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_jobs_dataframe(db_path: str) -> pd.DataFrame:
    columns = [
        "id",
        "title",
        "company",
        "location",
        "salary",
        "description",
        "skills",
        "posted_at",
        "fetched_at",
    ]
    db_file = Path(db_path)
    if not db_file.exists():
        return pd.DataFrame(columns=columns)

    query = (
        "SELECT id, title, company, location, salary, description, skills, posted_at, fetched_at "
        "FROM jobs ORDER BY COALESCE(fetched_at, posted_at) DESC"
    )

    try:
        with sqlite3.connect(db_file) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query).fetchall()
    except sqlite3.Error:
        return pd.DataFrame(columns=columns)

    if not rows:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame([dict(row) for row in rows])
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    return frame[columns]


def normalize_text(value: str | None) -> str:
    return " ".join((value or "").split()).strip()


def parse_skills(record: dict[str, Any]) -> list[str]:
    stored = record.get("skills")
    if isinstance(stored, str) and stored.strip():
        return [skill.strip() for skill in stored.split(",") if skill.strip()]

    description = str(record.get("description") or "")
    nlp, matcher = get_default_matcher()
    return extract_skills(description, nlp=nlp, matcher=matcher)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def compute_skill_annotations(
    titles: tuple[str, ...],
    descriptions: tuple[str, ...],
    stored_skills: tuple[str, ...],
) -> tuple[list[list[str]], list[str]]:
    """Annotate every job with its skill list and domain in one cached pass.

    Skill extraction runs a spaCy PhraseMatcher per record, which is by far the
    most expensive step of frame preparation, so the results are cached on the
    raw text columns instead of being recomputed on every Streamlit rerun.
    """
    nlp, matcher = get_default_matcher()
    skill_lists: list[list[str]] = []
    domains: list[str] = []
    for title, description, stored in zip(titles, descriptions, stored_skills, strict=True):
        if stored.strip():
            skills = [skill.strip() for skill in stored.split(",") if skill.strip()]
        else:
            skills = extract_skills(description, nlp=nlp, matcher=matcher)
        skill_lists.append(skills)
        domains.append(classify_domain(f"{title} {description}", skills))
    return skill_lists, domains


def classify_domain(text: str, skills: list[str]) -> str:
    source = f"{text} {' '.join(skills)}".lower()
    scores: dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword and keyword in source:
                score += 1
        scores[domain] = score
    best_domain, best_score = max(scores.items(), key=lambda item: item[1])
    return best_domain if best_score > 0 else "General Tech"


def normalize_location(location: str | None) -> str:
    value = normalize_text(location)
    if not value:
        return "Unknown"
    first_token = value.split(",")[0].strip()
    return first_token.title()


def prepare_jobs_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    prepared = frame.copy()
    prepared["title"] = prepared["title"].fillna("Untitled Role").astype(str)
    prepared["company"] = prepared["company"].fillna("Unknown Company").astype(str)
    prepared["location"] = prepared["location"].fillna("Unknown").astype(str)
    prepared["salary"] = prepared["salary"].fillna("").astype(str)
    prepared["description"] = prepared["description"].fillna("").astype(str)
    prepared["skills"] = prepared["skills"].fillna("").astype(str)

    prepared["posted_at_dt"] = pd.to_datetime(prepared["posted_at"], utc=True, errors="coerce")
    prepared["fetched_at_dt"] = pd.to_datetime(prepared["fetched_at"], utc=True, errors="coerce")
    prepared["analysis_dt"] = prepared["posted_at_dt"].where(
        prepared["posted_at_dt"].notna(),
        prepared["fetched_at_dt"],
    )
    prepared["age_days"] = (
        pd.Timestamp.now(tz="UTC") - prepared["analysis_dt"]
    ).dt.total_seconds() / 86_400.0
    prepared["age_days"] = prepared["age_days"].round(1)
    prepared["city"] = prepared["location"].apply(normalize_location)
    skill_lists, domains = compute_skill_annotations(
        tuple(prepared["title"].tolist()),
        tuple(prepared["description"].tolist()),
        tuple(prepared["skills"].tolist()),
    )
    prepared["skill_list"] = skill_lists
    prepared["domain"] = domains
    return prepared


def filter_jobs(frame: pd.DataFrame, city: str, domains: list[str], age_range: tuple[int, int]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    filtered = frame.copy()
    if city != "All Cities":
        filtered = filtered[filtered["city"] == city]
    if domains:
        filtered = filtered[filtered["domain"].isin(domains)]
    else:
        return filtered.iloc[0:0]

    min_age, max_age = age_range
    filtered = filtered[filtered["age_days"].between(min_age, max_age, inclusive="both")]
    return filtered.sort_values(by=["analysis_dt", "fetched_at_dt"], ascending=False, na_position="last")


def aggregate_skill_counts(frame: pd.DataFrame) -> Counter[str]:
    counts: Counter[str] = Counter()
    for skills in frame.get("skill_list", []):
        if isinstance(skills, list):
            for skill in skills:
                if skill:
                    counts[skill.strip()] += 1
    return counts


def build_snapshot_context(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "summary_text": "No active listings match the current filters.",
            "record_count": 0,
            "top_skills": [],
            "top_cities": [],
            "top_companies": [],
        }

    skill_counts = aggregate_skill_counts(frame)
    city_counts = frame["city"].value_counts().head(5)
    company_counts = frame["company"].value_counts().head(5)
    record_count = int(len(frame))
    top_skills = skill_counts.most_common(TOP_SKILLS_DISPLAY)
    top_cities = list(city_counts.items())
    top_companies = list(company_counts.items())

    parts = [
        f"Active filtered listings: {record_count}",
        "Top skills: " + ", ".join(f"{skill} ({count})" for skill, count in top_skills) if top_skills else "Top skills: none",
        "Top cities: " + ", ".join(f"{city} ({count})" for city, count in top_cities) if top_cities else "Top cities: none",
        "Top companies: " + ", ".join(f"{company} ({count})" for company, count in top_companies) if top_companies else "Top companies: none",
    ]
    summary_text = " | ".join(parts)
    summary_text = summary_text[:MAX_CONTEXT_CHARACTERS]

    return {
        "summary_text": summary_text,
        "record_count": record_count,
        "top_skills": top_skills,
        "top_cities": top_cities,
        "top_companies": top_companies,
    }


def safe_summary_text(text: str | None) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[redacted email]", text)
    cleaned = re.sub(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b", "[redacted phone]", cleaned)
    cleaned = re.sub(r"https?://\S+", "[redacted url]", cleaned)
    cleaned = re.sub(r"\b\d{10,}\b", "[redacted id]", cleaned)
    return cleaned.strip()


def build_ticker_html(frame: pd.DataFrame) -> str:
    """Render the live skill-demand tape from the full (unfiltered) market frame."""
    if frame.empty:
        return ""
    top_skills = aggregate_skill_counts(frame).most_common(14)
    if not top_skills:
        return ""
    items = "".join(
        f'<span class="tick"><span class="tick-skill">{html.escape(skill.upper())}</span>'
        f'<span class="tick-count">&#9650; {count}</span></span>'
        for skill, count in top_skills
    )
    # The tape is duplicated once so the -50% translate loops seamlessly.
    return f'<div class="ticker-wrap" aria-hidden="true"><div class="ticker">{items}{items}</div></div>'


def render_hero(frame: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow"><span class="live-dot"></span>LIVE FEED &middot; DATA &amp; AI ROLES &middot; INDIA</div>
            <h1>Job Market <span class="accent">Intelligence</span></h1>
            <p>Live listings, skill demand, forecasts, and resume fit — every amber mark on this page traces back to your profile.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    ticker = build_ticker_html(frame)
    if ticker:
        st.markdown(ticker, unsafe_allow_html=True)


def set_candidate_profile(*, source: str, resume_text: str, skills: list[str]) -> None:
    try:
        category_info = predict_resume_category(resume_text)
    except Exception:
        category_info = None
    try:
        quality_info = predict_resume_quality(resume_text) if source == "resume" else None
    except Exception:
        quality_info = None

    st.session_state[SESSION_PROFILE_KEY] = {
        "source": source,
        "resume_text": resume_text,
        "skills": skills,
        "category": category_info,
        "quality": quality_info,
    }
    st.session_state[SESSION_RESUME_TEXT_KEY] = resume_text
    st.session_state.pop(SESSION_RESULTS_KEY, None)


def clear_candidate_profile() -> None:
    for key in (SESSION_PROFILE_KEY, SESSION_RESUME_TEXT_KEY, SESSION_RESUME_UPLOAD_KEY, SESSION_RESULTS_KEY):
        st.session_state.pop(key, None)


def get_profile_skills_lower() -> set[str]:
    profile = st.session_state.get(SESSION_PROFILE_KEY) or {}
    return {str(skill).lower() for skill in profile.get("skills", []) if str(skill).strip()}


def render_onboarding(frame: pd.DataFrame) -> None:
    """Full-screen calibration gate: nothing else renders until a profile exists."""
    st.markdown(
        "<style>section[data-testid='stSidebar'], [data-testid='collapsedControl'], "
        "[data-testid='stSidebarCollapsedControl'] { display: none !important; }</style>",
        unsafe_allow_html=True,
    )

    total_roles = int(len(frame))
    companies = int(frame["company"].nunique()) if not frame.empty else 0
    cities = int(frame["city"].nunique()) if not frame.empty else 0
    freshest = format_freshness(frame)

    st.markdown(
        f"""
        <div class="gate-hero">
            <div class="eyebrow"><span class="live-dot"></span>LIVE FEED &middot; DATA &amp; AI ROLES &middot; INDIA</div>
            <h1 class="gate-title">The market is live.<br/><span class="accent">Tell it who you are.</span></h1>
            <p class="gate-sub">
                Every chart, forecast, and job match in this dashboard tunes itself to your profile.
                Upload a resume or list your skills to unlock it. Resume files are parsed in memory
                and never written to disk.
            </p>
            <div class="gate-stats">
                <div class="gate-stat"><span class="n">{total_roles:,}</span><span class="l">live roles tracked</span></div>
                <div class="gate-stat"><span class="n">{companies:,}</span><span class="l">companies hiring</span></div>
                <div class="gate-stat"><span class="n">{cities:,}</span><span class="l">cities covered</span></div>
                <div class="gate-stat"><span class="n">{freshest}</span><span class="l">freshest posting</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ticker = build_ticker_html(frame)
    if ticker:
        st.markdown(ticker, unsafe_allow_html=True)

    paths = st.columns(2, gap="large")

    with paths[0]:
        with st.container(border=True):
            st.markdown(
                '<div class="path-label">With a resume</div>'
                '<div class="path-title">Upload or paste it</div>'
                '<p class="path-hint">PDF, DOCX, or TXT — your skills are extracted automatically.</p>',
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader(
                "Resume file",
                type=["pdf", "docx", "txt"],
                key="gate_resume_upload",
                label_visibility="collapsed",
            )
            pasted_text = st.text_area(
                "Or paste resume text",
                height=150,
                key="gate_resume_text",
                placeholder="Paste resume bullets, project summaries, or a markdown CV…",
            )
            if st.button("Unlock with resume", type="primary", use_container_width=True):
                resume_text = ""
                if uploaded_file is not None:
                    try:
                        resume_text = extract_text_from_upload(uploaded_file.getvalue(), uploaded_file.name)
                    except Exception as exc:
                        st.error(f"Could not read {uploaded_file.name}: {exc}")
                if not resume_text.strip():
                    resume_text = pasted_text or ""
                if not resume_text.strip():
                    st.warning("Add a resume file or paste its text to continue.")
                else:
                    skills = extract_skills(resume_text)
                    set_candidate_profile(source="resume", resume_text=resume_text.strip(), skills=skills)
                    st.rerun()

    with paths[1]:
        with st.container(border=True):
            st.markdown(
                '<div class="path-label">Without one</div>'
                '<div class="path-title">List your skills</div>'
                '<p class="path-hint">Pick from the market catalog and add anything missing.</p>',
                unsafe_allow_html=True,
            )
            chosen_skills = st.multiselect(
                "Skills you work with",
                options=sorted(SKILL_CATALOG),
                key="gate_skill_select",
                placeholder="Python, SQL, PyTorch…",
            )
            extra_raw = st.text_input(
                "Add skills not in the list (comma-separated)",
                key="gate_skill_extra",
                placeholder="e.g. dbt, Rust, MLflow",
            )
            if st.button("Unlock with skills", type="primary", use_container_width=True):
                extras = [part.strip() for part in extra_raw.split(",") if part.strip()]
                skills = list(dict.fromkeys(chosen_skills + extras))
                if not skills:
                    st.warning("Pick at least one skill to continue.")
                else:
                    resume_text = "Skills: " + ", ".join(skills) + "."
                    set_candidate_profile(source="skills", resume_text=resume_text, skills=skills)
                    st.rerun()

    if frame.empty:
        st.caption("The job warehouse is empty right now. You can still unlock — or seed demo listings first.")
        if st.button("Seed demo data"):
            from ingestion.seed_data import seed_database

            count = seed_database()
            st.cache_data.clear()
            st.success(f"Seeded {count} demo listings. They will appear once you unlock.")


def render_profile_strip() -> None:
    profile = st.session_state.get(SESSION_PROFILE_KEY) or {}
    skills = [str(skill) for skill in profile.get("skills", []) if str(skill).strip()]
    origin = "from your resume" if profile.get("source") == "resume" else "entered manually"
    count_label = f"{len(skills)} skills {origin}" if skills else f"profile {origin} — no catalog skills detected"

    meta_bits = [count_label]
    category_info = profile.get("category")
    if category_info:
        meta_bits.append(f"reads as {category_info['category']} ({category_info['confidence']:.0f}%)")
    quality_info = profile.get("quality")
    if quality_info:
        meta_bits.append(f"AI score {quality_info['ai_score']:.0f}/100")

    chips = "".join(f'<span class="context-chip you">{html.escape(skill)}</span>' for skill in skills[:10])
    if len(skills) > 10:
        chips += f'<span class="context-chip">+{len(skills) - 10} more</span>'

    columns = st.columns([6, 1])
    with columns[0]:
        st.markdown(
            '<div class="profile-strip"><span class="profile-flag">Calibrated</span>'
            f'<span class="profile-meta">{html.escape(" · ".join(meta_bits))}</span>{chips}</div>',
            unsafe_allow_html=True,
        )
    with columns[1]:
        if st.button("Recalibrate", use_container_width=True, help="Clear your profile and start over"):
            clear_candidate_profile()
            st.rerun()


def format_freshness(frame: pd.DataFrame) -> str:
    if frame.empty or "age_days" not in frame.columns:
        return "—"
    ages = pd.to_numeric(frame["age_days"], errors="coerce").dropna()
    if ages.empty:
        return "—"
    freshest = float(ages.min())
    if freshest < 1.0:
        hours = max(0, round(freshest * 24))
        return "now" if hours == 0 else f"{hours}h ago"
    return f"{freshest:.0f}d ago"


def render_metrics(frame: pd.DataFrame) -> None:
    total_listings = int(len(frame))
    unique_companies = int(frame["company"].nunique()) if not frame.empty else 0
    unique_cities = int(frame["city"].nunique()) if not frame.empty else 0
    freshest_label = format_freshness(frame)

    cols = st.columns(4)
    metrics = [
        ("Total Listings", f"{total_listings:,}"),
        ("Unique Companies", f"{unique_companies:,}"),
        ("Cities Covered", f"{unique_cities:,}"),
        ("Freshest Posting", freshest_label),
    ]

    for column, (label, value) in zip(cols, metrics, strict=False):
        with column:
            st.metric(label, value)


def render_skill_chart(frame: pd.DataFrame) -> None:
    st.subheader("Top 20 Most Demanded Skills Across Filtered Listings")
    if frame.empty:
        st.info("No active listings are available for skill analysis. Run the ingestion cycle or adjust the filters.")
        return

    try:
        with st.spinner("Building skill frequency profile..."):
            skill_counts = aggregate_skill_counts(frame)
            if not skill_counts:
                st.info("The current records do not include extractable skills.")
                return

            top_skills = skill_counts.most_common(20)
            skill_names = [skill for skill, _ in reversed(top_skills)]
            volumes = [count for _, count in reversed(top_skills)]
            total_mentions = sum(skill_counts.values()) or 1

            profile_skills = get_profile_skills_lower()
            bar_colors = [
                COLOR_SIGNAL if skill.lower() in profile_skills else COLOR_RADAR
                for skill in skill_names
            ]
            ownership = [
                "on your profile" if skill.lower() in profile_skills else "not on your profile yet"
                for skill in skill_names
            ]

            figure = go.Figure(
                go.Bar(
                    x=volumes,
                    y=skill_names,
                    orientation="h",
                    marker=dict(
                        color=bar_colors,
                        line=dict(color="rgba(233,238,248,0.12)", width=0.5),
                    ),
                    text=[f"{count:,}" for count in volumes],
                    textposition="outside",
                    customdata=[[label] for label in ownership],
                    hovertemplate="%{y}<br>Mentions: %{x}<br>%{customdata[0]}<extra></extra>",
                )
            )
            figure.update_layout(
                template="plotly_dark",
                font=PLOTLY_FONT,
                height=max(560, 30 * len(skill_names) + 220),
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(title="Mention count", gridcolor="rgba(140,154,178,0.12)"),
                yaxis=dict(title="", autorange="reversed"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(figure, use_container_width=True)

            st.caption("Amber bars are skills already on your profile; teal bars are still open gaps.")
            summary_bits = [
                f"{skill} ({count / total_mentions:.1%})" for skill, count in top_skills[:5]
            ]
            st.caption("Top demand signals: " + ", ".join(summary_bits))
    except Exception as exc:
        st.warning(f"Skill chart could not be rendered cleanly: {exc}")


def render_forecast_chart(frame: pd.DataFrame) -> None:
    st.subheader(f"{FORECAST_HORIZON_DAYS}-Day Rolling Demand Trend & OLS Projection")
    if frame.empty:
        st.info("No records are available for trend forecasting.")
        return

    available_skills = sorted(
        {
            skill
            for skill_list in frame["skill_list"]
            if isinstance(skill_list, list)
            for skill in skill_list
            if skill
        }
    )
    if not available_skills:
        st.info("The active records do not expose enough skill signals to build a trend forecast.")
        return

    focus_skill = st.selectbox(
        "Select focus skill",
        options=available_skills,
        index=0,
        help="The forecast uses rolling historical demand and a linear OLS projection from the filtered database.",
    )

    try:
        with st.spinner("Calculating rolling averages and trend projections..."):
            timeseries = build_skill_timeseries(
                frame[["id", "title", "company", "location", "description", "skills", "posted_at", "fetched_at"]].to_dict(
                    orient="records"
                ),
                history_days=HISTORY_WINDOW_DAYS,
                target_skills=[focus_skill],
            )
            if timeseries.empty or focus_skill not in timeseries.columns:
                st.info("No trend series could be assembled for the selected skill.")
                return

            series = timeseries[focus_skill]
            rolling_mean = series.rolling(window=3, min_periods=1).mean()
            forecast = fit_ols_forecast(series)
            forecast_future = forecast[forecast["is_forecast"]].copy()
            history_dates = series.index

            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=history_dates,
                    y=series.values,
                    mode="lines+markers",
                    name="Historical demand",
                    line=dict(color=COLOR_RADAR, width=2.5),
                    marker=dict(size=7),
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=history_dates,
                    y=rolling_mean.values,
                    mode="lines",
                    name="Rolling average",
                    line=dict(color=COLOR_DIM, width=3),
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=forecast_future["date"],
                    y=forecast_future["prediction"],
                    mode="lines",
                    name="OLS projection",
                    line=dict(color=COLOR_SIGNAL, width=3, dash="dash"),
                )
            )

            figure.update_layout(
                template="plotly_dark",
                font=PLOTLY_FONT,
                height=560,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(title="Date", gridcolor="rgba(140,154,178,0.12)"),
                yaxis=dict(title="Job count", gridcolor="rgba(140,154,178,0.12)"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(figure, use_container_width=True)

            slope_estimate = float(forecast_future["prediction"].iloc[-1] - forecast_future["prediction"].iloc[0]) / 6.0
            direction = "upward" if slope_estimate >= 0 else "downward"
            st.caption(
                f"Model insight: {focus_skill} is tracking a {direction} linear trend across the next "
                f"{FORECAST_HORIZON_DAYS} days with an average slope of {slope_estimate:.2f} jobs/day."
            )
    except Exception as exc:
        st.warning(f"Trend chart could not be rendered cleanly: {exc}")


def build_match_status(score_pct: float) -> str:
    if score_pct >= 90:
        return "🟢 Exceptional"
    if score_pct >= FIT_SCORE_SUCCESS_THRESHOLD:
        return "🟢 High Match"
    if score_pct >= FIT_SCORE_WARNING_THRESHOLD:
        return "🟡 Moderate"
    return "🔴 Borderline"


def render_resume_match(frame: pd.DataFrame) -> None:
    st.subheader("Resume fit against the live market")
    st.write(
        "Your profile is already loaded below — edit it freely or upload a newer file, "
        "then score it against the active listings."
    )

    if frame.empty:
        st.info("Resume scoring is unavailable because the filtered dataset is empty.")
        return

    if SESSION_RESUME_TEXT_KEY not in st.session_state:
        profile = st.session_state.get(SESSION_PROFILE_KEY) or {}
        st.session_state[SESSION_RESUME_TEXT_KEY] = profile.get("resume_text") or (
            "Senior data scientist with Python, SQL, machine learning, PyTorch, FastAPI, and RAG experience."
        )

    uploaded_file = st.file_uploader(
        "Upload resume (PDF, DOCX, or TXT) — optional",
        type=["pdf", "docx", "txt"],
        help="Processed in-memory only; the file is never written to disk.",
    )

    if uploaded_file is not None:
        upload_signature = (uploaded_file.name, uploaded_file.size)
        if st.session_state.get(SESSION_RESUME_UPLOAD_KEY) != upload_signature:
            try:
                extracted_text = extract_text_from_upload(uploaded_file.getvalue(), uploaded_file.name)
            except Exception as exc:
                st.warning(f"Could not read uploaded file: {exc}")
            else:
                if extracted_text.strip():
                    st.session_state[SESSION_RESUME_TEXT_KEY] = extracted_text
                    st.session_state[SESSION_RESUME_UPLOAD_KEY] = upload_signature
                    st.success(f"Loaded resume text from {uploaded_file.name}.")
                else:
                    st.warning("No readable text was found in the uploaded file.")

    resume_text = st.text_area(
        "Resume text",
        height=180,
        placeholder="Paste resume bullets, project summaries, or a markdown CV here.",
        key=SESSION_RESUME_TEXT_KEY,
    )

    calculate = st.button("Score My Resume", type="primary")
    if not calculate:
        if SESSION_RESULTS_KEY in st.session_state:
            render_fit_results_table(st.session_state[SESSION_RESULTS_KEY])
            return

        st.caption("Click the button to score the resume against the current filtered job set.")
        return

    st.session_state.pop(SESSION_RESULTS_KEY, None)

    try:
        with st.spinner("Ranking job matches against the resume..."):
            results = score_resume_against_jobs(
                resume_text,
                frame.to_dict(orient="records"),
                top_n=TOP_JOBS_RESUME_DISPLAY,
            )
            category_info = predict_resume_category(resume_text)
            quality_info = predict_resume_quality(resume_text)

        if not results:
            st.info("No ranked matches were produced for the supplied resume and active filters.")
            return

        if category_info is not None:
            st.caption(
                f"\U0001f9e0 Detected resume profile: **{category_info['category']}** "
                f"({category_info['confidence']:.0f}% confidence) from a classifier trained on a Kaggle "
                "resume dataset — matching job titles get a small ranking boost."
            )

        if quality_info is not None:
            st.caption(
                f"\U0001f4ca ML-predicted AI Score: **{quality_info['ai_score']:.0f}/100** | "
                f"Estimated hire likelihood: **{quality_info['hire_likelihood']:.0f}%** "
                "from a model trained on the AI-Powered Resume Screening Dataset 2025."
            )

        result_frame = pd.DataFrame(results)
        result_frame.insert(0, "Rank", [f"#{index:02d}" for index in range(1, len(result_frame) + 1)])
        result_frame["Status"] = result_frame["score_pct"].apply(build_match_status)
        result_frame.rename(columns={"score_pct": "Fit Score", "title": "Job Title"}, inplace=True)
        result_frame["Posted At"] = pd.to_datetime(result_frame["posted_at"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
        result_frame = result_frame[["Rank", "Job Title", "company", "location", "Fit Score", "Status", "Posted At"]].rename(
            columns={"company": "Company", "location": "Location"}
        )

        top_score = float(result_frame.iloc[0]["Fit Score"])
        top_status = build_match_status(top_score)
        alert_text = f"Top fit score: {top_score:.1f}% ({top_status})."
        if get_fit_score_bucket(top_score) == "success":
            st.success(alert_text)
        elif get_fit_score_bucket(top_score) == "warning":
            st.warning(alert_text)
        else:
            st.error(alert_text)

        st.session_state[SESSION_RESULTS_KEY] = result_frame
        render_fit_results_table(result_frame)

        resume_skills = set(extract_skills(resume_text))
        demand_skills = [skill for skill, _ in aggregate_skill_counts(frame).most_common(12)]
        matched_skills = [skill for skill in demand_skills if skill in resume_skills]
        missing_skills = [skill for skill in demand_skills if skill not in resume_skills]

        gap_columns = st.columns(2)
        with gap_columns[0]:
            if matched_skills:
                chips = " ".join(
                    f'<span class="context-chip you">{html.escape(skill)}</span>' for skill in matched_skills[:8]
                )
                st.markdown(f"**✅ In-demand skills you already have**<br/>{chips}", unsafe_allow_html=True)
        with gap_columns[1]:
            if missing_skills:
                chips = " ".join(
                    f'<span class="context-chip gap-chip">{html.escape(skill)}</span>' for skill in missing_skills[:8]
                )
                st.markdown(f"**🚧 Skill gaps vs market demand**<br/>{chips}", unsafe_allow_html=True)
    except Exception as exc:
        st.warning(f"Resume scoring is temporarily unavailable: {exc}")


def infer_domain_options(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return list(DOMAIN_KEYWORDS.keys())
    ordered = []
    for domain in frame["domain"].value_counts().index.tolist():
        if domain not in ordered:
            ordered.append(domain)
    for domain in DOMAIN_KEYWORDS:
        if domain not in ordered:
            ordered.append(domain)
    return ordered


def render_live_listings(frame: pd.DataFrame) -> None:
    st.subheader("Live Listings Table")
    if frame.empty:
        st.info("There are no live listings that satisfy the current city, domain, and freshness filters.")
        return

    st.caption("Apply tab-local filters before reviewing the interactive table below.")

    filter_columns = st.columns([1.1, 1.1, 1.2])
    city_options = ["All Cities"] + sorted(c for c in frame["city"].dropna().unique().tolist() if c)
    domain_options = ["All Domains"] + infer_domain_options(frame)
    with filter_columns[0]:
        city_filter = st.selectbox("City", options=city_options, index=0, key="live_city_filter")
    with filter_columns[1]:
        domain_filter = st.selectbox("Domain", options=domain_options, index=0, key="live_domain_filter")
    with filter_columns[2]:
        age_filter = st.slider(
            "Posting age (days)",
            min_value=0,
            max_value=POSTING_FRESHNESS_MAX_DAYS,
            value=(0, POSTING_FRESHNESS_MAX_DAYS),
            key="live_age_filter",
        )

    display_frame = frame.copy()
    if city_filter != "All Cities":
        display_frame = display_frame[display_frame["city"] == city_filter]
    if domain_filter != "All Domains":
        display_frame = display_frame[display_frame["domain"] == domain_filter]
    display_frame = display_frame[display_frame["age_days"].between(age_filter[0], age_filter[1], inclusive="both")]

    if display_frame.empty:
        st.info("No listings match the current tab-local filters.")
        return

    display_frame["posted_at"] = pd.to_datetime(display_frame["posted_at"], utc=True, errors="coerce")
    display_frame["fetched_at"] = pd.to_datetime(display_frame["fetched_at"], utc=True, errors="coerce")
    display_frame["posted_at"] = display_frame["posted_at"].dt.strftime("%Y-%m-%d %H:%M UTC")
    display_frame["fetched_at"] = display_frame["fetched_at"].dt.strftime("%Y-%m-%d %H:%M UTC")
    display_frame["skill_preview"] = display_frame["skill_list"].apply(lambda skills: ", ".join(skills[:5]) if skills else "")

    table = display_frame[
        [
            "title",
            "company",
            "city",
            "domain",
            "salary",
            "skill_preview",
            "age_days",
            "posted_at",
        ]
    ].rename(
        columns={
            "title": "Job Title",
            "company": "Company",
            "city": "City",
            "domain": "Domain",
            "salary": "Salary",
            "skill_preview": "Skill Preview",
            "age_days": "Age (days)",
            "posted_at": "Posted At",
        }
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Age (days)": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    st.download_button(
        "⬇️ Download these listings (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="filtered_job_listings.csv",
        mime="text/csv",
        help="Export the currently filtered listings to a spreadsheet-friendly CSV.",
    )


def render_sidebar(frame: pd.DataFrame) -> dict[str, Any]:
    with st.sidebar:
        st.markdown('<div class="side-title">Market controls</div>', unsafe_allow_html=True)

        cities = ["All Cities"]
        if not frame.empty:
            cities.extend(sorted(c for c in frame["city"].dropna().unique().tolist() if c))

        st.markdown('<div class="side-label">Geography</div>', unsafe_allow_html=True)
        city_choice = st.selectbox("Geographic filter", options=cities, index=0, label_visibility="collapsed")

        st.markdown('<div class="side-label">Domains</div>', unsafe_allow_html=True)
        domain_options = infer_domain_options(frame)
        default_domains = set(domain_options)
        selected_domains: list[str] = []
        domain_columns = st.columns(2)
        for index, domain in enumerate(domain_options):
            checkbox_key = f"job_market_domain_{re.sub(r'[^a-z0-9]+', '_', domain.lower()).strip('_')}"
            default_value = st.session_state.get(checkbox_key, domain in default_domains)
            with domain_columns[index % 2]:
                if st.checkbox(domain, value=default_value, key=checkbox_key):
                    selected_domains.append(domain)

        st.markdown('<div class="side-label">Posting freshness</div>', unsafe_allow_html=True)
        if frame.empty:
            age_max_default = 30
        else:
            max_age = math.ceil(
                pd.to_numeric(frame["age_days"], errors="coerce")
                .fillna(POSTING_FRESHNESS_MAX_DAYS)
                .clip(upper=POSTING_FRESHNESS_MAX_DAYS)
                .max()
            )
            age_max_default = max(1, min(POSTING_FRESHNESS_MAX_DAYS, max_age))
        age_range = st.slider(
            "Posting freshness (days)",
            min_value=0,
            max_value=POSTING_FRESHNESS_MAX_DAYS,
            value=(0, age_max_default),
            label_visibility="collapsed",
        )
        st.caption(f"Filter: Last {age_range[1]} Days")

        st.markdown("---")
        st.markdown('<div class="side-label">System</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="side-fact">DB refresh &middot; every 4 hours</div>'
            '<div class="side-fact">Infra cost &middot; &#8377;0 (free)</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        if st.button("Seed demo data", help="Populate the database with synthetic listings for a demo"):
            from ingestion.seed_data import seed_database

            count = seed_database()
            st.cache_data.clear()
            st.success(f"Seeded {count} demo records. Refresh the page.")

        if st.button("Clear chat history", help="Reset the market analyst conversation"):
            st.session_state.pop(SESSION_MESSAGES_KEY, None)
            st.session_state.pop(SESSION_CONTEXT_KEY, None)
            st.rerun()

        if st.button("Recalibrate profile", help="Clear your resume/skills profile and return to the gate"):
            clear_candidate_profile()
            st.rerun()

    return {
        "city": city_choice,
        "domains": selected_domains,
        "age_range": age_range,
        "control_signature": "|".join([city_choice, ",".join(selected_domains), f"{age_range[0]}-{age_range[1]}"]),
    }


def render_context_summary(frame: pd.DataFrame) -> None:
    context = build_snapshot_context(frame)
    st.markdown("#### Context Snapshot")
    if context["record_count"] == 0:
        st.info("No active listings match the current filters, so the chat context is empty.")
        return

    st.caption(context["summary_text"])
    for label, values in (
        ("Top skills", context["top_skills"]),
        ("Top cities", context["top_cities"]),
        ("Top companies", context["top_companies"]),
    ):
        chips = " ".join(
            f'<span class="context-chip">{html.escape(str(item[0]))} ({item[1]})</span>' for item in values[:5]
        )
        st.markdown(f"**{label}**<br/>{chips}", unsafe_allow_html=True)


def build_system_prompt() -> str:
    return (
        "You are the GenAI Market Analyst for a real-time job intelligence dashboard. "
        "Answer only from the supplied context snapshot and recent conversation. "
        "Do not reveal, infer, or ask for personal identity data, emails, phone numbers, addresses, or resumes. "
        "Keep answers concise, actionable, and tied to hiring signals, skills, cities, companies, or upskilling advice. "
        "If the answer is uncertain, say so and provide the safest local summary."
    )


def build_local_fallback_answer(question: str, context: dict[str, Any]) -> str:
    q = question.lower()
    skills = [skill for skill, _ in context.get("top_skills", [])[:5]]
    cities = [city for city, _ in context.get("top_cities", [])[:5]]
    companies = [company for company, _ in context.get("top_companies", [])[:5]]

    skill_text = ", ".join(skills) if skills else "the available technical stack"
    city_text = ", ".join(cities) if cities else "the current market geography"
    company_text = ", ".join(companies) if companies else "the active hiring entities"

    if any(token in q for token in ("skill", "gap", "missing", "resume", "qualif")):
        return (
            f"Based on the filtered listings, the strongest skill signals are {skill_text}. "
            "A practical move is to foreground those keywords in projects, add measurable outcomes, "
            "and keep your resume aligned to the top 3 demands from the job pool."
        )
    if any(token in q for token in ("city", "region", "location", "metro")):
        return (
            f"The current demand is clustered around {city_text}. "
            "If you are tailoring a search, prioritize location-specific keywords and hybrid/remote tags where present."
        )
    if any(token in q for token in ("company", "employer", "brand")):
        return (
            f"The most active companies in this slice are {company_text}. "
            "You can tailor applications by mirroring the skill combinations that repeat in those employer listings."
        )
    if any(token in q for token in ("trend", "forecast", "hiring", "demand")):
        return (
            f"The filtered dataset suggests demand is concentrated in {skill_text}, especially across {city_text}. "
            "Use that signal to prioritize targeted upskilling, resume wording, and job search focus."
        )

    return (
        f"{LOCAL_SUMMARY_INTRO} The current snapshot highlights {skill_text} across {city_text}, "
        f"with repeat activity from {company_text}. Ask me about skill gaps, regional demand, or resume tuning."
    )


def call_market_chat(
    *,
    question: str,
    context: dict[str, Any],
    history: list[dict[str, str]],
) -> tuple[str, str]:
    chat_api_key_name = str(CONFIG.get("secrets", {}).get("chat_api_key_name", ""))
    credential_value = get_credential(chat_api_key_name, api_file_path=ROOT_DIR / "API.txt") or get_credential(
        chat_api_key_name, api_file_path=ROOT_DIR / "api.txt"
    )
    compressed_context = context["summary_text"]
    safe_question = safe_summary_text(question)
    trimmed_history = history[-MAX_CHAT_TURNS * 2 :]

    if not credential_value:
        return build_local_fallback_answer(safe_question, context), "local fallback"

    messages: list[dict[str, str]] = [
        {"role": "system", "content": build_system_prompt()},
        {
            "role": "system",
            "content": (
                "Context Summary: "
                f"{compressed_context}\n"
                "Use this snapshot to answer market intelligence questions. "
                "Do not exceed the provided context or invent unsupported counts."
            ),
        },
    ]
    messages.extend(trimmed_history)
    messages.append({"role": "user", "content": safe_question})

    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 450,
        "top_p": 1.0,
        "stream": False,
    }

    try:
        response = requests.post(
            CHAT_ENDPOINT,
            headers={
                "Authorization": f"Bearer {credential_value}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        if not content:
            raise ValueError("Groq returned an empty completion.")
        return content, "groq"
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return build_local_fallback_answer(safe_question, context), "local fallback"


def render_chat_footer(frame: pd.DataFrame, active_city: str) -> None:
    st.divider()
    st.markdown('<div class="section-eyebrow">GenAI &middot; Market analyst</div>', unsafe_allow_html=True)
    st.markdown("### Ask the analyst")

    context = build_snapshot_context(frame)
    st.session_state[SESSION_CONTEXT_KEY] = context
    if SESSION_MESSAGES_KEY not in st.session_state:
        st.session_state[SESSION_MESSAGES_KEY] = [
            {
                "role": "assistant",
                "content": (
                    "Ask me about regional skill gaps, hiring trends, company concentration, "
                    "resume improvements, or upskilling priorities."
                ),
            }
        ]

    with st.container(border=True):
        active_region = "All" if active_city == "All Cities" else active_city
        st.caption(
            f"Context transmitted: Snapshot of {context['record_count']:,} listings | "
            f"Top skills: {', '.join(skill for skill, _ in context['top_skills'][:5]) or 'None'} | "
            f"Top companies: {', '.join(company for company, _ in context['top_companies'][:3]) or 'None'} | "
            f"Active region: {active_region}"
        )

        for message in st.session_state[SESSION_MESSAGES_KEY]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    user_prompt = st.chat_input("Ask about salary benchmarks, missing skills, or hiring trends.")
    if not user_prompt:
        return

    safe_prompt = safe_summary_text(user_prompt)
    prior_history = list(st.session_state[SESSION_MESSAGES_KEY])
    st.session_state[SESSION_MESSAGES_KEY].append({"role": "user", "content": safe_prompt})

    with st.chat_message("user"):
        st.markdown(safe_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating the market analyst response..."):
            answer, source = call_market_chat(
                question=safe_prompt,
                context=context,
                history=prior_history,
            )
        st.markdown(answer)
        st.caption(f"Response source: {source}")

    st.session_state[SESSION_MESSAGES_KEY].append({"role": "assistant", "content": answer})
    if len(st.session_state[SESSION_MESSAGES_KEY]) > MAX_CHAT_TURNS * 2 + 1:
        first_message = st.session_state[SESSION_MESSAGES_KEY][0]
        st.session_state[SESSION_MESSAGES_KEY] = [first_message] + st.session_state[SESSION_MESSAGES_KEY][-(MAX_CHAT_TURNS * 2) :]


def main() -> None:
    configure_page()
    inject_styles()

    try:
        raw_frame = load_jobs_dataframe(str(DB_PATH))
    except Exception as exc:
        st.error(f"The dashboard could not read the SQLite warehouse: {exc}")
        return

    frame = prepare_jobs_frame(raw_frame)

    # The calibration gate: every feature stays locked until the user uploads a
    # resume or lists their skills. Nothing below this point renders without a profile.
    if SESSION_PROFILE_KEY not in st.session_state:
        render_onboarding(frame)
        return

    sidebar_state = render_sidebar(frame)

    control_signature = sidebar_state["control_signature"]
    previous_control_signature = st.session_state.get(SESSION_CONTROL_KEY)
    if previous_control_signature != control_signature:
        st.session_state[SESSION_CONTROL_KEY] = control_signature
        if previous_control_signature is not None:
            load_jobs_dataframe.clear()
            st.rerun()

    filtered_frame = filter_jobs(frame, sidebar_state["city"], sidebar_state["domains"], sidebar_state["age_range"])

    freshest_marker = ""
    if not filtered_frame.empty and "fetched_at_dt" in filtered_frame.columns:
        freshest_ts = filtered_frame["fetched_at_dt"].max()
        if pd.notna(freshest_ts):
            freshest_marker = pd.Timestamp(freshest_ts).isoformat()

    filter_signature = "|".join(
        [
            sidebar_state["city"],
            ",".join(sorted(sidebar_state["domains"])),
            f"{sidebar_state['age_range'][0]}-{sidebar_state['age_range'][1]}",
            str(len(filtered_frame)),
            freshest_marker,
        ]
    )
    if st.session_state.get(SESSION_FILTER_KEY) != filter_signature:
        st.session_state[SESSION_FILTER_KEY] = filter_signature
        st.session_state.pop(SESSION_MESSAGES_KEY, None)
        st.session_state.pop(SESSION_CONTEXT_KEY, None)

    render_hero(frame)
    render_profile_strip()
    render_metrics(filtered_frame)

    tabs = st.tabs(
        [
            "Skill Demand",
            "Forecasts",
            "Resume Fit",
            "Live Listings",
        ]
    )

    with tabs[0]:
        render_skill_chart(filtered_frame)

    with tabs[1]:
        render_forecast_chart(filtered_frame)

    with tabs[2]:
        render_resume_match(filtered_frame)

    with tabs[3]:
        render_live_listings(filtered_frame)

    render_chat_footer(filtered_frame, sidebar_state["city"])


if __name__ == "__main__":
    main()
