# Business Requirements Document
## Real-Time Job Market Intelligence Dashboard

**Version:** 1.0  
**Date:** June 2026  
**Status:** Draft  

---

## 1. Executive Summary

This document outlines the business requirements for a **Real-Time Job Market Intelligence Dashboard** — a full-stack, ML-powered web application that aggregates live job listings, extracts skill demand signals, clusters roles, scores resume fit, and surfaces insights via a natural language chat interface.

The product is targeted at data science / AI job seekers in India and serves as a portfolio project demonstrating end-to-end data engineering, NLP, machine learning, and GenAI capabilities.

---

## 2. Business Context & Objectives

### 2.1 Problem Statement

Job seekers in the data science and AI space lack a single, real-time view of:
- Which skills are in demand across companies and cities
- How demand for specific skills is trending week-over-week
- How well their resume matches current openings
- Where the best-fit opportunities are concentrated geographically

### 2.2 Business Objectives

| # | Objective |
|---|-----------|
| BO-1 | Aggregate live job listings from multiple sources (Indeed, LinkedIn, Google for Jobs) into a single queryable database |
| BO-2 | Surface skill demand trends to help candidates prioritise upskilling efforts |
| BO-3 | Provide a personalised resume fit score against live openings |
| BO-4 | Enable natural language querying of job market data via an LLM interface |
| BO-5 | Deploy as a publicly accessible, zero-cost web application suitable for live interview demos |

### 2.3 Success Metrics

| Metric | Target |
|--------|--------|
| Job listings in DB at launch | ≥ 500 records |
| Data freshness | Refreshed every 4 hours |
| Resume fit scoring latency | < 3 seconds |
| LLM chat response time | < 5 seconds |
| Infrastructure cost | ₹0 (free-tier only) |

---

## 3. Scope

### 3.1 In Scope

- Automated data ingestion pipeline from JSearch API and Indeed Scraper
- SQLite storage with deduplication and normalisation
- ML layer: skill extraction (NLP), role clustering (K-Means), resume fit scoring (TF-IDF), demand trend forecasting (linear regression)
- Streamlit dashboard with four core views
- Groq-powered LLM chat interface
- Deployment to Streamlit Cloud

### 3.2 Out of Scope

- User authentication and saved profiles
- Mobile native application
- Paid job listing integrations
- Email notification system (noted as a future enhancement)
- Naukri direct API integration (manual scraping carries ToS risk)

---

## 4. Stakeholders

| Role | Responsibility |
|------|----------------|
| Product Owner (Developer) | Builds and maintains the system; primary user |
| End Users | Data science / AI job seekers in India |
| Recruiters / Interviewers | Audience for live demo during placement interviews |

---

## 5. Functional Requirements

### 5.1 Data Ingestion

| ID | Requirement |
|----|-------------|
| FR-1 | The system shall fetch job listings from the JSearch API (via RapidAPI) using configurable query terms and locations |
| FR-2 | The system shall fetch supplementary listings from the Indeed Scraper (omkarcloud) for India-specific data |
| FR-3 | The ingestion scheduler shall run automatically every 4 hours using APScheduler |
| FR-4 | The system shall support at least 4 concurrent job search queries (e.g. "data scientist, Delhi"; "ML engineer, Bangalore"; "AI engineer, remote"; "data analyst, Noida") |
| FR-5 | The system shall deduplicate records by `job_id` before persisting to the database |
| FR-6 | Each job record shall store: `id`, `title`, `company`, `location`, `salary`, `description` (up to 2,000 chars), `skills`, `posted_at`, `fetched_at` |

### 5.2 Data Storage & Cleaning

| ID | Requirement |
|----|-------------|
| FR-7 | The system shall persist all job data to a local SQLite database (`data/jobs.db`) |
| FR-8 | The cleaner module shall normalise city name variants to canonical forms (e.g. "NCR", "New Delhi" → "Delhi") |
| FR-9 | The cleaner module shall parse and normalise salary strings into a consistent numeric or range format |
| FR-10 | The cleaner module shall strip HTML tags and extraneous whitespace from job descriptions |

### 5.3 ML & NLP Layer

| ID | Requirement |
|----|-------------|
| FR-11 | The system shall extract skill keywords from job descriptions using spaCy and a curated skill keyword list |
| FR-12 | The skill extraction module shall support a configurable keyword list covering at minimum: Python, SQL, TensorFlow, PyTorch, Scikit-learn, LangChain, Docker, MLOps, LLM, RAG, XGBoost, NLP, Pandas, Power BI, Tableau, Spark, Kafka, FastAPI |
| FR-13 | The system shall cluster job descriptions into 3–7 role segments using K-Means on TF-IDF vectors |
| FR-14 | The system shall compute a cosine similarity fit score between a user-supplied resume and each job description using TF-IDF |
| FR-15 | The system shall forecast skill demand for the next 7 days using a linear regression model trained on daily job count time series |

### 5.4 Dashboard — Core Views

| ID | Requirement |
|----|-------------|
| FR-16 | **Skill Heatmap:** The dashboard shall display a bar chart of the top 20 most demanded skills across all fetched listings, filterable by city, domain, and date range |
| FR-17 | **Trending Skills Chart:** The dashboard shall display a line chart showing the 7-day rolling demand trend per skill, with a 7-day linear forecast |
| FR-18 | **Resume Fit Scorer:** The dashboard shall accept free-text resume input and return a ranked list of the top 20 matching jobs with their fit scores (0–100%) |
| FR-19 | **Live Job Listings Table:** The dashboard shall display a sortable, filterable table of all current listings showing title, company, location, and fit score (when a resume is provided) |
| FR-20 | The dashboard shall expose sidebar filters for: City, Domain (multi-select), and "Posted in last N days" (slider, 1–30) |
| FR-21 | The dashboard shall display three KPI metrics at the top: total listings, unique companies, cities covered |
| FR-22 | Dashboard data shall be cached with a 1-hour TTL (`st.cache_data`) to minimise redundant DB reads |

### 5.5 GenAI Chat Interface

| ID | Requirement |
|----|-------------|
| FR-23 | The dashboard shall include a natural language chat input that allows users to ask questions about the current job market snapshot |
| FR-24 | The LLM context shall include: top 10 skills, top 5 cities, top 5 hiring companies derived from the current filtered dataset |
| FR-25 | The system shall use the Groq API (`llama-3.3-70b-versatile`) as the LLM backend |
| FR-26 | The chat interface shall respond to questions including (but not limited to): trending skills by city, resume gap analysis, top hiring companies, salary benchmarks, and skill recommendations |

### 5.6 Deployment

| ID | Requirement |
|----|-------------|
| FR-27 | The application shall be deployable to Streamlit Cloud on the free tier |
| FR-28 | All API keys shall be managed as Streamlit secrets (not hardcoded) |
| FR-29 | The project shall include a `requirements.txt` listing all dependencies |
| FR-30 | The repository shall include a README with setup instructions and a demo screenshot |

---

## 6. Non-Functional Requirements

| ID | Category | Requirement |
|----|----------|-------------|
| NFR-1 | Performance | Dashboard initial load time < 5 seconds on a standard internet connection |
| NFR-2 | Reliability | Ingestion failures (e.g. API rate limit hit) shall be logged and retried on the next scheduled run without crashing the scheduler |
| NFR-3 | Scalability | The SQLite schema shall support up to 100,000 job records without query degradation |
| NFR-4 | Cost | Total monthly infrastructure cost shall be ₹0; all external services shall operate within free-tier limits |
| NFR-5 | Security | No PII shall be stored; resume text entered by users shall not be persisted to disk |
| NFR-6 | Maintainability | Skill keyword list shall be externalised to a config file, not hardcoded |
| NFR-7 | Portability | The application shall run locally on any machine with Python 3.10+ without OS-specific dependencies |

---

## 7. System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   Data Sources   │────▶│  Ingestion Layer  │────▶│      ML Layer        │
│                 │     │                  │     │                     │
│ • JSearch API   │     │ • fetcher.py     │     │ • skill_extractor.py│
│ • Indeed Scraper│     │ • cleaner.py     │     │ • clustering.py     │
│ • Indeed MCP    │     │ • scheduler.py   │     │ • trend_forecast.py │
│   (prototype)   │     │ • SQLite DB      │     │ • resume scorer     │
└─────────────────┘     └──────────────────┘     └──────────┬──────────┘
                                                             │
                                                             ▼
                                                  ┌─────────────────────┐
                                                  │   Streamlit Dashboard│
                                                  │                     │
                                                  │ • Skill heatmap     │
                                                  │ • Trend charts      │
                                                  │ • Resume scorer     │
                                                  │ • Listings table    │
                                                  │ • Groq LLM chat     │
                                                  └─────────────────────┘
```

---

## 8. Data Sources

| Source | Purpose | Free Tier Limit | Notes |
|--------|---------|-----------------|-------|
| JSearch API (RapidAPI) | Primary — real-time listings from Google for Jobs, LinkedIn, Indeed | 200 req/month | 30+ data points per listing |
| Indeed Scraper (omkarcloud) | Secondary — India-specific listings | 5,000 req/month | Open-source REST wrapper |
| Indeed MCP (Claude session) | Prototyping and schema validation only | Session-based | Not for production use |

**Ingestion budget:** 200–300 API calls/day across sources; fetch once every 4–6 hours for 4–5 job categories.

---

## 9. Technology Stack

| Layer | Technology | Cost |
|-------|-----------|------|
| Language | Python 3.10+ | Free |
| ML / NLP | Scikit-learn, spaCy | Free |
| Data processing | Pandas, NumPy | Free |
| Storage | SQLite | Free |
| Scheduling | APScheduler | Free |
| Dashboard | Streamlit | Free |
| Visualisation | Plotly | Free |
| LLM | Groq API (Llama 3.3 70B) | Free (14,400 tokens/min) |
| Hosting | Streamlit Cloud | Free |
| **Total** | | **₹0** |

---

## 10. Project Folder Structure

```
job-market-dashboard/
├── data/
│   └── jobs.db               # SQLite database
├── ingestion/
│   ├── fetcher.py             # API calls — JSearch / Indeed
│   ├── cleaner.py             # Dedup, normalise salaries, parse dates
│   └── scheduler.py          # APScheduler — runs every 4 hours
├── ml/
│   ├── skill_extractor.py
│   ├── clustering.py
│   └── trend_forecast.py
├── dashboard/
│   └── app.py                 # Streamlit application
└── requirements.txt
```

---

## 11. Build Timeline

| Phase | Days | Deliverables |
|-------|------|-------------|
| **Week 1 — Foundation** | 1–7 | API registration, `fetcher.py`, `cleaner.py`, `scheduler.py`; DB with 500+ rows |
| **Week 2 — ML Pipeline** | 8–14 | `skill_extractor.py`, K-Means clustering, TF-IDF resume scorer, demand trend forecast |
| **Week 3 — Dashboard & Deploy** | 15–21 | Streamlit dashboard (all 4 views), Groq LLM chat, GitHub push, Streamlit Cloud deploy, README + demo video |

---

## 12. Future Enhancements

| Enhancement | Priority | Notes |
|-------------|----------|-------|
| Email alerts for new role matches | Medium | Notify user when new jobs match their profile |
| GitHub skill gap analyser | Medium | Compare user's GitHub repo skills vs job demand |
| Resume auto-tailor with LLM | High | Suggest resume edits to improve fit score for target roles |
| CGPA cutoff filter | Low | Filter listings by stated CGPA requirements |
| Fresher vs experienced toggle | Low | Segment listings by experience level |
| Naukri integration | Low | Requires careful ToS review |

---

## 13. Assumptions & Constraints

- All external APIs remain available on their current free-tier terms
- The application is a portfolio/demo project; it is not intended for commercial use
- Resume text entered by users is processed in-memory only and is never written to disk or transmitted to third parties beyond the LLM API call
- API keys are treated as secrets and are never committed to version control
- The SQLite database is appropriate for single-user or demo use; multi-user production deployment would require migration to PostgreSQL

---

## 14. Glossary

| Term | Definition |
|------|-----------|
| TF-IDF | Term Frequency–Inverse Document Frequency; a numerical statistic used to reflect the importance of a word in a document corpus |
| K-Means | An unsupervised ML clustering algorithm that partitions data into K groups |
| spaCy | An open-source NLP library used here for job description parsing |
| Groq | An LLM inference API provider; free tier supports Llama 3.3 70B |
| Streamlit | A Python framework for building and deploying data web apps |
| APScheduler | Advanced Python Scheduler; used to run the ingestion pipeline on a recurring interval |
| JSearch | A RapidAPI-hosted job search aggregator that indexes Google for Jobs results |
| Fit Score | Cosine similarity (0–100%) between a user's resume and a job description, computed via TF-IDF |

---

*End of Document*
