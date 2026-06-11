# Real-Time Job Market Intelligence Dashboard

> Live job intelligence for data science, ML, and AI roles in India — skill demand analytics, resume matching, trend forecasting, and a Groq-powered market analyst chat, all in one Streamlit app.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Visuals-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SQLite](https://img.shields.io/badge/SQLite-Local%20DB-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org/)
[![Groq](https://img.shields.io/badge/Groq-LLM%20Chat-111827?style=for-the-badge)](https://groq.com/)

## Table of Contents

- [What It Does](#what-it-does)
- [Core Features](#core-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [How the ML Works](#how-the-ml-works)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Running Tests](#running-tests)
- [Deployment](#deployment)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)

## What It Does

This project turns live job postings into a recruiter-friendly analytics dashboard for data science, ML, and AI roles across Delhi/NCR, Bangalore, and remote India-based positions. It answers four questions at a glance:

- **What skills are in demand right now**, and how does that break down by domain (GenAI, ML, Data Engineering, Analytics, Backend)?
- **Which skills are trending up or down** over the past week, with a short-term forecast?
- **How well does a candidate's resume match the current job market**, and which open roles are the best fit?
- **What should a candidate do next** — answered conversationally by an LLM grounded in the live, filtered job snapshot?

## Core Features

| Feature | Description |
|---|---|
| 📊 Skill Demand Heatmap | Top 20 in-demand skills across the currently filtered listings, extracted via a curated NLP keyword catalog and visualized with Plotly. |
| 📈 Trend Forecasting | Rolling 7-day history plus an OLS-based forecast that projects whether demand for a skill is rising or falling. |
| 📄 Resume Fit Scoring | TF-IDF + cosine similarity ranks live job postings against an uploaded resume (PDF/DOCX/text) and surfaces the best matches. |
| 🧠 Resume Quality & Category Models | Trained classifiers estimate resume quality and predict the most likely job category for a resume. |
| 💬 GenAI Market Analyst Chat | Compresses the current filtered job snapshot into context and answers candidate questions via Groq's Llama-3.3-70B. |
| 🔄 Scheduled Ingestion | APScheduler-driven pipeline refreshes job listings from JSearch and job-board scrapers every few hours. |

## Architecture

```text
┌───────────────────┐     ┌───────────────────┐     ┌────────────────────┐
│   Data Sources     │────▶│  Ingestion Layer   │────▶│   ML / SQLite       │
│                    │     │                    │     │                    │
│ • JSearch API      │     │ • fetcher.py       │     │ • skill_extractor  │
│ • python-jobspy    │     │ • cleaner.py       │     │ • clustering       │
│   (Indeed, etc.)   │     │ • scheduler.py     │     │ • trend_forecast   │
└───────────────────┘     └───────────────────┘     │ • resume_scorer    │
                                                       └────────────────────┘
                                                                │
                                                                ▼
                                                      ┌────────────────────┐
                                                      │   Streamlit App     │
                                                      │                    │
                                                      │ • Skill heatmap     │
                                                      │ • Trend chart       │
                                                      │ • Resume scorer     │
                                                      │ • Listings table    │
                                                      │ • Groq chat (Llama) │
                                                      └────────────────────┘
```

## Tech Stack

| Layer | Tool | Purpose | Cost |
|---|---|---|---|
| Ingestion | JSearch API, python-jobspy, APScheduler | Pull new listings on a schedule | Free-tier |
| Storage | SQLite | Persist cleaned job records locally | Free |
| NLP | spaCy, curated skill keyword catalog | Extract skills from job descriptions | Free |
| ML | scikit-learn, TF-IDF, K-Means, OLS regression | Score, cluster, and forecast job data | Free |
| Visualization | Streamlit, Plotly | Interactive dashboard UI | Free |
| GenAI | Groq Llama-3.3-70B | Conversational market analyst grounded in live data | Free-tier |
| CI/CD | GitHub Actions | Run tests and refresh analytics on schedule | Free |

## How the ML Works

### TF-IDF
TF-IDF gives each word a weight based on how often it appears in one job description compared with the full dataset. It helps the app find the words that really matter instead of treating every word as equally important.

### K-Means
K-Means groups similar job descriptions into role clusters, such as analytics-heavy roles versus ML-heavy roles, so recruiters can quickly see how the market is splitting into different job families.

### Cosine Similarity
Cosine similarity measures how close two text vectors are by comparing direction rather than raw word counts, making it well suited to matching a resume against a job description even when the wording differs.

### Linear Regression (OLS)
A best-fit line through the recent history of job counts per skill is extended into the future, giving a simple, explainable forecast of whether demand for that skill is rising or falling.

## Project Structure

```text
.
├── config/             # YAML settings + config loader
├── dashboard/          # Streamlit app (app.py)
├── data/               # SQLite database and resume datasets (gitignored)
├── ingestion/          # Job fetching, cleaning, and scheduling
├── ml/                 # Skill extraction, clustering, forecasting, resume scoring/classifiers
├── tests/              # Integration and smoke tests
├── .streamlit/         # Streamlit configuration
└── requirements.txt    # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/) (for the chat feature)
- A [JSearch API key](https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch) (for live ingestion)

### Installation

```bash
git clone https://github.com/vishvchhabra-21/Real-Time-Job-Market-Intelligence-Dashboard.git
cd Real-Time-Job-Market-Intelligence-Dashboard

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate      # macOS/Linux

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Run the Dashboard

```bash
streamlit run dashboard/app.py
```

The app reads job data from `data/jobs.db`. To populate it with live listings, run the ingestion pipeline (see `ingestion/scheduler.py` and `ingestion/seed_data.py`).

## Configuration

App behavior (ingestion queries, ML parameters, dashboard display limits) is controlled by [`config/settings.yaml`](config/settings.yaml).

Secrets are read via environment variables or `.streamlit/secrets.toml` (never committed):

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Powers the GenAI market analyst chat |
| `JSEARCH_KEY` | Powers live job ingestion from the JSearch API |

## Running Tests

```bash
pytest
```

Integration and smoke tests live in [`tests/`](tests/) and validate the ingestion pipeline, ML models, and end-to-end data flow.

## Deployment

1. Push the repository to GitHub.
2. Create a Streamlit Cloud app and set the main file path to `dashboard/app.py`.
3. Add `GROQ_API_KEY` and `JSEARCH_KEY` to Streamlit secrets, then deploy.

## Future Enhancements

- Email alerts for new role matches.
- GitHub skill-gap analyzer comparing portfolio repos against live demand.
- Resume auto-tailor with LLM suggestions for target roles.
- CGPA cutoff filter for fresher-focused shortlisting.
- Fresher vs. experienced toggle for segmenting listings by candidate level.

## Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for branch naming, PR conventions, and guidance on adding new skills or job sources.
