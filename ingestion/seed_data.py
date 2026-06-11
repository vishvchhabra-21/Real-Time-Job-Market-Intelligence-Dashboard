from __future__ import annotations

import random
import sqlite3
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DB_PATH = ROOT_DIR / "data" / "jobs.db"

SKILL_KEYWORDS = [
    "Python",
    "SQL",
    "TensorFlow",
    "PyTorch",
    "Scikit-learn",
    "LangChain",
    "Docker",
    "MLOps",
    "LLM",
    "RAG",
    "XGBoost",
    "NLP",
    "Pandas",
    "Power BI",
    "Tableau",
    "Spark",
    "FastAPI",
    "NumPy",
    "Airflow",
    "AWS",
    "Kubernetes",
    "Snowflake",
    "Databricks",
    "Hugging Face",
    "Computer Vision",
    "Streamlit",
    "Excel",
    "Statistics",
    "Time Series",
    "Generative AI",
]

JOB_TITLES = [
    "Data Scientist",
    "ML Engineer",
    "Senior Data Scientist",
    "AI Engineer",
    "Data Analyst",
    "NLP Engineer",
    "MLOps Engineer",
    "LLM Engineer",
    "Business Intelligence Analyst",
    "Deep Learning Engineer",
    "Data Engineer",
    "Generative AI Engineer",
    "Computer Vision Engineer",
    "Business Analyst",
    "Analytics Engineer",
    "Research Scientist - AI",
    "Applied Scientist",
    "Decision Scientist",
]

COMPANIES = [
    "Flipkart",
    "Paytm",
    "Razorpay",
    "Groww",
    "Ola",
    "Zomato",
    "CRED",
    "PhonePe",
    "Meesho",
    "Swiggy",
    "Infosys",
    "TCS",
    "Wipro",
    "HCL",
    "Amazon India",
    "Google India",
    "Microsoft India",
    "IBM India",
    "Zerodha",
    "Freshworks",
    "Postman",
    "Sprinklr",
    "Fractal Analytics",
    "Mu Sigma",
    "Tiger Analytics",
    "InMobi",
]

LOCATION_COUNTS = {
    "Delhi": 35,
    "Bangalore": 30,
    "Remote": 25,
    "Noida": 20,
    "Gurgaon": 18,
    "Hyderabad": 12,
    "Mumbai": 10,
    "Pune": 10,
}

SALARY_RANGES = [
    "8-15 LPA",
    "10-18 LPA",
    "12-22 LPA",
    "15-25 LPA",
    "18-30 LPA",
    "22-35 LPA",
    "25-40 LPA",
    "35-55 LPA",
]

CREATE_JOBS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    company     TEXT,
    location    TEXT,
    salary      TEXT,
    description TEXT,
    skills      TEXT,
    posted_at   TEXT,
    fetched_at  TEXT,
    source      TEXT
)
"""

INSERT_JOB_SQL = """
INSERT INTO jobs (
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


def ensure_database(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(CREATE_JOBS_TABLE_SQL)
        columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)").fetchall()}
        if "source" not in columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN source TEXT")
        connection.commit()


def build_description(title: str, company: str, location: str, skills: list[str]) -> str:
    primary, secondary, tertiary = skills[:3]
    extra = ", ".join(skills[3:])
    work_mode = "distributed product teams" if location == "Remote" else f"teams based in {location}"
    description = (
        f"{company} is hiring a {title} to build data products for high-scale Indian users. "
        f"The role involves owning experimentation, model development, and production analytics with "
        f"{primary}, {secondary}, and {tertiary}. You will collaborate with product, engineering, and "
        f"business teams to improve forecasting, personalization, risk, and growth decisions. "
        f"Strong foundations in statistics, clean SQL, reusable Python code, and stakeholder communication "
        f"are expected. Experience with {extra} is valued, along with comfort deploying reliable pipelines for "
        f"{work_mode}."
    )
    if len(description) < 300:
        description += (
            " The team values practical problem solving, readable documentation, monitoring, and measurable "
            "business impact across weekly releases."
        )
    return description[:800]


def extract_skills(description: str) -> str:
    lowered = description.lower()
    present = [skill for skill in SKILL_KEYWORDS if skill.lower() in lowered]
    return ", ".join(present)


def generate_records(total: int | None = None) -> list[dict[str, str]]:
    expected_total = sum(LOCATION_COUNTS.values())
    total = expected_total if total is None else total
    if total != expected_total:
        raise ValueError(f"Synthetic data generator is configured for exactly {expected_total} records.")

    rng = random.Random(20260611)
    now = datetime.now(timezone.utc)
    fetched_at = now.isoformat()
    records: list[dict[str, str]] = []

    location_sequence = [
        location
        for location, count in LOCATION_COUNTS.items()
        for _ in range(count)
    ]

    for index, location in enumerate(location_sequence):
        title = JOB_TITLES[index % len(JOB_TITLES)]
        company = COMPANIES[(index * 5) % len(COMPANIES)]
        salary = SALARY_RANGES[(index + rng.randrange(len(SALARY_RANGES))) % len(SALARY_RANGES)]
        skill_count = rng.randint(5, 8)
        start = (index * 2 + rng.randrange(len(SKILL_KEYWORDS))) % len(SKILL_KEYWORDS)
        skills = [SKILL_KEYWORDS[(start + offset) % len(SKILL_KEYWORDS)] for offset in range(skill_count)]
        description = build_description(title, company, location, skills)
        posted_at = (now - timedelta(days=index % 14, hours=rng.randint(0, 23))).isoformat()

        records.append(
            {
                "id": str(uuid.uuid4()),
                "title": title,
                "company": company,
                "location": location,
                "salary": salary,
                "description": description,
                "skills": extract_skills(description),
                "posted_at": posted_at,
                "fetched_at": fetched_at,
                "source": "synthetic",
            }
        )

    return records


def seed_database(db_path: str | Path = DB_PATH) -> int:
    db_file = Path(db_path)
    ensure_database(db_file)
    records = generate_records()

    with sqlite3.connect(db_file) as connection:
        connection.execute("DELETE FROM jobs WHERE source = ?", ("synthetic",))
        connection.executemany(
            INSERT_JOB_SQL,
            [
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
                    record["source"],
                )
                for record in records
            ],
        )
        connection.commit()

    return len(records)


def main() -> None:
    count = seed_database()
    print(f"Seeded {count} records into data/jobs.db")


if __name__ == "__main__":
    main()
