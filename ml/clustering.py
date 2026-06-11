from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from config.loader import load_config


DB_PATH = WORKSPACE_ROOT / "data" / "jobs.db"
CONFIG = load_config()
KMEANS_MIN_CLUSTERS = int(CONFIG["ml"]["kmeans_min_clusters"])
KMEANS_MAX_CLUSTERS = int(CONFIG["ml"]["kmeans_max_clusters"])


@dataclass
class ClusteringArtifacts:
    labels: list[int]
    cluster_count: int
    feature_names: list[str]
    matrix_shape: tuple[int, int]
    vectorizer: TfidfVectorizer | None
    model: KMeans | None


def normalize_text(text: str | None) -> str:
    return " ".join((text or "").split())


def load_job_records(db_path: str | Path = DB_PATH, limit: int | None = None) -> list[dict[str, Any]]:
    db_file = Path(db_path)
    query = "SELECT id, title, company, location, description FROM jobs ORDER BY fetched_at DESC"
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
            "title": "Data Scientist",
            "company": "Mock One",
            "location": "Delhi",
            "description": "Python SQL Pandas machine learning statistics experimentation forecasting.",
        },
        {
            "id": "mock-2",
            "title": "ML Engineer",
            "company": "Mock Two",
            "location": "Bangalore",
            "description": "PyTorch TensorFlow Docker FastAPI model deployment MLOps pipelines.",
        },
        {
            "id": "mock-3",
            "title": "BI Analyst",
            "company": "Mock Three",
            "location": "Noida",
            "description": "Power BI Tableau SQL dashboard visualization reporting metrics.",
        },
        {
            "id": "mock-4",
            "title": "NLP Engineer",
            "company": "Mock Four",
            "location": "Remote",
            "description": "LLM RAG NLP LangChain vector search prompt engineering transformers.",
        },
        {
            "id": "mock-5",
            "title": "Data Platform Engineer",
            "company": "Mock Five",
            "location": "Pune",
            "description": "Spark Kafka Docker batch processing streaming platform reliability.",
        },
    ]


def build_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2),
        max_features=max_features,
    )


def _resolve_cluster_bounds(min_k: int | None = None, max_k: int | None = None) -> tuple[int, int]:
    resolved_min = KMEANS_MIN_CLUSTERS if min_k is None else min_k
    resolved_max = KMEANS_MAX_CLUSTERS if max_k is None else max_k
    return resolved_min, resolved_max


def _candidate_k_values(sample_count: int, min_k: int | None = None, max_k: int | None = None) -> list[int]:
    min_k, max_k = _resolve_cluster_bounds(min_k=min_k, max_k=max_k)
    upper_bound = min(max_k, sample_count - 1)
    if sample_count <= 1:
        return [1]
    if upper_bound < min_k:
        return [min(sample_count, max(1, upper_bound + 1))]
    return list(range(min_k, upper_bound + 1))


def choose_optimal_cluster_count(
    matrix,
    *,
    min_k: int | None = None,
    max_k: int | None = None,
    random_state: int = 42,
) -> int:
    sample_count = matrix.shape[0]
    candidates = _candidate_k_values(sample_count, min_k=min_k, max_k=max_k)
    if candidates == [1]:
        return 1
    if len(candidates) == 1:
        return candidates[0]

    best_k = candidates[0]
    best_score = -1.0

    for k in candidates:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(matrix)
        unique_labels = len(set(labels))
        if unique_labels <= 1 or unique_labels >= sample_count:
            continue
        score = silhouette_score(matrix, labels)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def cluster_job_descriptions(
    records: list[dict[str, Any]],
    *,
    cluster_count: int | None = None,
    min_k: int | None = None,
    max_k: int | None = None,
    random_state: int = 42,
) -> tuple[list[dict[str, Any]], ClusteringArtifacts]:
    if not records:
        vectorizer = build_vectorizer()
        artifacts = ClusteringArtifacts(
            labels=[],
            cluster_count=0,
            feature_names=[],
            matrix_shape=(0, 0),
            vectorizer=vectorizer,
            model=None,
        )
        return [], artifacts

    descriptions = [normalize_text(str(record.get("description") or record.get("title") or "")) for record in records]
    vectorizer = build_vectorizer()
    matrix = vectorizer.fit_transform(descriptions)

    if matrix.shape[1] == 0:
        raise ValueError("TF-IDF vectorization produced no usable features.")

    chosen_k = cluster_count or choose_optimal_cluster_count(
        matrix,
        min_k=min_k,
        max_k=max_k,
        random_state=random_state,
    )
    chosen_k = max(1, min(chosen_k, len(records)))

    model = KMeans(n_clusters=chosen_k, random_state=random_state, n_init=10)
    labels = model.fit_predict(matrix)

    labeled_records: list[dict[str, Any]] = []
    for record, label in zip(records, labels, strict=False):
        enriched = dict(record)
        enriched["cluster_label"] = int(label)
        labeled_records.append(enriched)

    artifacts = ClusteringArtifacts(
        labels=[int(label) for label in labels],
        cluster_count=chosen_k,
        feature_names=vectorizer.get_feature_names_out().tolist(),
        matrix_shape=matrix.shape,
        vectorizer=vectorizer,
        model=model,
    )
    return labeled_records, artifacts


if __name__ == "__main__":
    verification_records = get_verification_records()
    labeled_records, artifacts = cluster_job_descriptions(verification_records)

    assert len(labeled_records) == len(verification_records)
    assert artifacts.matrix_shape[0] == len(verification_records)
    assert 1 <= artifacts.cluster_count <= min(7, len(verification_records))
    assert len(artifacts.labels) == len(verification_records)

    print("Clustering verification passed.")
    print(f"Documents clustered: {len(verification_records)}")
    print(f"TF-IDF matrix shape: {artifacts.matrix_shape}")
    print(f"Chosen cluster count: {artifacts.cluster_count}")
    print(f"Cluster labels: {artifacts.labels}")
