"""Train a resume-quality model from the "AI-Powered Resume Screening Dataset 2025".

Usage:
    python -m ml.train_resume_quality_model

Expects ``data/resume_dataset/AI_Resume_Screening.csv`` with columns
``Skills``, ``Certifications``, ``Experience (Years)``, ``Education``,
``Projects Count``, ``AI Score (0-100)``, and ``Recruiter Decision``.

Saves a single artifact to ``ml/models/resume_quality_model.joblib`` containing
a TF-IDF vectorizer (fit on Skills + Certifications), a numeric feature scaler
(Experience, Projects Count, education level), a Ridge regressor for the
0-100 AI Score, and a LogisticRegression classifier for Hire/Reject. This
artifact is read by ``ml.resume_scorer.predict_resume_quality`` to add a
"ML-predicted AI Score" and "hire likelihood" to the resume scorer.

The raw dataset is never bundled with the app; only this small trained
artifact is loaded at runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from ml.resume_scorer import MODELS_DIR, QUALITY_MODEL_PATH, clean_resume_text, education_level_from_text


DATASET_PATH = WORKSPACE_ROOT / "data" / "resume_dataset" / "AI_Resume_Screening.csv"


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}.\n"
            "Download 'AI_Resume_Screening.csv' from the Kaggle dataset "
            "'ai-powered-resume-screening-dataset-2025' (the same one referenced by "
            "Resume Dataset/ai-powered-resume-screening-dataset-2025.ipynb) and place it at that path."
        )

    frame = pd.read_csv(DATASET_PATH)
    frame = frame.dropna(subset=["Skills", "AI Score (0-100)", "Recruiter Decision"])

    text_features = (frame["Skills"].fillna("") + " " + frame["Certifications"].fillna("")).apply(clean_resume_text)
    education_levels = frame["Education"].apply(education_level_from_text)

    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True, ngram_range=(1, 2), max_features=2000)
    text_matrix = vectorizer.fit_transform(text_features)

    numeric_columns = np.column_stack(
        [
            frame["Experience (Years)"].to_numpy(dtype=float),
            frame["Projects Count"].to_numpy(dtype=float),
            education_levels.to_numpy(dtype=float),
        ]
    )
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_columns)

    features = hstack([text_matrix, csr_matrix(numeric_scaled)]).tocsr()

    ai_scores = frame["AI Score (0-100)"].to_numpy(dtype=float)
    hire_labels = (frame["Recruiter Decision"].astype(str).str.strip().str.lower() == "hire").astype(int).to_numpy()

    X_train, X_test, y_score_train, y_score_test, y_hire_train, y_hire_test = train_test_split(
        features, ai_scores, hire_labels, test_size=0.2, random_state=42, stratify=hire_labels
    )

    regressor = Ridge(alpha=1.0)
    regressor.fit(X_train, y_score_train)
    score_predictions = np.clip(regressor.predict(X_test), 0.0, 100.0)
    mae = mean_absolute_error(y_score_test, score_predictions)

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_hire_train)
    hire_accuracy = accuracy_score(y_hire_test, classifier.predict(X_test))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"vectorizer": vectorizer, "scaler": scaler, "regressor": regressor, "classifier": classifier},
        QUALITY_MODEL_PATH,
    )

    print(f"Training rows: {len(frame)}")
    print(f"AI Score MAE (test, 0-100 scale): {mae:.2f}")
    print(f"Hire/Reject accuracy (test):      {hire_accuracy:.3f}")
    print(f"Saved model -> {QUALITY_MODEL_PATH}")


if __name__ == "__main__":
    main()
