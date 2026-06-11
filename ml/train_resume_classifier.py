"""Train a resume-category classifier from a Kaggle resume dataset.

Usage:
    python -m ml.train_resume_classifier

Expects a CSV with ``Category`` and ``Resume`` columns (~962 resumes across 25
job categories) at ``data/resume_dataset/Resume Screening.csv`` or
``data/resume_dataset/UpdatedResumeDataSet.csv`` -- the dataset referenced by
``Resume Dataset/resume-screening-analysis.ipynb``. Saves a single artifact to
``ml/models/resume_category_model.joblib`` containing the fitted TF-IDF
vectorizer, classifier, and label encoder. This artifact is read by
``ml.resume_scorer.predict_resume_category`` to label uploaded resumes and
gently boost matching job titles in the resume scorer.

The raw dataset is never bundled with the app; only this small trained
artifact is loaded at runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from ml.resume_scorer import CATEGORY_MODEL_PATH, MODELS_DIR, clean_resume_text


DATASET_DIR = WORKSPACE_ROOT / "data" / "resume_dataset"
DATASET_CANDIDATES = (
    DATASET_DIR / "Resume Screening.csv",
    DATASET_DIR / "UpdatedResumeDataSet.csv",
)


def _resolve_dataset_path() -> Path:
    for candidate in DATASET_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Resume category dataset not found.\n"
        "Place the ~962-row CSV with 'Category' and 'Resume' columns referenced by "
        "Resume Dataset/resume-screening-analysis.ipynb "
        f"(e.g. 'Resume Screening.csv') in {DATASET_DIR}."
    )


def main() -> None:
    dataset_path = _resolve_dataset_path()
    frame = pd.read_csv(dataset_path)
    frame = frame.dropna(subset=["Category", "Resume"])
    frame["clean_resume"] = frame["Resume"].apply(clean_resume_text)
    frame = frame[frame["clean_resume"].str.len() > 0]

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(frame["Category"])

    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True, ngram_range=(1, 2), max_features=5000)
    features = vectorizer.fit_transform(frame["clean_resume"])

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)

    train_accuracy = accuracy_score(y_train, classifier.predict(X_train))
    test_predictions = classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"vectorizer": vectorizer, "classifier": classifier, "label_encoder": label_encoder},
        CATEGORY_MODEL_PATH,
    )

    print(f"Training rows: {len(frame)} | Categories: {len(label_encoder.classes_)}")
    print(f"Train accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy:  {test_accuracy:.3f}")
    print()
    print(classification_report(y_test, test_predictions, target_names=label_encoder.classes_, zero_division=0))
    print(f"Saved model -> {CATEGORY_MODEL_PATH}")


if __name__ == "__main__":
    main()
