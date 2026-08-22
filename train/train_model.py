"""Train a TF-IDF + Logistic Regression sentiment classifier.

Downloads a 20k subset of Hugging Face `amazon_polarity`, trains the model,
prints evaluation metrics, and writes artifacts to ../models/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"

TRAIN_SIZE = 16_000
TEST_SIZE = 4_000
RANDOM_STATE = 42


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_subset() -> pd.DataFrame:
    """Load a stratified 20k subset of amazon_polarity."""
    print("Downloading amazon_polarity from Hugging Face (first run may take a while)...")
    # Stream then collect enough rows of each class for a balanced 20k sample.
    ds = load_dataset("fancyzhx/amazon_polarity", split="train", streaming=True)
    ds = ds.take(80_000)

    rows: list[dict] = []
    needed = TRAIN_SIZE + TEST_SIZE
    per_class = needed // 2
    counts = {0: 0, 1: 0}

    for example in ds:
        label = int(example["label"])
        if counts[label] >= per_class:
            if counts[0] >= per_class and counts[1] >= per_class:
                break
            continue
        content = example.get("content") or example.get("text") or ""
        title = example.get("title") or ""
        text = f"{title} {content}".strip()
        if not text:
            continue
        rows.append({"text": text, "label": label})
        counts[label] += 1

    df = pd.DataFrame(rows)
    print(f"Collected {len(df)} reviews  (neg={counts[0]}, pos={counts[1]})")
    return df


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_subset()
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0].drop_duplicates(subset=["text"])

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=TEST_SIZE / (TRAIN_SIZE + TEST_SIZE),
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    print(f"Train size: {len(x_train)}  Test size: {len(x_test)}")

    vectorizer = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
    )
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = LogisticRegression(
        max_iter=1000,
        C=2.0,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )
    print("Training Logistic Regression...")
    model.fit(x_train_vec, y_train)

    y_pred = model.predict(x_test_vec)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["negative", "positive"],
        output_dict=True,
        digits=4,
    )
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"], digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, MODELS_DIR / "model.joblib")
    joblib.dump(vectorizer, MODELS_DIR / "vectorizer.joblib")
    METRICS_PATH.write_text(json.dumps(report, indent=2))
    print(f"Saved model artifacts to {MODELS_DIR}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
