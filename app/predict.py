from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

LABEL_MAP = {0: "negative", 1: "positive"}

_model = None
_vectorizer = None


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_artifacts() -> None:
    global _model, _vectorizer
    model_path = MODELS_DIR / "model.joblib"
    vectorizer_path = MODELS_DIR / "vectorizer.joblib"
    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError(
            "Model files not found. Run `python train/train_model.py` first, "
            f"or place model.joblib and vectorizer.joblib in {MODELS_DIR}."
        )
    _model = joblib.load(model_path)
    _vectorizer = joblib.load(vectorizer_path)


def is_loaded() -> bool:
    return _model is not None and _vectorizer is not None


def predict_one(text: str) -> dict:
    if not is_loaded():
        load_artifacts()

    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Review text is empty after cleaning.")

    features = _vectorizer.transform([cleaned])
    proba = _model.predict_proba(features)[0]
    classes = list(_model.classes_)
    scores = {LABEL_MAP.get(int(cls), str(cls)): float(p) for cls, p in zip(classes, proba)}
    # Ensure both keys exist even if a class is missing.
    scores.setdefault("negative", 0.0)
    scores.setdefault("positive", 0.0)

    label_idx = int(np.argmax(proba))
    sentiment = LABEL_MAP.get(int(classes[label_idx]), "unknown")
    confidence = float(proba[label_idx])

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "probabilities": {
            "negative": round(scores["negative"], 4),
            "positive": round(scores["positive"], 4),
        },
    }


def predict_many(texts: list[str]) -> list[dict]:
    return [predict_one(t) for t in texts]
