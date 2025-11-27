import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = "Data KB1- Cleaned (Tiktok Songs).csv"
MODEL_PATH = "song_emotion_model.joblib"


def load_kb1_dataset(path: str = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)

    # Gunakan semua fitur numerik sebagai input model
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "song_emotion" not in df.columns:
        raise ValueError("Kolom 'song_emotion' tidak ditemukan di dataset KB1.")

    X = df[feature_cols].values
    y = df["song_emotion"].astype(str).str.strip().str.lower()
    return X, y


def build_pipeline() -> Pipeline:
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    return clf


def train_with_cross_validation() -> None:
    print(f"Loading KB1 dataset from {DATA_PATH} ...")
    X, y = load_kb1_dataset(DATA_PATH)
    print(f"Total samples: {len(y)}")

    pipeline = build_pipeline()

    # 5-fold stratified cross validation di SELURUH dataset (tanpa sampling)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("Running 5-fold stratified cross validation (accuracy)...")
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

    print("CV scores (accuracy):", scores)
    print("CV mean ± std: {:.4f} ± {:.4f}".format(scores.mean(), scores.std()))

    # Fit ulang di SELURUH dataset (100% data)
    print("\nTraining final model on full dataset (100% data)...")
    pipeline.fit(X, y)

    # Tampilkan classification report di training set (hanya untuk informasi)
    y_pred = pipeline.predict(X)
    print("\nClassification report on full training data (for reference):")
    print(classification_report(y, y_pred))

    # Simpan model
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved trained song_emotion model to {MODEL_PATH}")


if __name__ == "__main__":
    train_with_cross_validation()
