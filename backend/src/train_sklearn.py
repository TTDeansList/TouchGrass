# backend/src/train_sklearn.py
import json
import os
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

from .preprocess import clean_text
from .rules import rule_flags
from .model_io import save_model, MODEL_DIR

RANDOM_STATE = 42

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "text" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'text' column")
    df["text"] = df["text"].astype(str).map(clean_text)

    # weak label: violation if any rule trips
    flags = df["text"].apply(rule_flags)
    df["quality_violation_any"] = flags.apply(lambda f: int(any(f.values())))
    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_train, df_temp = train_test_split(df, test_size=(test_size + val_size), random_state=RANDOM_STATE, stratify=df["quality_violation_any"])
    rel_val = val_size / (test_size + val_size)
    df_val, df_test = train_test_split(df_temp, test_size=(1 - rel_val), random_state=RANDOM_STATE, stratify=df_temp["quality_violation_any"])
    return df_train, df_val, df_test

def build_pipeline(max_features: int = 20000) -> Pipeline:
    pipe = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2),
            max_features=max_features,
            min_df=2
        )),
        ("clf", LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])
    return pipe

def train_quality_baseline(df: pd.DataFrame, outdir: Path = MODEL_DIR) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = prepare_dataframe(df)
    df_train, df_val, df_test = split_data(df)

    X_train, y_train = df_train["text"].tolist(), df_train["quality_violation_any"].tolist()
    X_val, y_val = df_val["text"].tolist(), df_val["quality_violation_any"].tolist()
    X_test, y_test = df_test["text"].tolist(), df_test["quality_violation_any"].tolist()

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    # Evaluate
    def report_split(name, X, y):
        preds = pipe.predict(X)
        rep = classification_report(y, preds, output_dict=True, zero_division=0)
        print(f"=== {name} ===")
        print(classification_report(y, preds, zero_division=0))
        return rep

    metrics = {
        "val": report_split("Validation", X_val, y_val),
        "test": report_split("Test", X_test, y_test),
    }

    # Save artifacts
    model_path = save_model(pipe)
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[INFO] Saved model to {model_path}")
    print(f"[INFO] Saved metrics to {outdir / 'metrics.json'}")
    return model_path

if __name__ == "__main__":
    # Minimal demo training using a tiny inline dataset
    data = {
        "text": [
            "Buy my course at www.spam.com, best discount now!",
            "Food was fresh and staff were friendly. Would return.",
            "I haven't been here but I hate the owner.",
            "Try my promo code TODAY!",
            "Great coffee near the river, cozy ambience.",
            "subscribe to my channel for updates",
            "never visited but looks bad",
            "amazing service and lovely staff",
        ]
    }
    df = pd.DataFrame(data)
    train_quality_baseline(df)
