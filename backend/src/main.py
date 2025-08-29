# backend/src/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any
import os

from .model_io import load_model, MODEL_PATH
from .rules import rule_flags
from .preprocess import clean_text

# Optional: Hugging Face zero-shot classification
HF_ENABLED = os.getenv("HF_ZEROSHOT", "0") == "1"
ZS_MODEL_NAME = os.getenv("HF_ZS_MODEL", "facebook/bart-large-mnli")

zs_classifier = None
if HF_ENABLED:
    try:
        from transformers import pipeline
        zs_classifier = pipeline("zero-shot-classification", model=ZS_MODEL_NAME)
    except Exception as e:
        zs_classifier = None
        print(f"[WARN] Could not initialize HF zero-shot pipeline: {e}")

app = FastAPI(title="ReviewGuard API", version="0.2.0")

# CORS for Vite dev server (http://localhost:5173 by default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReviewIn(BaseModel):
    text: str

class PredictOut(BaseModel):
    quality_violation_prob: float
    flags: Dict[str, bool]
    zeroshot: Optional[Dict[str, float]] = None
    action: str
    cleaned_text: str

# Load model at startup (if exists)
clf = None
if MODEL_PATH.exists():
    try:
        clf = load_model(MODEL_PATH)
        print(f"[INFO] Loaded model from: {MODEL_PATH}")
    except Exception as e:
        print(f"[WARN] Could not load model: {e}")

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "model_loaded": clf is not None,
        "model_path": str(MODEL_PATH),
        "hf_zeroshoot_enabled": bool(zs_classifier),
    }

@app.post("/predict", response_model=PredictOut)
def predict(inp: ReviewIn) -> PredictOut:
    global clf
    text = inp.text or ""
    cleaned = clean_text(text)

    proba = 0.0
    if clf is not None:
        try:
            # Assumes binary classifier with predict_proba
            proba = float(clf.predict_proba([cleaned])[0][1])
        except Exception:
            # Fallback to decision_function if needed
            try:
                from math import exp
                score = float(clf.decision_function([cleaned])[0])
                proba = 1 / (1 + exp(-score))
            except Exception:
                proba = 0.0

    flags = rule_flags(text)

    zs = None
    if zs_classifier is not None:
        labels = ["advertisement", "irrelevant", "rant_without_visit", "valid"]
        try:
            res = zs_classifier(text, candidate_labels=labels, multi_label=True)
            zs = {label: float(score) for label, score in zip(res["labels"], res["scores"])}
        except Exception as e:
            print(f"[WARN] Zero-shot inference failed: {e}")
            zs = None

    # Simple policy: flag if model is confident OR any rule trips
    threshold = float(os.getenv("MODEL_FLAG_THRESHOLD", "0.70"))
    should_flag = (proba >= threshold) or any(flags.values())
    action = "flag" if should_flag else "ok"

    return PredictOut(
        quality_violation_prob=proba,
        flags=flags,
        zeroshot=zs,
        action=action,
        cleaned_text=cleaned,
    )
