# backend/src/model_io.py
from pathlib import Path
from typing import Any
import joblib
import time

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "sk_baseline.joblib"

def save_model(model: Any, path: Path = MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    joblib.dump(model, tmp)
    time.sleep(0.05)
    tmp.replace(path)
    return path

def load_model(path: Path = MODEL_PATH) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)
