from joblib import dump, load
from pathlib import Path
from typing import Any

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "sk_baseline.joblib"

def save_model(model: Any, path: Path = MODEL_PATH):
    dump(model, path)
    return path

def load_model(path: Path = MODEL_PATH):
    return load(path)
