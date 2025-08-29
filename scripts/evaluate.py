# scripts/evaluate.py
"""
Evaluate the saved model on data/processed/test.csv and write a classification report.
"""
import json
from pathlib import Path
import sys
import pandas as pd
from sklearn.metrics import classification_report

# --- Path bootstrap ---
ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
# ----------------------

from backend.src.model_io import load_model, MODEL_PATH
from backend.src.preprocess import clean_text
from backend.src.rules import rule_flags

DATA_PROC = ROOT / "data" / "processed"

def main():
    test_path = DATA_PROC / "test.csv"
    if not test_path.exists():
        raise SystemExit(f"Missing {test_path}. Run scripts/prepare_data.py first.")
    df = pd.read_csv(test_path)
    df["text"] = df["text"].astype(str).map(clean_text)
    y_true = df["text"].apply(lambda t: int(any(rule_flags(t).values()))).tolist()

    model = load_model(MODEL_PATH)
    y_pred = model.predict(df["text"].tolist())

    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    print(classification_report(y_true, y_pred, zero_division=0))
    out = ROOT / "backend" / "models" / "test_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"[OK] Saved report to {out}")

if __name__ == "__main__":
    main()
