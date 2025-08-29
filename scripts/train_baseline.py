# scripts/train_baseline.py
"""
Train the baseline TF-IDF + Logistic Regression model on processed data.
"""
from pathlib import Path
import pandas as pd

# Import training function from backend
try:
    from backend.src.train_sklearn import train_quality_baseline
except Exception:
    from src.train_sklearn import train_quality_baseline  # when running inside backend

ROOT = Path(__file__).resolve().parents[1]
DATA_PROC = ROOT / "data" / "processed"

def main():
    train_path = DATA_PROC / "train.csv"
    val_path = DATA_PROC / "val.csv"
    if not train_path.exists():
        raise SystemExit(f"Missing {train_path}. Run scripts/prepare_data.py first.")
    # Use train + val for more data (small datasets)
    df_train = pd.read_csv(train_path)
    if val_path.exists():
        df_val = pd.read_csv(val_path)
        df = pd.concat([df_train, df_val], ignore_index=True)
    else:
        df = df_train
    train_quality_baseline(df)

if __name__ == "__main__":
    main()
