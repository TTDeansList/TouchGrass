# scripts/predict_cli.py
"""
Quick CLI to score a single review or a CSV of reviews using the saved model and rule flags.
"""
from pathlib import Path
import sys
import pandas as pd

try:
    from backend.src.model_io import load_model, MODEL_PATH
    from backend.src.rules import rule_flags
    from backend.src.preprocess import clean_text
except Exception:
    from src.model_io import load_model, MODEL_PATH
    from src.rules import rule_flags
    from src.preprocess import clean_text

def score_text(model, text: str):
    cleaned = clean_text(text)
    proba = float(model.predict_proba([cleaned])[0][1])
    flags = rule_flags(text)
    action = "flag" if (proba >= 0.7 or any(flags.values())) else "ok"
    return proba, flags, action

def main():
    model = load_model(MODEL_PATH)
    if len(sys.argv) == 1:
        print("Usage:")
        print("  python scripts/predict_cli.py \"Your review text here\"")
        print("  python scripts/predict_cli.py path/to/file.csv  # with a 'text' column")
        sys.exit(0)
    arg = sys.argv[1]
    p = Path(arg)
    if p.exists() and p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        if "text" not in df.columns:
            raise SystemExit("CSV must contain a 'text' column.")
        rows = []
        for t in df["text"].astype(str).tolist():
            proba, flags, action = score_text(model, t)
            rows.append({"text": t, "proba": proba, "flags": flags, "action": action})
        out = p.with_name(p.stem + "_scored.csv")
        out_df = pd.DataFrame(rows)
        out_df.to_csv(out, index=False)
        print(f"[OK] Wrote {len(out_df)} rows to {out}")
    else:
        text = arg
        proba, flags, action = score_text(model, text)
        print({"text": text, "quality_violation_prob": proba, "flags": flags, "action": action})

if __name__ == "__main__":
    main()
