# scripts/pseudo_label_zeroshot.py
"""
Generate pseudo-labels for policy categories using HF zero-shot classification.
Writes data/processed/pseudo_labels.csv with columns:
text, ad_prob, irrelevant_prob, non_visit_prob, valid_prob, any_violation
"""
import os
from pathlib import Path
import pandas as pd

from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PROC = ROOT / "data" / "processed"

def main():
    src = DATA_PROC / "train.csv"
    if not src.exists():
        raise SystemExit(f"Missing {src}. Run scripts/prepare_data.py first.")

    df = pd.read_csv(src)
    clf = pipeline("zero-shot-classification", model=os.getenv("HF_ZS_MODEL", "facebook/bart-large-mnli"))

    labels = ["advertisement", "irrelevant", "rant_without_visit", "valid"]
    rows = []
    for t in df["text"].astype(str).tolist():
        out = clf(t, candidate_labels=labels, multi_label=True)
        score_map = {lab: float(score) for lab, score in zip(out["labels"], out["scores"])}
        rows.append({
            "text": t,
            "ad_prob": score_map.get("advertisement", 0.0),
            "irrelevant_prob": score_map.get("irrelevant", 0.0),
            "non_visit_prob": score_map.get("rant_without_visit", 0.0),
            "valid_prob": score_map.get("valid", 0.0),
        })
    out_df = pd.DataFrame(rows)
    out_df["any_violation"] = ((out_df["ad_prob"] > 0.6) | (out_df["irrelevant_prob"] > 0.6) | (out_df["non_visit_prob"] > 0.6)).astype(int)
    out_path = DATA_PROC / "pseudo_labels.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[OK] Wrote {len(out_df)} pseudo-labeled rows to {out_path}")

if __name__ == "__main__":
    main()
