# scripts/pseudo_label_zeroshot.py
"""
Generate pseudo-labels for policy categories using HF zero-shot classification.
Writes data/processed/pseudo_labels.csv with columns:
text, ad_prob, irrelevant_prob, non_visit_prob, valid_prob, any_violation
"""
import os
from pathlib import Path
import pandas as pd

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PROC = ROOT / "data" / "processed"

def main():
    # --- args ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size for ZS inference")
    ap.add_argument("--src", type=str, default=str(DATA_PROC / "train.csv"))
    ap.add_argument("--dst", type=str, default=str(DATA_PROC / "pseudo_labels.csv"))
    ap.add_argument("--local-dir", type=str, default=str(ROOT / "backend/models/bart-large-mnli"))
    args = ap.parse_args()

    # src = DATA_PROC / "train.csv"
    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"Missing {src}. Run scripts/prepare_data.py first.")

    df = pd.read_csv(src)
    # clf = pipeline("zero-shot-classification", model=os.getenv("HF_ZS_MODEL", "facebook/bart-large-mnli"))
    # --- force offline & load from disk ---
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")  # hard guard: no network
    local_dir = args.local_dir  # e.g., backend/models/bart-large-mnli
    tok = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(local_dir, local_files_only=True)
    clf = pipeline(
        "zero-shot-classification",
        model=mdl,
        tokenizer=tok,
        device=0 if torch.cuda.is_available() else -1,
    )
    
    labels = ["advertisement", "irrelevant", "rant_without_visit", "valid"]
    # rows = []
    # for t in df["text"].astype(str).tolist():
    #     out = clf(t, candidate_labels=labels, multi_label=True)
    #     score_map = {lab: float(score) for lab, score in zip(out["labels"], out["scores"])}
    #     rows.append({
    #         "text": t,
    #         "ad_prob": score_map.get("advertisement", 0.0),
    #         "irrelevant_prob": score_map.get("irrelevant", 0.0),
    #         "non_visit_prob": score_map.get("rant_without_visit", 0.0),
    #         "valid_prob": score_map.get("valid", 0.0),
    #     })
    texts = df["text"].astype(str).tolist()
    outputs = clf(
        texts,
        candidate_labels=labels,
        hypothesis_template="This review is {}.",  # slightly sharper NLI prompt
        multi_label=True,
        batch_size=args.batch_size,
    )

    rows = []
    for t, out in zip(texts, outputs):
        score_map = {lab: float(score) for lab, score in zip(out["labels"], out["scores"])}
        rows.append(
            {
                "text": t,
                "ad_prob": score_map.get("advertisement", 0.0),
                "irrelevant_prob": score_map.get("irrelevant", 0.0),
                "non_visit_prob": score_map.get("rant_without_visit", 0.0),
                "valid_prob": score_map.get("valid", 0.0),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df["any_violation"] = ((out_df["ad_prob"] > 0.6) | (out_df["irrelevant_prob"] > 0.6) | (out_df["non_visit_prob"] > 0.6)).astype(int)
    # out_path = DATA_PROC / "pseudo_labels.csv"
    out_path = Path(args.dst)
    out_df.to_csv(out_path, index=False)
    print(f"[OK] Wrote {len(out_df)} pseudo-labeled rows to {out_path}")

if __name__ == "__main__":
    main()
