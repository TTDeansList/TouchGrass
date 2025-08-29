# scripts/prepare_data.py
"""
Prepare data by loading any CSV/JSON files in data/raw/, extracting a unified 'text' column,
cleaning, deduplicating, and writing splits to data/processed/.
"""

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

# Try to import clean_text from backend
try:
    from backend.src.preprocess import clean_text  # when running from repo root
except Exception:
    try:
        from src.preprocess import clean_text  # when running inside backend
    except Exception:
        # minimal fallback
        import re, html
        URL_RE = re.compile(r'https?://\S+|www\.\S+')
        EMOJI_RE = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
        def clean_text(s: str) -> str:
            s = html.unescape(s or "")
            s = s.strip().lower()
            s = URL_RE.sub(" <URL> ", s)
            s = EMOJI_RE.sub(" ", s)
            s = re.sub(r"\s+", " ", s)
            return s

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DATA_PROC.mkdir(parents=True, exist_ok=True)

TEXT_CANDIDATE_COLS = ["text", "review", "review_text", "content", "comments", "body", "snippet", "reviewBody"]

def find_text_column(df: pd.DataFrame) -> Optional[str]:
    lc = [c.lower() for c in df.columns]
    mapping = {c.lower(): c for c in df.columns}
    for cand in TEXT_CANDIDATE_COLS:
        if cand in lc:
            return mapping[cand]
    # Try heuristic: longest string column
    str_cols = [c for c in df.columns if df[c].dtype == object]
    return str_cols[0] if str_cols else None

def load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".csv"]:
        return pd.read_csv(path)
    if path.suffix.lower() in [".json", ".jsonl"]:
        try:
            return pd.read_json(path, lines=path.suffix.lower()==".jsonl")
        except Exception:
            # try line-by-line
            rows = []
            with open(path) as f:
                for line in f:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
            return pd.DataFrame(rows)
    raise ValueError(f"Unsupported file: {path}")

def main(sample: Optional[int] = None):
    files: List[Path] = []
    for ext in ("*.csv", "*.json", "*.jsonl"):
        files.extend(DATA_RAW.glob(ext))
    if not files:
        raise SystemExit(f"No raw files found in {DATA_RAW}. Please put dataset files there.")

    dfs = []
    for f in files:
        try:
            df = load_any(f)
            col = find_text_column(df)
            if col is None:
                print(f"[WARN] No text-like column in {f.name}; skipping.")
                continue
            sub = df[[col]].rename(columns={col: "text"})
            dfs.append(sub)
            print(f"[INFO] Loaded {len(sub)} rows from {f.name} (col='{col}')")
        except Exception as e:
            print(f"[WARN] Failed to load {f}: {e}")

    if not dfs:
        raise SystemExit("No usable data loaded.")
    df = pd.concat(dfs, ignore_index=True)
    df["text"] = df["text"].astype(str).map(clean_text)
    df = df.dropna().drop_duplicates()
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42)

    # Split
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Save
    train_df.to_csv(DATA_PROC / "train.csv", index=False)
    val_df.to_csv(DATA_PROC / "val.csv", index=False)
    test_df.to_csv(DATA_PROC / "test.csv", index=False)

    print(f"[OK] Saved processed splits to {DATA_PROC}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="Optional sample size for quick runs")
    args = ap.parse_args()
    main(sample=args.sample)
