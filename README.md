# Filtering the Noise: ML for Trustworthy Location Reviews

Hackathon project implementing an end-to-end pipeline to detect low-quality and irrelevant location reviews, following the TikTok TechJam guidance.

## Quickstart

```bash
# From repo root
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt

# Prepare data
mkdir -p data/raw
# Put CSV/JSON review files into data/raw (e.g., Kaggle/UCSD Google Local Reviews)
python scripts/prepare_data.py --sample 5000

# Train baseline model (TF-IDF + Logistic Regression)
python scripts/train_baseline.py

# Evaluate on test split
python scripts/evaluate.py

# Run API
uvicorn backend.src.main:app --reload
# Visit http://127.0.0.1:8000/docs
