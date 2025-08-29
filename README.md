# 🔍 Filtering the Noise: ML for Trustworthy Location Reviews

Hackathon project implementing an end-to-end pipeline to detect low-quality and irrelevant location reviews, following the TikTok TechJam guidance.

---

## 🚀 Quickstart

### 1. Setup environment
```bash
# From repo root
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 2. Prepare data
``` bash
mkdir -p data/raw
# Put CSV/JSON review files into data/raw
# (e.g., Kaggle/UCSD Google Local Reviews dataset)
python scripts/prepare_data.py --sample 5000
```

### 3. Train baseline model (TF-IDF + Logistic Regression)
``` bash
python scripts/train_baseline.py
```

### 4. Evaluate on test split
``` bash
python scripts/evaluate.py
```

---

## Testing

### 🖥️ Streamlit Demo (UI)
```bash
# Terminal A — backend
uvicorn backend.src.main:app --reload

# Optional: enable zero-shot enrichment
export HF_ZEROSHOT=1
export HF_ZS_MODEL=facebook/bart-large-mnli
uvicorn backend.src.main:app --reload

# Terminal B — streamlit
streamlit run frontend_streamlit/app.py
```

### 🔧 Testing from CLI
```bash
python scripts/predict_cli.py "I haven't been here but it looks terrible"
```

### 📡 Testing via API
```bash
# Single Request
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "Best pizza! Visit www.pizzapromo.com for discounts!"}'

# Health Check
curl http://127.0.0.1:8000/health
```

---

## ✨ Features
- Preprocessing & weak labeling via regex rules
- TF-IDF + Logistic Regression baseline
- Robust splitting for tiny datasets
- REST API (/predict, /health) with FastAPI
- CLI scorer for quick testing
- Streamlit UI for demo
- Optional zero-shot enrichment (HuggingFace)
