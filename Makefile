# Makefile
.PHONY: venv install data train eval api test

venv:
	python -m venv .venv

install:
	pip install -r backend/requirements.txt

data:
	python scripts/prepare_data.py --sample 5000

train:
	python scripts/train_baseline.py

eval:
	python scripts/evaluate.py

api:
	uvicorn backend.src.main:app --reload

test:
	pytest -q


# --- Offline model workflow ---
prefetch-models:
	python scripts/prefetch_models.py

pseudo-label-offline:
	TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
	python scripts/pseudo_label_zeroshot.py --batch-size 8