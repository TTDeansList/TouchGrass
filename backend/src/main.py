from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from src.model_io import load_model, MODEL_PATH
from src.rules import rule_flags

app = FastAPI(title="ReviewGuard API", version="0.1.0")

# CORS for Vite dev server (http://localhost:5173 by default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model (train first if not present)
try:
    clf = load_model(MODEL_PATH)
except Exception:
    clf = None

class ReviewIn(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"ok": True, "model_loaded": clf is not None}

@app.post("/predict")
def predict(inp: ReviewIn):
    if clf is None:
        return {"error": "Model not loaded. Train and save to backend/models/sk_baseline.joblib"}
    proba = float(clf.predict_proba([inp.text])[0][1])
    flags = rule_flags(inp.text)
    action = "flag" if (proba > 0.7 or any(flags.values())) else "ok"
    return {"quality_violation_prob": proba, "flags": flags, "action": action}
