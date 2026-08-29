"""
main.py
=======
FastAPI application for the P3 Norm Classifier web app.
Stage 1: BERT-only norm detection.
Stage 2: All three country classifiers (BERT / DeBERTa / RoBERTa) — shown as comparative stats.
"""

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from inference import predict, get_available_models, preload_all

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="P3 Norm Classifier API",
    description=(
        "Two-stage cultural norm classifier. "
        "Stage 1: BERT detects if the sentence is a norm. "
        "Stage 2: DeBERTa, BERT, and RoBERTa country classifiers run in parallel "
        "and their results are returned for comparative analysis."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent   # PROJECT_P3/
RESULTS_DIR = BASE_DIR / "results"


# ── Startup: preload all models ───────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("⏳ Preloading BERT norm model + all country models…")
    preload_all()
    print("✅ All models ready.")


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    sentence: str = Field(..., min_length=1, max_length=2000)
    norm_threshold: float = Field(0.6, ge=0.0, le=1.0)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/models")
def models():
    return get_available_models()


@app.post("/api/predict")
def predict_endpoint(req: PredictRequest):
    """
    Run the two-stage prediction pipeline.

    Response:
    {
        "sentence": str,
        "is_norm": bool,
        "bert_norm_confidence": float,
        "country_results": {
            "bert":    [{country, flag, confidence}, ...],
            "deberta": [{country, flag, confidence}, ...],
            "roberta": [{country, flag, confidence}, ...]
        } | null
    }
    """
    if not req.sentence.strip():
        raise HTTPException(status_code=422, detail="Sentence must not be empty.")
    result = predict(req.sentence, req.norm_threshold)
    return result


@app.get("/api/results")
def results():
    """Return all metrics JSON files bundled together."""
    data = {}

    # Norm classifier results (BERT is the primary norm model)
    for model_key in ["deberta", "bert", "roberta"]:
        path = RESULTS_DIR / f"{model_key}_results.json"
        if path.exists():
            with open(path) as f:
                data[f"norm_{model_key}"] = json.load(f)

    # Country classifier results
    country_dir = RESULTS_DIR / "country"
    if country_dir.exists():
        for model_key in ["deberta", "bert", "roberta"]:
            path = country_dir / f"country_{model_key}_results.json"
            if path.exists():
                with open(path) as f:
                    data[f"country_{model_key}"] = json.load(f)

    # Comparison CSVs
    cmp = RESULTS_DIR / "metrics_comparison.csv"
    if cmp.exists():
        import pandas as pd
        data["norm_comparison"] = pd.read_csv(cmp).to_dict(orient="records")

    country_cmp = RESULTS_DIR / "country" / "country_metrics_comparison.csv"
    if country_cmp.exists():
        import pandas as pd
        data["country_comparison"] = pd.read_csv(country_cmp).to_dict(orient="records")

    return data


@app.get("/api/results/plots/{filename}")
def serve_plot(filename: str):
    """Serve a training-curve or confusion-matrix PNG."""
    candidates = [
        RESULTS_DIR / "plots" / filename,
        RESULTS_DIR / "country" / "plots" / filename,
        RESULTS_DIR / "large_models" / "plots" / filename,
    ]
    for path in candidates:
        if path.exists() and path.suffix == ".png":
            return FileResponse(str(path), media_type="image/png")
    raise HTTPException(status_code=404, detail=f"Plot '{filename}' not found.")


@app.get("/api/results/plots")
def list_plots():
    """List all available plot filenames."""
    plots = []
    for base in [
        RESULTS_DIR / "plots",
        RESULTS_DIR / "country" / "plots",
    ]:
        if base.exists():
            plots.extend([p.name for p in base.glob("*.png")])
    return {"plots": plots}
