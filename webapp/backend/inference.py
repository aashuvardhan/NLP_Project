"""
inference.py
============
Loads trained checkpoints once at startup and exposes a single
`predict(sentence, norm_threshold)` function that runs:
  - BERT norm classifier (Stage 1) to decide if the sentence is a norm
  - All 3 country classifiers (DeBERTa / BERT / RoBERTa) if it is a norm
Returns a structured dict ready to be serialised as JSON.
"""

import os

# Force offline mode BEFORE importing transformers/huggingface_hub.
# Newer versions of huggingface_hub validate absolute paths as repo IDs which
# breaks local checkpoint loading in Docker. Offline mode bypasses Hub entirely.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Paths ────────────────────────────────────────────────────────────────────
# In Docker: /app/inference.py lives in /app/, and saved_models is mounted at
# /app/saved_models (via docker-compose volume ./saved_models:/app/saved_models).
# In local dev: inference.py is at webapp/backend/inference.py, so PROJECT_P3/
# is 3 levels up.
_HERE = Path(__file__).resolve().parent   # /app  (Docker) or .../webapp/backend (local)
_DOCKER_MODELS = _HERE / "saved_models"
_LOCAL_MODELS  = _HERE.parent.parent.parent / "saved_models"   # PROJECT_P3/saved_models
SAVED_MODELS = _DOCKER_MODELS if _DOCKER_MODELS.exists() else _LOCAL_MODELS

# Stage 1: only BERT is used for norm classification
NORM_MODEL_DIR = SAVED_MODELS / "bert_best"

COUNTRY_MODEL_DIRS = {
    "deberta": SAVED_MODELS / "country_deberta_best",
    "bert":    SAVED_MODELS / "country_bert_best",
    "roberta": SAVED_MODELS / "country_roberta_best",
}

DEVICE = torch.device("cpu")   # CPU inference — no GPU required

# ── Country flags lookup ──────────────────────────────────────────────────────
COUNTRY_FLAGS = {
    "Afghanistan": "🇦🇫", "Albania": "🇦🇱", "Algeria": "🇩🇿", "Angola": "🇦🇴",
    "Argentina": "🇦🇷", "Australia": "🇦🇺", "Austria": "🇦🇹", "Bangladesh": "🇧🇩",
    "Belgium": "🇧🇪", "Bhutan": "🇧🇹", "Bosnia and Herzegovina": "🇧🇦",
    "Brazil": "🇧🇷", "Bulgaria": "🇧🇬", "Cambodia": "🇰🇭", "Canada": "🇨🇦",
    "China": "🇨🇳", "Croatia": "🇭🇷", "Czech Republic": "🇨🇿", "Denmark": "🇩🇰",
    "Egypt": "🇪🇬", "Estonia": "🇪🇪", "Finland": "🇫🇮", "France": "🇫🇷",
    "Germany": "🇩🇪", "Greece": "🇬🇷", "Hungary": "🇭🇺", "India": "🇮🇳",
    "Indonesia": "🇮🇩", "Ireland": "🇮🇪", "Israel": "🇮🇱", "Italy": "🇮🇹",
    "Japan": "🇯🇵", "Malaysia": "🇲🇾", "Mexico": "🇲🇽", "Netherlands": "🇳🇱",
    "New Zealand": "🇳🇿", "Nigeria": "🇳🇬", "North Korea": "🇰🇵", "Norway": "🇳🇴",
    "Philippines": "🇵🇭", "Poland": "🇵🇱", "Portugal": "🇵🇹", "Romania": "🇷🇴",
    "Russia": "🇷🇺", "Singapore": "🇸🇬", "South Africa": "🇿🇦",
    "South Korea": "🇰🇷", "Spain": "🇪🇸", "Sweden": "🇸🇪", "Switzerland": "🇨🇭",
    "Thailand": "🇹🇭", "Turkey": "🇹🇷", "Ukraine": "🇺🇦",
    "United Kingdom": "🇬🇧", "United States": "🇺🇸", "Vietnam": "🇻🇳",
}


# ── Model cache ───────────────────────────────────────────────────────────────
class _ModelCache:
    """Lazy-loaded cache. Models are loaded once, then kept in memory."""

    def __init__(self):
        self._norm_model = None              # (tokenizer, model) for BERT
        self._country_models: dict = {}      # key → (tokenizer, model, id2label)

    # ── BERT norm classifier ──────────────────────────────────────────────────
    def load_norm(self):
        if self._norm_model is not None:
            return self._norm_model
        path = str(NORM_MODEL_DIR)
        print(f"[cache] Loading BERT norm model from {path}…")
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
        model.eval().to(DEVICE)
        self._norm_model = (tokenizer, model)
        print("[cache] BERT norm model ready.")
        return self._norm_model

    # ── Country classifiers ───────────────────────────────────────────────────
    def load_country(self, key: str):
        if key in self._country_models:
            return self._country_models[key]
        path = COUNTRY_MODEL_DIRS[key]
        print(f"[cache] Loading country model '{key}' from {path}…")
        label_map_path = path / "label_map.json"
        with open(label_map_path, "r") as f:
            raw_map = json.load(f)
        # label_map.json may be {"id2label": {"0": "Country", ...}, "label2id": {...}}
        # or a flat {"0": "Country", ...} — handle both.
        if "id2label" in raw_map:
            id2label = {int(k): v for k, v in raw_map["id2label"].items()}
        else:
            id2label = {int(k): v for k, v in raw_map.items() if k.isdigit()}
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True)
        model.eval().to(DEVICE)
        self._country_models[key] = (tokenizer, model, id2label)
        print(f"[cache] Country '{key}' ready.")
        return self._country_models[key]

    def norm_model_available(self) -> bool:
        return NORM_MODEL_DIR.exists()

    def available_country_models(self):
        return [k for k, p in COUNTRY_MODEL_DIRS.items() if p.exists()]


_cache = _ModelCache()


def preload_all():
    """Call at startup to load BERT norm model + all available country models."""
    if _cache.norm_model_available():
        _cache.load_norm()
    for key in _cache.available_country_models():
        _cache.load_country(key)


# ── Inference helpers ─────────────────────────────────────────────────────────
def _norm_predict(sentence: str) -> dict:
    """Run BERT norm classifier. Returns {is_norm, confidence}."""
    tokenizer, model = _cache.load_norm()
    enc = tokenizer(
        sentence, return_tensors="pt",
        max_length=128, truncation=True, padding="max_length"
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]
    norm_prob = float(probs[1])   # label 1 = norm
    return {"confidence": round(norm_prob, 4)}


def _country_predict(key: str, sentence: str, top_k: int = 3) -> list:
    """Returns top-k [{country, flag, confidence}] for one country backbone."""
    tokenizer, model, id2label = _cache.load_country(key)
    enc = tokenizer(
        sentence, return_tensors="pt",
        max_length=128, truncation=True, padding="max_length"
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]
    top_indices = torch.topk(probs, k=min(top_k, len(probs))).indices.tolist()
    return [
        {
            "country": id2label[i],
            "flag": COUNTRY_FLAGS.get(id2label[i], "🌍"),
            "confidence": round(float(probs[i]), 4),
        }
        for i in top_indices
    ]


# ── Public API ────────────────────────────────────────────────────────────────
def predict(sentence: str, norm_threshold: float = 0.6) -> dict:
    """
    Full two-stage prediction.

    Stage 1: BERT decides if the sentence is a cultural norm.
    Stage 2: If it is a norm, all three country classifiers predict the country.

    Response schema:
    {
        "sentence": str,
        "is_norm": bool,
        "bert_norm_confidence": float,
        "country_results": {
            "bert":    [...top3...],
            "deberta": [...top3...],
            "roberta": [...top3...]
        } | null
    }
    """
    sentence = sentence.strip()

    # ── Stage 1: BERT norm detection ─────────────────────────────────────────
    norm_result = _norm_predict(sentence)
    is_norm = norm_result["confidence"] >= norm_threshold

    # ── Stage 2: country classification (only if norm) ────────────────────────
    country_results = None
    if is_norm:
        country_results = {}
        for key in ["bert", "deberta", "roberta"]:
            if key in _cache.available_country_models():
                country_results[key] = _country_predict(key, sentence, top_k=3)
            else:
                country_results[key] = []

    return {
        "sentence": sentence,
        "is_norm": is_norm,
        "bert_norm_confidence": norm_result["confidence"],
        "country_results": country_results,
    }


def get_available_models() -> dict:
    return {
        "norm_model": "bert",
        "norm_model_available": _cache.norm_model_available(),
        "country_models": _cache.available_country_models(),
    }
