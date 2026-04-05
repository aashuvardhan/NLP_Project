"""
country_classifier.py
=====================
Stage 2 of the Norm pipeline:  given a sentence that IS a norm,
predict which single country it belongs to.

Combined inference pipeline:
  sentence → NormClassifier → (if norm) → CountryClassifier → country name

HOW TO RUN:
-----------
# Train the country classifier:
    python src/country_classifier.py --mode train

# Evaluate on test set:
    python src/country_classifier.py --mode eval

# Run the full pipeline on custom sentences:
    python src/country_classifier.py --mode predict --sentences "People bow when greeting elders." "Shaking hands is common."

# Train with custom hyperparameters:
    python src/country_classifier.py --mode train --epochs 5 --batch_size 16 --lr 2e-5

# Use large DeBERTa for country classification:
    python src/country_classifier.py --mode train --model deberta_large

OUTPUTS:
--------
  results/country/  -> metrics, confusion matrix, per-class report
  saved_models/country_best/  -> best HuggingFace checkpoint
"""

import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# 1. PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR         = os.path.join(BASE, "data")
RESULTS_DIR      = os.path.join(BASE, "results", "country")
PLOTS_DIR        = os.path.join(RESULTS_DIR, "plots")
COUNTRY_MDL_DIR  = os.path.join(BASE, "saved_models", "country_best")

# Norm classifier checkpoints — one per backbone (matches transformer_model.py naming)
NORM_MODEL_DIRS = {
    "deberta":       os.path.join(BASE, "saved_models", "deberta_best"),
    "bert":          os.path.join(BASE, "saved_models", "bert_best"),
    "roberta":       os.path.join(BASE, "saved_models", "roberta_best"),
    "deberta_large": os.path.join(BASE, "saved_models", "large_models", "deberta_large_best"),
    "roberta_large": os.path.join(BASE, "saved_models", "large_models", "roberta_large_best"),
    "bert_large":    os.path.join(BASE, "saved_models", "large_models", "bert_large_best"),
}
NORM_MDL_DIR = NORM_MODEL_DIRS["deberta"]   # default

for d in [RESULTS_DIR, PLOTS_DIR, COUNTRY_MDL_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL OPTIONS
# ─────────────────────────────────────────────────────────────────────────────
BACKBONE_CHOICES = {
    "deberta":       "microsoft/deberta-v3-base",
    "deberta_large": "microsoft/deberta-v3-large",
    "roberta":       "roberta-base",
    "bert":          "bert-base-uncased",
}

DEFAULT_CFG = {
    "backbone":        "deberta",
    "max_length":      128,
    "batch_size":      16,
    "grad_accumulation": 2,
    "epochs":          5,
    "learning_rate":   2e-5,
    "warmup_ratio":    0.1,
    "weight_decay":    0.01,
    "fp16":            True,
    "seed":            42,
    "patience":        2,
    "min_samples":     30,   # drop countries with fewer than this many labelled norms
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# 3. CULTURAL-GROUP → CANONICAL COUNTRY MAPPING
#    Covers every variant seen in the dataset that maps to a single country.
#    Anything NOT in this map (regional, religious, sub-national ambiguous,
#    generic) is treated as "Unknown" and excluded from training.
# ─────────────────────────────────────────────────────────────────────────────
RAW_TO_COUNTRY = {
    # ── United States ──────────────────────────────────────────────────────
    "American": "United States",
    "Americans": "United States",
    "Californians": "United States",
    "New Yorkers": "United States",
    "Texans": "United States",
    "Southerners": "United States",
    "Southern Americans": "United States",
    "Midwesterners": "United States",
    "Minnesotans": "United States",
    "Floridians": "United States",
    "Southern Californians": "United States",
    "Oregonians": "United States",
    "united_states_of_america": "United States",
    "non-American": "United States",          # norms written from US contrast frame

    # ── United Kingdom ──────────────────────────────────────────────────────
    "British": "United Kingdom",
    "British people": "United Kingdom",
    "Scottish": "United Kingdom",
    "Welsh": "United Kingdom",
    "English": "United Kingdom",
    "Londoners": "United Kingdom",
    "Charlotte residents": "United Kingdom",  # small sample — keep for completeness

    # ── Germany ────────────────────────────────────────────────────────────
    "German": "Germany",
    "Germans": "Germany",
    "germany": "Germany",

    # ── Australia ──────────────────────────────────────────────────────────
    "Australian": "Australia",
    "Australians": "Australia",

    # ── France ─────────────────────────────────────────────────────────────
    "French": "France",
    "French people": "France",
    "france": "France",

    # ── Italy ──────────────────────────────────────────────────────────────
    "Italian": "Italy",
    "Italians": "Italy",
    "Italian and Italian-American": "Italy",
    "italy": "Italy",

    # ── South Korea ────────────────────────────────────────────────────────
    "Korean": "South Korea",
    "South Korean": "South Korea",

    # ── North Korea ────────────────────────────────────────────────────────
    "Democratic People's Republic of Korea": "North Korea",
    "North Korean": "North Korea",

    # ── Netherlands ────────────────────────────────────────────────────────
    "Dutch": "Netherlands",
    "Dutch people": "Netherlands",
    "netherlands": "Netherlands",

    # ── China ──────────────────────────────────────────────────────────────
    "China": "China",
    "Chinese": "China",
    "china": "China",

    # ── Afghanistan ────────────────────────────────────────────────────────
    "Afghanistan": "Afghanistan",
    "afghanistan": "Afghanistan",

    # ── Japan ──────────────────────────────────────────────────────────────
    "Japanese": "Japan",
    "japan": "Japan",

    # ── Spain ──────────────────────────────────────────────────────────────
    "Spanish": "Spain",
    "Spaniards": "Spain",
    "spain": "Spain",

    # ── Sweden ─────────────────────────────────────────────────────────────
    "Swedish": "Sweden",
    "Swedes": "Sweden",
    "sweden": "Sweden",

    # ── Canada ─────────────────────────────────────────────────────────────
    "Canada": "Canada",
    "Canadian": "Canada",
    "Canadians": "Canada",
    "Québécois": "Canada",
    "Qu b cois": "Canada",
    "canada": "Canada",

    # ── Denmark ────────────────────────────────────────────────────────────
    "Denmark": "Denmark",
    "Danish": "Denmark",

    # ── Brazil ─────────────────────────────────────────────────────────────
    "Brazil": "Brazil",
    "Brazilian": "Brazil",
    "Brazilians": "Brazil",
    "brazil": "Brazil",

    # ── Philippines ────────────────────────────────────────────────────────
    "Filipino": "Philippines",
    "Filipinos": "Philippines",
    "philippines": "Philippines",

    # ── Ireland ────────────────────────────────────────────────────────────
    "Irish": "Ireland",
    "ireland": "Ireland",

    # ── Norway ─────────────────────────────────────────────────────────────
    "Norwegian": "Norway",
    "Norwegians": "Norway",

    # ── Russia ─────────────────────────────────────────────────────────────
    "Russians": "Russia",
    "Russian": "Russia",
    "russia": "Russia",

    # ── Mexico ─────────────────────────────────────────────────────────────
    "Mexican": "Mexico",
    "Mexicans": "Mexico",
    "mexico": "Mexico",

    # ── Finland ────────────────────────────────────────────────────────────
    "Finnish": "Finland",
    "Finns": "Finland",

    # ── Indonesia ──────────────────────────────────────────────────────────
    "Indonesian": "Indonesia",
    "Indonesians": "Indonesia",

    # ── Poland ─────────────────────────────────────────────────────────────
    "Polish": "Poland",
    "poland": "Poland",

    # ── Egypt ──────────────────────────────────────────────────────────────
    "Egypt": "Egypt",
    "Egyptians": "Egypt",
    "egypt": "Egypt",

    # ── India ──────────────────────────────────────────────────────────────
    "Indians": "India",
    "Indian": "India",
    "Indian and Indian American": "India",

    # ── New Zealand ────────────────────────────────────────────────────────
    "New Zealanders": "New Zealand",
    "New Zealander": "New Zealand",
    "new_zealand": "New Zealand",

    # ── Switzerland ────────────────────────────────────────────────────────
    "Swiss": "Switzerland",

    # ── Albania ────────────────────────────────────────────────────────────
    "Albania": "Albania",
    "Albanian": "Albania",

    # ── Argentina ──────────────────────────────────────────────────────────
    "Argentina": "Argentina",
    "Argentinian": "Argentina",
    "Argentinians": "Argentina",
    "Argentine": "Argentina",
    "argentina": "Argentina",

    # ── Malaysia ───────────────────────────────────────────────────────────
    "Malaysians": "Malaysia",
    "Malaysian": "Malaysia",
    "malaysia": "Malaysia",

    # ── South Africa ───────────────────────────────────────────────────────
    "South African": "South Africa",
    "south_africa": "South Africa",

    # ── Cambodia ───────────────────────────────────────────────────────────
    "Cambodia": "Cambodia",
    "cambodia": "Cambodia",

    # ── Portugal ───────────────────────────────────────────────────────────
    "Portuguese": "Portugal",
    "portugal": "Portugal",

    # ── Algeria ────────────────────────────────────────────────────────────
    "Algeria": "Algeria",

    # ── Bangladesh ─────────────────────────────────────────────────────────
    "Bangladesh": "Bangladesh",
    "bangladesh": "Bangladesh",

    # ── Greece ─────────────────────────────────────────────────────────────
    "Greek": "Greece",
    "Greeks": "Greece",
    "greece": "Greece",

    # ── Estonia ────────────────────────────────────────────────────────────
    "Estonia": "Estonia",
    "Estonians": "Estonia",

    # ── Singapore ──────────────────────────────────────────────────────────
    "Singaporean": "Singapore",
    "Singaporeans": "Singapore",
    "singapore": "Singapore",

    # ── Bhutan ─────────────────────────────────────────────────────────────
    "Bhutan": "Bhutan",

    # ── Ukraine ────────────────────────────────────────────────────────────
    "Ukrainian": "Ukraine",
    "ukraine": "Ukraine",

    # ── Nigeria ────────────────────────────────────────────────────────────
    "Nigerian": "Nigeria",

    # ── Vietnam ────────────────────────────────────────────────────────────
    "Vietnamese": "Vietnam",
    "vietnam": "Vietnam",

    # ── Bosnia and Herzegovina ─────────────────────────────────────────────
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "bosnia_and_herzegovina": "Bosnia and Herzegovina",

    # ── Angola ─────────────────────────────────────────────────────────────
    "Angola": "Angola",

    # ── Turkey ─────────────────────────────────────────────────────────────
    "Turkish": "Turkey",
    "Turkish people": "Turkey",
    "t rkiye": "Turkey",

    # ── Romania ────────────────────────────────────────────────────────────
    "Romanian": "Romania",
    "Romanians": "Romania",
    "romania": "Romania",

    # ── Belgium ────────────────────────────────────────────────────────────
    "Belgian": "Belgium",
    "Belgians": "Belgium",
    "Belgium": "Belgium",

    # ── Hungary ────────────────────────────────────────────────────────────
    "Hungarians": "Hungary",
    "hungary": "Hungary",

    # ── Ecuador ────────────────────────────────────────────────────────────
    "Ecuador": "Ecuador",

    # ── Austria ────────────────────────────────────────────────────────────
    "Austrian": "Austria",
    "Austrians": "Austria",
    "austria": "Austria",

    # ── Thailand ───────────────────────────────────────────────────────────
    "Thai": "Thailand",
    "Thai people": "Thailand",
    "thailand": "Thailand",

    # ── Croatia ────────────────────────────────────────────────────────────
    "Croatian": "Croatia",
    "croatia": "Croatia",

    # ── Bulgaria ───────────────────────────────────────────────────────────
    "Bulgarian": "Bulgaria",
    "Bulgarians": "Bulgaria",
    "Bulgaria": "Bulgaria",

    # ── Czech Republic ─────────────────────────────────────────────────────
    "Czechs": "Czech Republic",
    "Czech": "Czech Republic",

    # ── Andorra ────────────────────────────────────────────────────────────
    "Andorra": "Andorra",

    # ── Cameroon ───────────────────────────────────────────────────────────
    "Cameroon": "Cameroon",

    # ── Belize ─────────────────────────────────────────────────────────────
    "Belize": "Belize",

    # ── El Salvador ────────────────────────────────────────────────────────
    "El Salvador": "El Salvador",

    # ── Djibouti ───────────────────────────────────────────────────────────
    "Djibouti": "Djibouti",

    # ── Kenya ──────────────────────────────────────────────────────────────
    "Kenyan": "Kenya",
    "kenya": "Kenya",

    # ── Iran ───────────────────────────────────────────────────────────────
    "iran": "Iran",

    # ── Ethiopia ───────────────────────────────────────────────────────────
    "ethiopia": "Ethiopia",

    # ── Somalia ────────────────────────────────────────────────────────────
    "somalia": "Somalia",

    # ── Saudi Arabia ───────────────────────────────────────────────────────
    "saudi_arabia": "Saudi Arabia",

    # ── Myanmar ────────────────────────────────────────────────────────────
    "myanmar": "Myanmar",

    # ── Sudan ──────────────────────────────────────────────────────────────
    "sudan": "Sudan",

    # ── Mauritius ──────────────────────────────────────────────────────────
    "mauritius": "Mauritius",

    # ── Pakistan ───────────────────────────────────────────────────────────
    "pakistan": "Pakistan",

    # ── Samoa ──────────────────────────────────────────────────────────────
    "samoa": "Samoa",

    # ── Tonga ──────────────────────────────────────────────────────────────
    "tonga": "Tonga",

    # ── Laos ───────────────────────────────────────────────────────────────
    "laos": "Laos",

    # ── Israel ─────────────────────────────────────────────────────────────
    "Israeli": "Israel",
    "Israelis": "Israel",
    "israel": "Israel",

    # ── Venezuela ──────────────────────────────────────────────────────────
    "venezuela": "Venezuela",

    # ── Papua New Guinea ───────────────────────────────────────────────────
    "papua_new_guinea": "Papua New Guinea",

    # ── Hong Kong ──────────────────────────────────────────────────────────
    "hong_kong": "Hong Kong",

    # ── Iraq ───────────────────────────────────────────────────────────────
    "iraq": "Iraq",

    # ── Peru ───────────────────────────────────────────────────────────────
    "peru": "Peru",

    # ── Serbia ─────────────────────────────────────────────────────────────
    "Serbian": "Serbia",
    "serbia": "Serbia",

    # ── Cyprus ─────────────────────────────────────────────────────────────
    "cyprus": "Cyprus",

    # ── Colombia ───────────────────────────────────────────────────────────
    "colombia": "Colombia",

    # ── Chile ──────────────────────────────────────────────────────────────
    "Chileans": "Chile",
    "chile": "Chile",

    # ── Belarus ────────────────────────────────────────────────────────────
    "Belarus": "Belarus",

    # ── Sri Lanka ──────────────────────────────────────────────────────────
    "sri_lanka": "Sri Lanka",

    # ── Zimbabwe ───────────────────────────────────────────────────────────
    "zimbabwe": "Zimbabwe",

    # ── Nepal ──────────────────────────────────────────────────────────────
    "nepal": "Nepal",

    # ── Taiwan ─────────────────────────────────────────────────────────────
    "Taiwanese": "Taiwan",
    "taiwan": "Taiwan",

    # ── Dominican Republic ─────────────────────────────────────────────────
    "Dominican Republic": "Dominican Republic",

    # ── North Macedonia ────────────────────────────────────────────────────
    "north_macedonia": "North Macedonia",

    # ── Timor-Leste ────────────────────────────────────────────────────────
    "timor-leste": "Timor-Leste",

    # ── Eritrea ────────────────────────────────────────────────────────────
    "Eritrea": "Eritrea",

    # ── Panama ─────────────────────────────────────────────────────────────
    "Panamanians": "Panama",

    # ── Palestinian Territories ────────────────────────────────────────────
    "palestinian_territories": "Palestine",

    # ── Puerto Rico ────────────────────────────────────────────────────────
    "Puerto Rican": "Puerto Rico",
    "Puerto Ricans": "Puerto Rico",

    # ── Lebanon ────────────────────────────────────────────────────────────
    "Lebanese": "Lebanon",

    # ── Botswana ───────────────────────────────────────────────────────────
    "Botswana": "Botswana",
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
def build_country_dataset(min_samples: int = 30):
    """
    Loads merged_full.csv, keeps only labelled norms (label==1),
    maps cultural_group → canonical country, drops unknowns and rare classes.
    Returns (df_country, label2id, id2label).
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "merged_full.csv"))
    norms = df[df["label"] == 1].copy()

    norms["country"] = norms["cultural_group"].map(RAW_TO_COUNTRY)

    # Drop rows with no mapping (regional, religious, generic)
    norms = norms.dropna(subset=["country"])
    norms = norms[norms["country"] != "none"]

    # Drop countries with too few examples for reliable classification
    counts = norms["country"].value_counts()
    valid_countries = counts[counts >= min_samples].index.tolist()
    norms = norms[norms["country"].isin(valid_countries)].copy()

    print(f"  Country dataset: {len(norms)} sentences across {norms['country'].nunique()} countries")
    print(f"  Dropped {len(df[df['label']==1]) - len(norms)} norms (no country / rare class)")

    # Build label maps (sorted alphabetically for reproducibility)
    countries = sorted(norms["country"].unique().tolist())
    label2id  = {c: i for i, c in enumerate(countries)}
    id2label  = {i: c for c, i in label2id.items()}

    norms["country_id"] = norms["country"].map(label2id)

    print(f"\n  Classes ({len(countries)}):")
    for c in countries:
        print(f"    {c:35s}  {counts.get(c, norms['country'].value_counts().get(c, 0)):>5d} samples")

    return norms[["sentence", "country", "country_id"]].reset_index(drop=True), label2id, id2label


def split_dataset(df):
    """Stratified 80/10/10 split."""
    train_df, temp_df = train_test_split(
        df, test_size=0.20, stratify=df["country_id"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["country_id"], random_state=42
    )
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. DATASET
# ─────────────────────────────────────────────────────────────────────────────
class CountryDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts      = df["sentence"].tolist()
        self.labels     = df["country_id"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return item


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAIN / EVAL LOOPS
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, scaler, cfg):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)
        tt_ids         = batch.get("token_type_ids")
        if tt_ids is not None:
            tt_ids = tt_ids.to(DEVICE)

        with autocast(enabled=cfg["fp16"]):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=tt_ids, labels=labels)
            loss = outputs.loss / cfg["grad_accumulation"]

        scaler.scale(loss).backward()

        if (step + 1) % cfg["grad_accumulation"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * cfg["grad_accumulation"]

    return total_loss / len(loader)


def evaluate(model, loader, id2label, cfg):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)
            tt_ids         = batch.get("token_type_ids")
            if tt_ids is not None:
                tt_ids = tt_ids.to(DEVICE)

            with autocast(enabled=cfg["fp16"]):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=tt_ids, labels=labels)

            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    metrics = {
        "accuracy": round(accuracy_score(all_labels, all_preds), 4),
        "f1_macro": round(f1_score(all_labels, all_preds, average="macro", zero_division=0), 4),
        "f1_weighted": round(f1_score(all_labels, all_preds, average="weighted", zero_division=0), 4),
        "loss": round(total_loss / len(loader), 4),
    }
    return metrics, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(labels, preds, id2label):
    n = len(id2label)
    country_names = [id2label[i] for i in range(n)]
    cm = confusion_matrix(labels, preds)

    fig_size = max(16, n // 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size - 2))
    sns.heatmap(cm, annot=(n <= 30), fmt="d", cmap="Blues",
                xticklabels=country_names, yticklabels=country_names,
                ax=ax, linewidths=0.3)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title("Country Classifier — Confusion Matrix", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "country_confusion_matrix.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Confusion matrix saved -> {path}")


def plot_training_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True)
    axes[1].plot(epochs, history["val_f1_macro"],    "g-o", label="Val F1-Macro")
    axes[1].plot(epochs, history["val_accuracy"],    "m-o", label="Val Accuracy")
    axes[1].set_title("Validation Metrics"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "country_training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Training curves saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train(cfg: dict):
    print("\n=== Building country dataset ===")
    df, label2id, id2label = build_country_dataset(cfg["min_samples"])
    train_df, val_df, test_df = split_dataset(df)

    # Save label maps so inference can load without re-building dataset
    label_map_path = os.path.join(COUNTRY_MDL_DIR, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
    print(f"  Label map saved -> {label_map_path}")

    backbone = BACKBONE_CHOICES[cfg["backbone"]]
    print(f"\n=== Training country classifier: {backbone} ===")
    print(f"  Device: {DEVICE}  |  Classes: {len(label2id)}")
    print(f"  Batch: {cfg['batch_size']}  |  GradAccum: {cfg['grad_accumulation']}"
          f"  |  LR: {cfg['learning_rate']}  |  Epochs: {cfg['epochs']}")

    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    train_loader = DataLoader(CountryDataset(train_df, tokenizer, cfg["max_length"]),
                              batch_size=cfg["batch_size"], shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(CountryDataset(val_df,   tokenizer, cfg["max_length"]),
                              batch_size=cfg["batch_size"]*2, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(CountryDataset(test_df,  tokenizer, cfg["max_length"]),
                              batch_size=cfg["batch_size"]*2, shuffle=False, num_workers=0, pin_memory=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        backbone, num_labels=len(label2id), ignore_mismatched_sizes=True
    ).to(DEVICE).float()

    total_steps  = (len(train_loader) // cfg["grad_accumulation"]) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    optimizer    = torch.optim.AdamW(model.parameters(),
                                     lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = GradScaler(enabled=cfg["fp16"])

    history       = {"train_loss": [], "val_loss": [], "val_f1_macro": [], "val_accuracy": []}
    best_val_f1   = 0.0
    patience_left = cfg["patience"]

    for epoch in range(1, cfg["epochs"] + 1):
        print(f"\n  Epoch {epoch}/{cfg['epochs']}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, cfg)
        val_metrics, _, _ = evaluate(model, val_loader, id2label, cfg)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        print(f"    train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} "
              f"| val_f1_macro={val_metrics['f1_macro']:.4f} | val_acc={val_metrics['accuracy']:.4f}")

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            patience_left = cfg["patience"]
            model.save_pretrained(COUNTRY_MDL_DIR)
            tokenizer.save_pretrained(COUNTRY_MDL_DIR)
            print(f"    ** Best model saved (val_f1={best_val_f1:.4f}) **")
        else:
            patience_left -= 1
            if patience_left == 0:
                print("    Early stopping triggered.")
                break

    # ── Test evaluation ──
    print("\n=== Test Set Evaluation ===")
    best_model = AutoModelForSequenceClassification.from_pretrained(COUNTRY_MDL_DIR).to(DEVICE)
    test_metrics, test_preds, test_labels_list = evaluate(best_model, test_loader, id2label, cfg)

    print(f"  accuracy   : {test_metrics['accuracy']}")
    print(f"  f1_macro   : {test_metrics['f1_macro']}")
    print(f"  f1_weighted: {test_metrics['f1_weighted']}")

    country_names = [id2label[i] for i in range(len(id2label))]
    print(f"\n{classification_report(test_labels_list, test_preds, target_names=country_names, zero_division=0)}")

    plot_training_curves(history)
    plot_confusion_matrix(test_labels_list, test_preds, id2label)

    with open(os.path.join(RESULTS_DIR, "country_results.json"), "w") as f:
        json.dump({"backbone": backbone, "config": cfg,
                   "num_classes": len(label2id),
                   "countries": [id2label[i] for i in range(len(id2label))],
                   "history": history, "test_metrics": test_metrics}, f, indent=2)

    print(f"\nAll outputs saved to {RESULTS_DIR}/")
    return test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# 9. COMBINED INFERENCE PIPELINE
#    Loads the norm classifier + country classifier and runs both on new text.
# ─────────────────────────────────────────────────────────────────────────────
class NormCountryPipeline:
    """
    End-to-end pipeline:
        sentence  →  is_norm (bool)  +  country (str | None)  +  confidences

    Usage:
        pipeline = NormCountryPipeline()
        results  = pipeline.predict(["People bow when greeting.", "The sky is blue."])
        for r in results:
            print(r)
    """

    def __init__(self,
                 norm_model_dir:    str = NORM_MDL_DIR,
                 country_model_dir: str = COUNTRY_MDL_DIR,
                 norm_threshold:    float = 0.5,
                 device:            torch.device = DEVICE):

        self.device         = device
        self.norm_threshold = norm_threshold

        # ── Pre-flight checks ──────────────────────────────────────────────
        if not os.path.isdir(norm_model_dir):
            raise FileNotFoundError(
                f"\nNorm model not found at: {norm_model_dir}\n"
                "Train the norm classifier first:\n"
                "  python src/transformer_model.py --model deberta\n"
                "Or pass the correct path:\n"
                "  python src/country_classifier.py --mode predict "
                "--norm_model_dir /your/path/to/deberta_best"
            )
        if not os.path.isfile(os.path.join(norm_model_dir, "config.json")):
            raise FileNotFoundError(
                f"config.json missing in {norm_model_dir}. "
                "The checkpoint may be incomplete — retrain the norm classifier."
            )

        label_map_path = os.path.join(country_model_dir, "label_map.json")
        if not os.path.isdir(country_model_dir) or not os.path.exists(label_map_path):
            raise FileNotFoundError(
                f"\nCountry model not found at: {country_model_dir}\n"
                "Train the country classifier first:\n"
                "  python src/country_classifier.py --mode train"
            )

        # ── Load norm classifier ───────────────────────────────────────────
        print(f"Loading norm classifier from   : {norm_model_dir}")
        self.norm_tokenizer = AutoTokenizer.from_pretrained(
            norm_model_dir, local_files_only=True)
        self.norm_model     = AutoModelForSequenceClassification.from_pretrained(
            norm_model_dir, local_files_only=True).to(device).eval()

        # ── Load country label map ─────────────────────────────────────────
        with open(label_map_path) as f:
            maps = json.load(f)
        self.id2label = {int(k): v for k, v in maps["id2label"].items()}

        # ── Load country classifier ────────────────────────────────────────
        print(f"Loading country classifier from : {country_model_dir}")
        self.country_tokenizer = AutoTokenizer.from_pretrained(
            country_model_dir, local_files_only=True)
        self.country_model     = AutoModelForSequenceClassification.from_pretrained(
            country_model_dir, local_files_only=True).to(device).eval()

        print(f"Pipeline ready. {len(self.id2label)} country classes | norm threshold={norm_threshold}\n")

    @torch.no_grad()
    def _encode(self, tokenizer, sentences, max_length=128):
        enc = tokenizer(
            sentences,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    @torch.no_grad()
    def predict(self, sentences: list[str]) -> list[dict]:
        """
        Returns a list of dicts, one per sentence:
        {
            "sentence":         str,
            "is_norm":          bool,
            "norm_confidence":  float,   # P(norm)
            "country":          str | None,   # None if not a norm or no country found
            "country_confidence": float | None,
        }
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        # ── Step 1: Norm classification ──
        norm_enc    = self._encode(self.norm_tokenizer, sentences)
        norm_logits = self.norm_model(**norm_enc).logits
        norm_probs  = torch.softmax(norm_logits, dim=-1)[:, 1].cpu().numpy()   # P(norm)
        is_norm     = norm_probs >= self.norm_threshold

        # ── Step 2: Country classification (only for predicted norms) ──
        norm_indices = [i for i, n in enumerate(is_norm) if n]
        country_preds = [None] * len(sentences)
        country_confs = [None] * len(sentences)

        if norm_indices:
            norm_sentences  = [sentences[i] for i in norm_indices]
            country_enc     = self._encode(self.country_tokenizer, norm_sentences)
            country_logits  = self.country_model(**country_enc).logits
            country_probs   = torch.softmax(country_logits, dim=-1).cpu().numpy()
            top_ids         = country_probs.argmax(axis=1)

            for j, orig_i in enumerate(norm_indices):
                country_preds[orig_i] = self.id2label[top_ids[j]]
                country_confs[orig_i] = round(float(country_probs[j, top_ids[j]]), 4)

        results = []
        for i, sent in enumerate(sentences):
            results.append({
                "sentence":           sent,
                "is_norm":            bool(is_norm[i]),
                "norm_confidence":    round(float(norm_probs[i]), 4),
                "country":            country_preds[i],
                "country_confidence": country_confs[i],
            })
        return results

    def predict_one(self, sentence: str) -> dict:
        return self.predict([sentence])[0]


# ─────────────────────────────────────────────────────────────────────────────
# 10. CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Country Classifier for Cultural Norms",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--mode", choices=["train", "eval", "predict"], default="train",
                        help="train | eval | predict")
    parser.add_argument("--model", choices=list(BACKBONE_CHOICES.keys()), default="deberta",
                        help="Backbone model (default: deberta)")
    parser.add_argument("--epochs",           type=int,   default=DEFAULT_CFG["epochs"])
    parser.add_argument("--batch_size",       type=int,   default=DEFAULT_CFG["batch_size"])
    parser.add_argument("--grad_accumulation",type=int,   default=DEFAULT_CFG["grad_accumulation"])
    parser.add_argument("--lr",               type=float, default=DEFAULT_CFG["learning_rate"])
    parser.add_argument("--max_length",       type=int,   default=DEFAULT_CFG["max_length"])
    parser.add_argument("--min_samples",      type=int,   default=DEFAULT_CFG["min_samples"])
    parser.add_argument("--norm_threshold",   type=float, default=0.6,
                        help="Confidence threshold for norm classification (default: 0.6)")
    parser.add_argument("--norm_model",       type=str,   default="deberta",
                        choices=list(NORM_MODEL_DIRS.keys()),
                        help="Which norm classifier checkpoint to use (default: deberta).\n"
                             "  deberta        -> saved_models/deberta_best\n"
                             "  bert           -> saved_models/bert_best\n"
                             "  roberta        -> saved_models/roberta_best\n"
                             "  deberta_large  -> saved_models/large_models/deberta_large_best\n"
                             "  roberta_large  -> saved_models/large_models/roberta_large_best\n"
                             "  bert_large     -> saved_models/large_models/bert_large_best")
    parser.add_argument("--norm_model_dir",   type=str,   default=None,
                        help="Explicit path to norm classifier checkpoint (overrides --norm_model).\n"
                             "Use when your Colab path differs from the default, e.g.:\n"
                             "  --norm_model_dir /content/NLP_Project/saved_models/deberta_best")
    parser.add_argument("--no_fp16",          action="store_true")
    parser.add_argument("--patience",         type=int,   default=DEFAULT_CFG["patience"])
    parser.add_argument("--sentences", nargs="+",
                        help="Sentences to run through the pipeline (--mode predict)")
    args = parser.parse_args()

    if args.mode == "train":
        cfg = DEFAULT_CFG.copy()
        cfg["backbone"]          = args.model
        cfg["epochs"]            = args.epochs
        cfg["batch_size"]        = args.batch_size
        cfg["grad_accumulation"] = args.grad_accumulation
        cfg["learning_rate"]     = args.lr
        cfg["max_length"]        = args.max_length
        cfg["min_samples"]       = args.min_samples
        cfg["fp16"]              = not args.no_fp16
        cfg["patience"]          = args.patience

        print(f"Device : {DEVICE}")
        if DEVICE.type == "cuda":
            print(f"GPU    : {torch.cuda.get_device_name(0)}")
        train(cfg)

    elif args.mode == "eval":
        # Re-run evaluation on test split without retraining
        cfg = DEFAULT_CFG.copy()
        cfg["fp16"] = not args.no_fp16
        print("Building dataset for evaluation...")
        df, label2id, id2label = build_country_dataset(args.min_samples)
        _, _, test_df = split_dataset(df)

        tokenizer = AutoTokenizer.from_pretrained(COUNTRY_MDL_DIR, local_files_only=True)
        test_loader = DataLoader(
            CountryDataset(test_df, tokenizer, args.max_length),
            batch_size=32, shuffle=False, num_workers=0,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            COUNTRY_MDL_DIR, local_files_only=True).to(DEVICE)
        metrics, preds, labels_list = evaluate(model, test_loader, id2label, cfg)
        country_names = [id2label[i] for i in range(len(id2label))]
        print(f"\nAccuracy   : {metrics['accuracy']}")
        print(f"F1-macro   : {metrics['f1_macro']}")
        print(f"F1-weighted: {metrics['f1_weighted']}")
        print(f"\n{classification_report(labels_list, preds, target_names=country_names, zero_division=0)}")

    elif args.mode == "predict":
        sentences = args.sentences or [
            "People bow deeply when greeting their elders.",
            "It is common to remove your shoes before entering a home.",
            "The mitochondria is the powerhouse of the cell.",
            "Guests are always offered tea upon arrival.",
            "In a store, people wait in line patiently.",
        ]
        # --norm_model_dir (explicit path) takes priority over --norm_model (name)
        norm_dir = args.norm_model_dir or NORM_MODEL_DIRS[args.norm_model]
        pipeline = NormCountryPipeline(
            norm_model_dir=norm_dir,
            norm_threshold=args.norm_threshold,
        )
        results  = pipeline.predict(sentences)

        print(f"\n{'─'*80}")
        print(f"{'SENTENCE':<45} {'NORM':>6}  {'CONF':>6}  {'COUNTRY':<30} {'CONFIDENCE SCORE':>6}")
        print(f"{'─'*80}")
        for r in results:
            snippet  = r["sentence"][:44]
            norm_str = "YES" if r["is_norm"] else "NO"
            country  = r["country"] or "—"
            c_conf   = f"{r['country_confidence']:.3f}" if r["country_confidence"] else "—"
            print(f"{snippet:<45} {norm_str:>6}  {r['norm_confidence']:.3f}  {country:<30} {c_conf:>6}")
        print(f"{'─'*80}\n")


if __name__ == "__main__":
    main()
