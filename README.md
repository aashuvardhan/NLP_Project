# P3: Norm Classifier

A transformer-based classifier that distinguishes **cultural norms** (behavioural/social expectations) from **generic sentences** (factual/observational statements).

**Bonus:** Identifies which culture a norm belongs to (USA, India, Japan, etc.)

---

## Project Structure

```
PROJECT_P3/
├── data/
│   ├── raw/                         # original source files (do not modify)
│   │   ├── culturebank_reddit.csv
│   │   ├── culturebank_tiktok.csv
│   │   ├── generic_kb_simplewiki.parquet
│   │   └── generics_kb_best.parquet
│   ├── merged_full.csv              # complete merged dataset (36,062 rows)
│   ├── train.csv                    # 80% — 28,849 rows
│   ├── val.csv                      # 10% —  3,606 rows
│   └── test.csv                     # 10% —  3,607 rows
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA — plots, stats, word clouds
│   ├── 02_transformer_norm_classifier.ipynb   # results analysis (after Phase 2)
│   └── 03_culture_classifier.ipynb            # bonus culture label classifier (Phase 3)
├── src/
│   ├── data_prep.py                 # builds the dataset from raw sources
│   ├── transformer_model.py         # Phase 1 — norm classifier (deberta/bert/roberta base)
│   ├── large_models.py              # Phase 2 — large model variants (deberta/bert/roberta large)
│   ├── country_classifier.py        # Phase 2 — country classifier + combined inference pipeline
│   ├── error_analysis.py            # error analysis on trained norm classifier
│   └── threshold_tuning.py          # threshold sweep (0.30→0.80) to reduce false positives
├── results/
│   ├── plots/                       # training curves, confusion matrices (auto-generated)
│   ├── large_models/                # large model results & plots (auto-generated)
│   ├── country/                     # country classifier results & plots (auto-generated)
│   ├── *_results.json               # per-model metrics & history (auto-generated)
│   └── metrics_comparison.csv       # side-by-side comparison (auto-generated)
├── saved_models/
│   ├── {model}_best/                # best norm classifier checkpoint (auto-generated)
│   ├── large_models/                # large model checkpoints (auto-generated)
│   └── country_best/                # country classifier checkpoint (auto-generated)
└── README.md
```

---

## Quick Start Order

```bash
# Step 1 — Build dataset (already done, skip if data/ folder exists)
python src/data_prep.py

# Step 2 — Explore the data (open in Jupyter)
notebooks/01_data_exploration.ipynb

# Step 3 — Train your assigned model
python src/transformer_model.py --model deberta

# On Google Colab
!python src/transformer_model.py --model deberta
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentencepiece` and `protobuf` are required for DeBERTa.

### 2. Verify GPU (recommended)

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Running the Models

All team members run from the `PROJECT_P3/` directory.

### Train your assigned model

```bash
# Team member 1 — DeBERTa-v3 (recommended, best accuracy)
python src/transformer_model.py --model deberta

# Team member 2 — BERT
python src/transformer_model.py --model bert

# Team member 3 — RoBERTa
python src/transformer_model.py --model roberta
```

**On Google Colab** (prefix with `!`):

```python
# DeBERTa-v3
!python src/transformer_model.py --model deberta

# BERT
!python src/transformer_model.py --model bert

# RoBERTa
!python src/transformer_model.py --model roberta
```

### Train all models on one machine

```bash
python src/transformer_model.py --model all
```

**On Google Colab:**

```python
!python src/transformer_model.py --model all
```

---

## Hyperparameter Flags

All defaults are pre-tuned for a 2 GB GPU. Override via CLI as needed:

| Flag | Default | Description |
|---|---|---|
| `--model` | `deberta` | Model to train: `deberta`, `bert`, `roberta`, `all` |
| `--epochs` | `4` | Max training epochs |
| `--batch_size` | `16` | Per-GPU batch size |
| `--lr` | `2e-5` | Learning rate |
| `--max_length` | `128` | Max token length |
| `--patience` | `2` | Early stopping patience |
| `--no_fp16` | off | Disable mixed precision (if GPU errors occur) |

**Examples:**

```bash
# More epochs, smaller batch (for low VRAM GPUs)
python src/transformer_model.py --model bert --epochs 5 --batch_size 8

# Higher learning rate experiment
python src/transformer_model.py --model roberta --lr 3e-5

# If you get CUDA out-of-memory errors
python src/transformer_model.py --model deberta --batch_size 8 --no_fp16
```

---

## Outputs

After training, results are saved automatically:

| Output | Location | Description |
|---|---|---|
| Best model weights | `saved_models/{model}_best/` | Loadable HuggingFace checkpoint |
| Metrics (JSON) | `results/{model}_results.json` | Accuracy, F1, precision, recall + full history |
| Training curves | `results/plots/{model}_training_curves.png` | Loss & F1 per epoch |
| Confusion matrix | `results/plots/{model}_confusion_matrix.png` | Test set predictions |
| Comparison table | `results/metrics_comparison.csv` | All models side-by-side (when running `all`) |
| Comparison chart | `results/plots/model_comparison.png` | Bar chart of all models |

---

## Adding a New Model

Open `src/transformer_model.py` and add one line to the `MODELS` dictionary:

```python
MODELS = {
    "deberta":  "microsoft/deberta-v3-base",
    "bert":     "bert-base-uncased",
    "roberta":  "roberta-base",
    "mymodel":  "any-huggingface-model-id",   # <-- add here
}
```

Then run:
```bash
python src/transformer_model.py --model mymodel
```

---

## Dataset Summary

| Split | Total | Norm (1) | Non-Norm (0) |
|---|---|---|---|
| train.csv | 28,849 | 14,425 | 14,424 |
| val.csv | 3,606 | 1,803 | 1,803 |
| test.csv | 3,607 | 1,803 | 1,804 |

**Sources used:**
- **Norm:** CultureBank Reddit + TikTok (filtered: agreement >= 0.70)
- **Non-Norm:** Wikipedia (~20–25K sentences, keyword-filtered to remove cultural/behavioral sentences)

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Add `--batch_size 8` or `--no_fp16` |
| `ModuleNotFoundError: sentencepiece` | `pip install sentencepiece protobuf` |
| `ModuleNotFoundError: seaborn` | `pip install seaborn` |
| Model downloads slowly | Models are cached after first download in `~/.cache/huggingface/` |
| Results not saving | Make sure you're running from inside `PROJECT_P3/` directory |

---

---

# Phase 2 — Country Prediction Pipeline

Extends the Phase 1 norm classifier with a second stage: **given a sentence that is a norm, predict which single country it belongs to.**

```
sentence → NormClassifier (threshold 0.6) → is_norm?
                                  YES → CountryClassifier → country name + confidence
                                  NO  → country = None
```

---

## Phase 2 Dataset

The country classifier trains only on **norm sentences that have a country label**. Non-norms and normbank entries (which have no country) are excluded.

| Stat | Value |
|---|---|
| Training sentences | 13,342 |
| Countries (classes) | 56 |
| Norm threshold (default) | 0.6 |

**Sources with country labels:** CultureBank Reddit, CultureBank TikTok, NormAD, CultureAtlas
**Sources without country labels (excluded):** NormBank (8,000 norms) — used only in norm classifier

---

## Phase 2 Quick Start (Full Pipeline — BERT Example)

```bash
# Step 1 — Build dataset (skip if already done)
python src/data_prep.py

# Step 2 — Train norm classifier
python src/transformer_model.py --model bert

# Step 3 — Train country classifier
python src/country_classifier.py --mode train --model bert

# Step 4 — Run the full pipeline
python src/country_classifier.py --mode predict \
  --norm_model bert \
  --sentences "In Japan, people bow when greeting someone as a sign of respect." \
              "The sky is blue."
```

**On Google Colab:**

```python
# Step 1
!python src/data_prep.py

# Step 2
!python src/transformer_model.py --model bert

# Step 3
!python src/country_classifier.py --mode train --model bert

# Step 4
!python src/country_classifier.py --mode predict \
  --norm_model bert \
  --sentences "In Japan, people bow when greeting someone as a sign of respect." \
              "The sky is blue."
```

---

## Country Classifier Commands

### Train

```bash
# Train with DeBERTa backbone (default)
python src/country_classifier.py --mode train

# Train with BERT backbone
python src/country_classifier.py --mode train --model bert

# Train with RoBERTa backbone
python src/country_classifier.py --mode train --model roberta

# Train with DeBERTa-large backbone
python src/country_classifier.py --mode train --model deberta_large --batch_size 8 --grad_accumulation 4
```

### Evaluate

```bash
# Re-run test set evaluation without retraining
python src/country_classifier.py --mode eval
```

### Predict

```bash
# Default (uses deberta norm classifier, threshold 0.6)
python src/country_classifier.py --mode predict \
  --sentences "In India, people touch the feet of elders as a sign of respect." \
              "The Himalayas are the tallest mountain range."

# Specify which norm classifier to use
python src/country_classifier.py --mode predict --norm_model bert \
  --sentences "In Japan, people avoid talking on the phone on public transport."

# Override norm threshold
python src/country_classifier.py --mode predict --norm_threshold 0.7 \
  --sentences "In China, people present business cards with both hands."

# Override norm model path explicitly (useful on Colab with custom paths)
python src/country_classifier.py --mode predict \
  --norm_model_dir /content/NLP_Project/saved_models/bert_best \
  --sentences "In America, people tip servers 15 to 20 percent after a meal."
```

---

## Country Classifier Hyperparameter Flags

| Flag | Default | Description |
|---|---|---|
| `--mode` | `train` | `train`, `eval`, or `predict` |
| `--model` | `deberta` | Country classifier backbone: `deberta`, `bert`, `roberta`, `deberta_large`, `roberta_large`, `bert_large` |
| `--norm_model` | `deberta` | Which norm classifier checkpoint to use at prediction time |
| `--norm_model_dir` | auto | Explicit path to norm classifier checkpoint (overrides `--norm_model`) |
| `--norm_threshold` | `0.6` | Minimum confidence to classify a sentence as a norm |
| `--epochs` | `5` | Max training epochs |
| `--batch_size` | `16` | Per-GPU batch size |
| `--grad_accumulation` | `2` | Gradient accumulation steps |
| `--lr` | `2e-5` | Learning rate |
| `--max_length` | `128` | Max token length |
| `--min_samples` | `30` | Drop countries with fewer than this many labelled norms |
| `--patience` | `2` | Early stopping patience |

---

## Large Model Variants

`src/large_models.py` runs the same pipeline with larger checkpoints (~340–400M params). Use when you want to compare large vs base model performance.

| Model | Checkpoint | ~VRAM |
|---|---|---|
| `deberta_large` | microsoft/deberta-v3-large | 14 GB (batch=8) |
| `roberta_large` | roberta-large | 12 GB (batch=8) |
| `bert_large` | bert-large-uncased | 11 GB (batch=8) |

```bash
# Train DeBERTa-large norm classifier (recommended first)
python src/large_models.py --model deberta_large

# Train RoBERTa-large
python src/large_models.py --model roberta_large

# Train BERT-large
python src/large_models.py --model bert_large

# Train all three sequentially
python src/large_models.py --model all

# Low VRAM (T4 16 GB) — reduce batch size
python src/large_models.py --model deberta_large --batch_size 4 --grad_accumulation 8
```

Large model results save to `results/large_models/` and checkpoints to `saved_models/large_models/` — separate from base model outputs.

---

## Phase 2 Outputs

| Output | Location | Description |
|---|---|---|
| Country model weights | `saved_models/country_best/` | Best HuggingFace checkpoint |
| Label map | `saved_models/country_best/label_map.json` | `id → country` mapping (required for inference) |
| Metrics (JSON) | `results/country/country_results.json` | Accuracy, F1-macro, F1-weighted + history |
| Training curves | `results/country/plots/country_training_curves.png` | Loss & F1 per epoch |
| Confusion matrix | `results/country/plots/country_confusion_matrix.png` | 56-class test set predictions |
| Large model results | `results/large_models/` | Per-model JSON + plots |
| Large model comparison | `results/large_models/large_models_comparison.csv` | Side-by-side (when running `all`) |

---

## Phase 2 Troubleshooting

| Problem | Fix |
|---|---|
| `Norm model not found` | Run `python src/transformer_model.py --model bert` first |
| `Country model not found` | Run `python src/country_classifier.py --mode train` first |
| `label_map.json not found` | Country model directory is incomplete — retrain with `--mode train` |
| Wrong norm model loaded | Use `--norm_model bert/roberta/deberta` to match what you trained |
| Path mismatch on Colab | Use `--norm_model_dir /full/path/to/checkpoint` to override |
| Low country confidence | Expected for cross-cultural norms (e.g. "use right hand for food") — shared across many countries |
| False positives (non-norms predicted as norms) | Raise `--norm_threshold` to `0.65` or `0.7` |
