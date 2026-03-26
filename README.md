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
│   └── transformer_model.py         # fine-tuning & evaluation for all models
├── results/
│   ├── plots/                       # training curves, confusion matrices (auto-generated)
│   ├── *_results.json               # per-model metrics & history (auto-generated)
│   └── metrics_comparison.csv       # side-by-side comparison (auto-generated)
├── saved_models/
│   └── {model}_best/                # best checkpoint per model (auto-generated)
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
```

---

## Setup

### 1. Install dependencies

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn sentencepiece protobuf
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

### Train all models on one machine

```bash
python src/transformer_model.py --model all
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
- **Non-Norm:** GenericsKB SimpleWiki + Best (filtered: quality score >= 0.55)

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Add `--batch_size 8` or `--no_fp16` |
| `ModuleNotFoundError: sentencepiece` | `pip install sentencepiece protobuf` |
| `ModuleNotFoundError: seaborn` | `pip install seaborn` |
| Model downloads slowly | Models are cached after first download in `~/.cache/huggingface/` |
| Results not saving | Make sure you're running from inside `PROJECT_P3/` directory |
