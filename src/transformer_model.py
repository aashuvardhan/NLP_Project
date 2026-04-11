"""
transformer_model.py
====================
Fine-tunes a transformer model for binary Norm classification.

HOW TO USE:
-----------
# Train a specific model:
    python src/transformer_model.py --model deberta
    python src/transformer_model.py --model bert
    python src/transformer_model.py --model roberta

# Train all models sequentially:
    python src/transformer_model.py --model all

# Override hyperparameters from command line:
    python src/transformer_model.py --model bert --epochs 3 --lr 3e-5 --batch_size 8

SUPPORTED MODELS (add more to the MODELS dict below):
------------------------------------------------------
    "deberta"  -> microsoft/deberta-v3-base   (~86M params, best accuracy)
    "bert"     -> bert-base-uncased           (~110M params)
    "roberta"  -> roberta-base                (~125M params)
    -- just add any HuggingFace model ID to MODELS dict --
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
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# 1. PATHS  (auto-resolved relative to this file)
# ─────────────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE, "data")
RESULTS_DIR = os.path.join(BASE, "results")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
MODELS_DIR  = os.path.join(BASE, "saved_models")

for d in [RESULTS_DIR, PLOTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL REGISTRY  ← add / remove models here
# ─────────────────────────────────────────────────────────────────────────────
MODELS = {
    "deberta":  "microsoft/deberta-v3-base",
    "bert":     "bert-base-uncased",
    "roberta":  "roberta-base",
    # "albert": "albert-base-v2",         # uncomment to add
    # "xlnet":  "xlnet-base-cased",       # uncomment to add
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. DEFAULT HYPERPARAMETERS  ← change these for your experiments
#    All of these can also be overridden via CLI flags (see main() below)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CFG = {
    "max_length":        128,    # token length per sentence
    "batch_size":        16,     # per-GPU batch size (lower if OOM: try 8)
    "grad_accumulation": 2,      # effective batch = batch_size * grad_accumulation
    "epochs":            10,     # max training epochs
    "learning_rate":     2e-5,   # AdamW learning rate
    "warmup_ratio":      0.1,    # fraction of steps used for LR warmup
    "weight_decay":      0.01,
    "fp16":              True,   # mixed precision (set False if GPU issues)
    "seed":              42,
    "patience":          2,      # early stopping: stop after N epochs no improvement
}

# Per-model overrides — merged on top of DEFAULT_CFG before training.
MODEL_CFG_OVERRIDES = {
    "bert": {
        "epochs":        5,
        "learning_rate": 1e-5,       # lower LR — BERT overfits faster than DeBERTa
        "warmup_ratio":  0.2,        # longer warmup to stabilize early training
        "weight_decay":  0.05,       # stronger regularization to reduce overconfidence
        "patience":      3,          # give more epochs before early-stopping
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 4. DATASET
# ─────────────────────────────────────────────────────────────────────────────
class NormDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.texts      = df["sentence"].tolist()
        self.labels     = df["label"].tolist()
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
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(preds, labels):
    return {
        "accuracy":  round(accuracy_score(labels, preds), 4),
        "f1":        round(f1_score(labels, preds, average="macro"), 4),
        "precision": round(precision_score(labels, preds, average="macro", zero_division=0), 4),
        "recall":    round(recall_score(labels, preds, average="macro",    zero_division=0), 4),
    }


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

        with autocast(enabled=cfg["fp16"]):
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
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


def evaluate(model, loader, cfg):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            with autocast(enabled=cfg["fp16"]):
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)

            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = round(total_loss / len(loader), 4)
    return metrics, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(history, model_key):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss")
    axes[0].set_title(f"{model_key.upper()} - Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, history["val_f1"],       "g-o", label="Val F1")
    axes[1].plot(epochs, history["val_accuracy"], "m-o", label="Val Accuracy")
    axes[1].set_title(f"{model_key.upper()} - Validation Metrics"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{model_key}_training_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Plot saved -> {path}")


def plot_confusion_matrix(labels, preds, model_key):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Norm", "Norm"],
                yticklabels=["Non-Norm", "Norm"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{model_key.upper()} - Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{model_key}_confusion_matrix.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Plot saved -> {path}")


def save_comparison_chart(all_results):
    rows = [{"Model": k.upper(), **v} for k, v in all_results.items()]
    df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    df.to_csv(os.path.join(RESULTS_DIR, "metrics_comparison.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df)); w = 0.20
    for i, (col, color) in enumerate(zip(
        ["accuracy", "f1", "precision", "recall"],
        ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    )):
        ax.bar(x + i*w, df[col], width=w, label=col.capitalize(), color=color)
    ax.set_xticks(x + w*1.5); ax.set_xticklabels(df["Model"])
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("Score")
    ax.set_title("Model Comparison - Norm Classifier")
    ax.legend(); ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Comparison chart saved -> {path}")
    print(df[["Model", "accuracy", "f1", "precision", "recall"]].to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model_key: str, train_df, val_df, test_df, cfg: dict):
    # Apply any per-model hyperparameter overrides on top of the base config
    cfg = cfg.copy()
    if model_key in MODEL_CFG_OVERRIDES:
        cfg.update(MODEL_CFG_OVERRIDES[model_key])
        print(f"  [INFO] Applying {model_key} overrides: {MODEL_CFG_OVERRIDES[model_key]}")

    model_name = MODELS[model_key]
    print(f"\n{'='*60}")
    print(f"  Model   : {model_name}")
    print(f"  Device  : {DEVICE}")
    print(f"  Epochs  : {cfg['epochs']}  |  Batch: {cfg['batch_size']}  |  LR: {cfg['learning_rate']}")
    print(f"{'='*60}")

    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_loader = DataLoader(NormDataset(train_df, tokenizer, cfg["max_length"]),
                              batch_size=cfg["batch_size"], shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(NormDataset(val_df,   tokenizer, cfg["max_length"]),
                              batch_size=cfg["batch_size"]*2, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(NormDataset(test_df,  tokenizer, cfg["max_length"]),
                              batch_size=cfg["batch_size"]*2, shuffle=False, num_workers=0, pin_memory=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True
    ).to(DEVICE).float()  # DeBERTa-v3 can load some params as FP16; force FP32 so GradScaler works

    total_steps  = (len(train_loader) // cfg["grad_accumulation"]) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    optimizer    = torch.optim.AdamW(model.parameters(),
                                     lr=cfg["learning_rate"],
                                     weight_decay=cfg["weight_decay"])
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = GradScaler(enabled=cfg["fp16"])

    history       = {"train_loss": [], "val_loss": [], "val_f1": [], "val_accuracy": []}
    best_val_f1   = 0.0
    patience_left = cfg["patience"]
    best_ckpt     = os.path.join(MODELS_DIR, f"{model_key}_best")

    for epoch in range(1, cfg["epochs"] + 1):
        print(f"\n  Epoch {epoch}/{cfg['epochs']}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, cfg)
        val_metrics, _, _ = evaluate(model, val_loader, cfg)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        print(f"    train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} "
              f"| val_f1={val_metrics['f1']:.4f} | val_acc={val_metrics['accuracy']:.4f}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            patience_left = cfg["patience"]
            model.save_pretrained(best_ckpt)
            tokenizer.save_pretrained(best_ckpt)
            print(f"    ** Best model saved (val_f1={best_val_f1:.4f}) **")
        else:
            patience_left -= 1
            if patience_left == 0:
                print("    Early stopping triggered.")
                break

    # ── Test evaluation ──
    print(f"\n  Loading best checkpoint for test set evaluation...")
    best_model = AutoModelForSequenceClassification.from_pretrained(best_ckpt).to(DEVICE)
    test_metrics, test_preds, test_labels = evaluate(best_model, test_loader, cfg)

    print(f"\n  TEST RESULTS [{model_key.upper()}]")
    for k, v in test_metrics.items():
        if k != "loss":
            print(f"    {k:12s}: {v}")
    print(f"\n{classification_report(test_labels, test_preds, target_names=['Non-Norm', 'Norm'])}")

    plot_training_curves(history, model_key)
    plot_confusion_matrix(test_labels, test_preds, model_key)

    with open(os.path.join(RESULTS_DIR, f"{model_key}_results.json"), "w") as f:
        json.dump({"model": model_key, "model_name": model_name,
                   "config": cfg, "history": history,
                   "test_metrics": test_metrics}, f, indent=2)

    del model, best_model
    torch.cuda.empty_cache()
    return test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# 9. CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Norm Classifier - Transformer Fine-tuning")
    parser.add_argument("--model",       default="deberta",
                        choices=list(MODELS.keys()) + ["all"],
                        help="Model to train (default: deberta). Use 'all' for all models.")
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_CFG["epochs"])
    parser.add_argument("--batch_size",  type=int,   default=DEFAULT_CFG["batch_size"])
    parser.add_argument("--lr",          type=float, default=DEFAULT_CFG["learning_rate"])
    parser.add_argument("--max_length",  type=int,   default=DEFAULT_CFG["max_length"])
    parser.add_argument("--no_fp16",     action="store_true", help="Disable mixed precision")
    parser.add_argument("--patience",    type=int,   default=DEFAULT_CFG["patience"])
    args = parser.parse_args()

    cfg = DEFAULT_CFG.copy()
    cfg["epochs"]        = args.epochs
    cfg["batch_size"]    = args.batch_size
    cfg["learning_rate"] = args.lr
    cfg["max_length"]    = args.max_length
    cfg["patience"]      = args.patience
    cfg["fp16"]          = not args.no_fp16

    print(f"\nDevice : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    print("\nLoading datasets...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]
    all_results   = {}

    for mk in models_to_run:
        all_results[mk] = train_model(mk, train_df, val_df, test_df, cfg)

    if len(all_results) > 1:
        save_comparison_chart(all_results)

    print("\nDone. All results in PROJECT_P3/results/")


if __name__ == "__main__":
    main()
