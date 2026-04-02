"""
large_models.py
===============
Fine-tunes LARGE transformer variants for binary Norm classification.
Mirrors transformer_model.py but uses bigger checkpoints with
memory-adjusted hyperparameters (smaller batch, higher grad accumulation).

HOW TO RUN (on Colab / any GPU machine):
-----------------------------------------
# DeBERTa-v3-large  (~400M params)  ← recommended first run
    python src/large_models.py --model deberta_large

# RoBERTa-large  (~355M params)
    python src/large_models.py --model roberta_large

# BERT-large-uncased  (~340M params)
    python src/large_models.py --model bert_large

# All three sequentially (saves a comparison chart at the end)
    python src/large_models.py --model all

# Override hyperparameters
    python src/large_models.py --model deberta_large --epochs 5 --lr 1e-5 --batch_size 4

COLAB QUICK-START (paste into a cell):
----------------------------------------
    !git clone <your-repo-url>
    %cd PROJECT_P3
    !pip install transformers datasets sentencepiece protobuf -q
    !python src/large_models.py --model deberta_large

MEMORY GUIDE (A100 40 GB is ideal; T4 16 GB works with batch_size=4):
-----------------------------------------------------------------------
    Model                  | ~VRAM at batch=8 | ~VRAM at batch=4
    deberta-v3-large       |      ~14 GB       |       ~8 GB
    roberta-large          |      ~12 GB       |       ~7 GB
    bert-large-uncased     |      ~11 GB       |       ~6 GB

If you hit OOM:  --batch_size 4 --grad_accumulation 8
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
# 1. PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE, "data")
RESULTS_DIR = os.path.join(BASE, "results", "large_models")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
MODELS_DIR  = os.path.join(BASE, "saved_models", "large_models")

for d in [RESULTS_DIR, PLOTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
MODELS = {
    "deberta_large": "microsoft/deberta-v3-large",   # ~400M params — best expected accuracy
    "roberta_large": "roberta-large",                 # ~355M params
    "bert_large":    "bert-large-uncased",            # ~340M params
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. DEFAULT HYPERPARAMETERS
#    Compared to base models:
#      batch_size 8  (was 16) — large models use ~2x VRAM per sample
#      grad_accumulation 4   (was 2)  — keeps effective batch = 32, same as base
#      learning_rate 1e-5    (was 2e-5) — lower LR stabilises large-model fine-tuning
#      max_length 128        — unchanged; raise to 256 only if your norms are long
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CFG = {
    "max_length":        128,
    "batch_size":        8,     # drop to 4 if OOM on T4
    "grad_accumulation": 4,     # effective batch = 32
    "epochs":            4,
    "learning_rate":     1e-5,
    "warmup_ratio":      0.1,
    "weight_decay":      0.01,
    "fp16":              True,
    "seed":              42,
    "patience":          2,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 4. DATASET  (identical to transformer_model.py)
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
        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }
        # token_type_ids only present for BERT-family
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return item


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
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(DEVICE)

        with autocast(enabled=cfg["fp16"]):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
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
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(DEVICE)

            with autocast(enabled=cfg["fp16"]):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                )

            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = round(total_loss / len(loader), 4)
    return metrics, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# 7. PLOTS  (tagged with "large_" prefix to avoid overwriting base results)
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
    df.to_csv(os.path.join(RESULTS_DIR, "large_models_comparison.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df)); w = 0.20
    for i, (col, color) in enumerate(zip(
        ["accuracy", "f1", "precision", "recall"],
        ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    )):
        ax.bar(x + i*w, df[col], width=w, label=col.capitalize(), color=color)
    ax.set_xticks(x + w*1.5); ax.set_xticklabels(df["Model"], rotation=15)
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("Score")
    ax.set_title("Large Model Comparison - Norm Classifier")
    ax.legend(); ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "large_models_comparison.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Comparison chart saved -> {path}")
    print(df[["Model", "accuracy", "f1", "precision", "recall"]].to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model_key: str, train_df, val_df, test_df, cfg: dict):
    model_name = MODELS[model_key]
    print(f"\n{'='*65}")
    print(f"  Model   : {model_name}")
    print(f"  Device  : {DEVICE}")
    print(f"  Epochs  : {cfg['epochs']}  |  Batch: {cfg['batch_size']}  "
          f"|  GradAccum: {cfg['grad_accumulation']}  |  LR: {cfg['learning_rate']}")
    print(f"  Effective batch size: {cfg['batch_size'] * cfg['grad_accumulation']}")
    print(f"{'='*65}")

    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_loader = DataLoader(
        NormDataset(train_df, tokenizer, cfg["max_length"]),
        batch_size=cfg["batch_size"], shuffle=True, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        NormDataset(val_df, tokenizer, cfg["max_length"]),
        batch_size=cfg["batch_size"] * 2, shuffle=False, num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        NormDataset(test_df, tokenizer, cfg["max_length"]),
        batch_size=cfg["batch_size"] * 2, shuffle=False, num_workers=0, pin_memory=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True
    ).to(DEVICE).float()

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

    result_path = os.path.join(RESULTS_DIR, f"{model_key}_results.json")
    with open(result_path, "w") as f:
        json.dump({
            "model": model_key,
            "model_name": model_name,
            "config": cfg,
            "history": history,
            "test_metrics": test_metrics,
        }, f, indent=2)
    print(f"  Results saved -> {result_path}")

    del model, best_model
    torch.cuda.empty_cache()
    return test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# 9. CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Norm Classifier — Large Model Fine-tuning",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model", default="deberta_large",
        choices=list(MODELS.keys()) + ["all"],
        help=(
            "Model to train (default: deberta_large).\n"
            "  deberta_large  -> microsoft/deberta-v3-large\n"
            "  roberta_large  -> roberta-large\n"
            "  bert_large     -> bert-large-uncased\n"
            "  all            -> run all three sequentially"
        ),
    )
    parser.add_argument("--epochs",           type=int,   default=DEFAULT_CFG["epochs"])
    parser.add_argument("--batch_size",       type=int,   default=DEFAULT_CFG["batch_size"])
    parser.add_argument("--grad_accumulation",type=int,   default=DEFAULT_CFG["grad_accumulation"])
    parser.add_argument("--lr",               type=float, default=DEFAULT_CFG["learning_rate"])
    parser.add_argument("--max_length",       type=int,   default=DEFAULT_CFG["max_length"])
    parser.add_argument("--no_fp16",          action="store_true", help="Disable mixed precision")
    parser.add_argument("--patience",         type=int,   default=DEFAULT_CFG["patience"])
    args = parser.parse_args()

    cfg = DEFAULT_CFG.copy()
    cfg["epochs"]            = args.epochs
    cfg["batch_size"]        = args.batch_size
    cfg["grad_accumulation"] = args.grad_accumulation
    cfg["learning_rate"]     = args.lr
    cfg["max_length"]        = args.max_length
    cfg["patience"]          = args.patience
    cfg["fp16"]              = not args.no_fp16

    print(f"\nDevice : {DEVICE}")
    if DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram  = props.total_memory / 1e9
        print(f"GPU    : {props.name}")
        print(f"VRAM   : {vram:.2f} GB")
        if vram < 12 and cfg["batch_size"] > 4:
            print(f"\n[WARNING] VRAM < 12 GB detected. Consider --batch_size 4 --grad_accumulation 8")
    else:
        print("[WARNING] No GPU found — training will be very slow on CPU.")

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

    print("\nDone. All results in PROJECT_P3/results/large_models/")


if __name__ == "__main__":
    main()
