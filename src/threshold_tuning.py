"""
threshold_tuning.py
===================
Finds the optimal classification threshold for the trained DeBERTa model
by sweeping over val.csv, then re-evaluates test.csv with the best threshold.

WHY:
----
Default threshold = 0.5 causes 877 false positives because the model barely
tips over 0.5 for NormBank taboo sentences (mean FP confidence = 0.54).
Raising the threshold forces the model to be MORE confident before calling
something a Norm, eliminating borderline FPs while keeping the 0.99-recall
Norm class nearly intact.

HOW TO USE (on Colab):
----------------------
    python src/threshold_tuning.py --model deberta

OUTPUTS:
--------
    results/threshold_tuning/
        ├── threshold_curves.png          (F1 / precision / recall vs threshold)
        ├── threshold_comparison.png      (before/after confusion matrices)
        ├── threshold_metrics.csv         (full metric table per threshold)
        └── threshold_summary.txt         (readable report)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "saved_models")
OUT_DIR    = os.path.join(BASE, "results", "threshold_tuning")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
class NormDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts      = df["sentence"].tolist()
        self.labels     = df["label"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.texts)

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


def get_probabilities(model, loader, fp16=True):
    """Return (true_labels, prob_norm) arrays for the full dataset."""
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            with autocast(enabled=fp16 and DEVICE.type == "cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.extend(probs[:, 1].tolist())   # P(Norm)
            all_labels.extend(batch["labels"].numpy().tolist())

    return np.array(all_labels), np.array(all_probs)


def apply_threshold(prob_norm, threshold):
    return (prob_norm >= threshold).astype(int)


def metrics_at_threshold(labels, prob_norm, threshold):
    preds = apply_threshold(prob_norm, threshold)
    return {
        "threshold":        round(threshold, 3),
        "accuracy":         round(accuracy_score(labels, preds), 4),
        "macro_f1":         round(f1_score(labels, preds, average="macro", zero_division=0), 4),
        "norm_f1":          round(f1_score(labels, preds, pos_label=1, zero_division=0), 4),
        "nonnorm_f1":       round(f1_score(labels, preds, pos_label=0, zero_division=0), 4),
        "norm_precision":   round(precision_score(labels, preds, pos_label=1, zero_division=0), 4),
        "norm_recall":      round(recall_score(labels, preds, pos_label=1, zero_division=0), 4),
        "nonnorm_precision":round(precision_score(labels, preds, pos_label=0, zero_division=0), 4),
        "nonnorm_recall":   round(recall_score(labels, preds, pos_label=0, zero_division=0), 4),
        "fp": int(((preds == 1) & (labels == 0)).sum()),
        "fn": int(((preds == 0) & (labels == 1)).sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_threshold_curves(results_df, best_threshold):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    t = results_df["threshold"]

    # --- F1 scores ---
    axes[0].plot(t, results_df["macro_f1"],   "k-o", ms=4, label="Macro F1")
    axes[0].plot(t, results_df["norm_f1"],    "b-o", ms=4, label="Norm F1")
    axes[0].plot(t, results_df["nonnorm_f1"], "r-o", ms=4, label="Non-Norm F1")
    axes[0].axvline(best_threshold, color="green", linestyle="--", label=f"Best ({best_threshold:.2f})")
    axes[0].set_title("F1 Score vs Threshold")
    axes[0].set_xlabel("Threshold"); axes[0].set_ylabel("F1")
    axes[0].legend(); axes[0].grid(alpha=0.4)

    # --- Precision / Recall for Norm ---
    axes[1].plot(t, results_df["norm_precision"], "b--o", ms=4, label="Norm Precision")
    axes[1].plot(t, results_df["norm_recall"],    "b-o",  ms=4, label="Norm Recall")
    axes[1].axvline(best_threshold, color="green", linestyle="--", label=f"Best ({best_threshold:.2f})")
    axes[1].set_title("Norm Class: Precision & Recall vs Threshold")
    axes[1].set_xlabel("Threshold")
    axes[1].legend(); axes[1].grid(alpha=0.4)

    # --- FP / FN counts ---
    axes[2].plot(t, results_df["fp"], "r-o", ms=4, label="False Positives")
    axes[2].plot(t, results_df["fn"], "b-o", ms=4, label="False Negatives")
    axes[2].axvline(best_threshold, color="green", linestyle="--", label=f"Best ({best_threshold:.2f})")
    axes[2].set_title("Error Counts vs Threshold")
    axes[2].set_xlabel("Threshold"); axes[2].set_ylabel("Count")
    axes[2].legend(); axes[2].grid(alpha=0.4)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "threshold_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrices(val_labels, val_probs, test_labels, test_probs,
                             default_t, best_t):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    configs = [
        (val_labels,  val_probs,  default_t, "Val Set — Default (0.50)",    axes[0][0]),
        (val_labels,  val_probs,  best_t,    f"Val Set — Tuned ({best_t:.2f})", axes[0][1]),
        (test_labels, test_probs, default_t, "Test Set — Default (0.50)",   axes[1][0]),
        (test_labels, test_probs, best_t,    f"Test Set — Tuned ({best_t:.2f})",axes[1][1]),
    ]

    for labels, probs, threshold, title, ax in configs:
        preds = apply_threshold(probs, threshold)
        cm    = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Non-Norm", "Norm"],
                    yticklabels=["Non-Norm", "Norm"], ax=ax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "threshold_comparison.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────
def write_report(results_df, best_row, test_default, test_best, best_threshold):
    lines = []
    lines.append("=" * 70)
    lines.append("  THRESHOLD TUNING REPORT — DeBERTa-v3-base Norm Classifier")
    lines.append("=" * 70)

    lines.append("\n  Threshold sweep on val.csv (80/10/10 validation split)")
    lines.append(f"  Thresholds tested: {results_df['threshold'].min()} → {results_df['threshold'].max()}")

    lines.append("\n" + "─" * 70)
    lines.append("  VAL SET — TOP 10 THRESHOLDS BY MACRO F1")
    lines.append("─" * 70)
    top10 = results_df.nlargest(10, "macro_f1")[
        ["threshold", "macro_f1", "nonnorm_recall", "norm_recall", "fp", "fn"]
    ]
    lines.append(top10.to_string(index=False))

    lines.append("\n" + "─" * 70)
    lines.append(f"  BEST THRESHOLD: {best_threshold:.2f}  (maximises macro F1 on val set)")
    lines.append("─" * 70)
    for k, v in best_row.items():
        lines.append(f"    {k:<22s}: {v}")

    lines.append("\n" + "─" * 70)
    lines.append("  TEST SET COMPARISON — Default (0.50) vs Tuned Threshold")
    lines.append("─" * 70)
    comparison = pd.DataFrame([test_default, test_best])
    lines.append(comparison[["threshold", "accuracy", "macro_f1",
                               "nonnorm_recall", "norm_recall", "fp", "fn"]].to_string(index=False))

    # Improvement summary
    fp_reduction = test_default["fp"] - test_best["fp"]
    fn_increase  = test_best["fn"] - test_default["fn"]
    f1_change    = test_best["macro_f1"] - test_default["macro_f1"]
    lines.append(f"\n  FP reduction : -{fp_reduction}  ({test_default['fp']} → {test_best['fp']})")
    lines.append(f"  FN increase  : +{fn_increase}   ({test_default['fn']} → {test_best['fn']})")
    lines.append(f"  Macro F1 change : {f1_change:+.4f}")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print(report)
    path = os.path.join(OUT_DIR, "threshold_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Report saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="deberta")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--t_min",      type=float, default=0.30)
    parser.add_argument("--t_max",      type=float, default=0.80)
    parser.add_argument("--t_step",     type=float, default=0.01)
    args = parser.parse_args()

    ckpt = os.path.join(MODELS_DIR, f"{args.model}_best")
    if not os.path.isdir(ckpt):
        print(f"ERROR: No saved model at {ckpt}")
        sys.exit(1)

    print(f"\nDevice : {DEVICE}")
    print(f"Loading model: {ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model     = AutoModelForSequenceClassification.from_pretrained(ckpt).to(DEVICE)
    fp16      = DEVICE.type == "cuda"

    val_df  = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"Val: {len(val_df)} rows  |  Test: {len(test_df)} rows")

    val_loader = DataLoader(
        NormDataset(val_df, tokenizer, args.max_length),
        batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        NormDataset(test_df, tokenizer, args.max_length),
        batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    print("\nGetting val probabilities...")
    val_labels, val_probs = get_probabilities(model, val_loader, fp16)

    print("Getting test probabilities...")
    test_labels, test_probs = get_probabilities(model, test_loader, fp16)

    # ── Sweep thresholds on val set ───────────────────────────────
    thresholds = np.arange(args.t_min, args.t_max + args.t_step / 2, args.t_step)
    print(f"\nSweeping {len(thresholds)} thresholds on val set...")
    records = [metrics_at_threshold(val_labels, val_probs, t) for t in thresholds]
    results_df = pd.DataFrame(records)
    results_df.to_csv(os.path.join(OUT_DIR, "threshold_metrics.csv"), index=False)
    print(f"  Saved: threshold_metrics.csv")

    # ── Best threshold: maximize macro F1 on val ─────────────────
    best_idx       = results_df["macro_f1"].idxmax()
    best_row       = results_df.iloc[best_idx].to_dict()
    best_threshold = best_row["threshold"]
    print(f"\n  Best threshold (val macro F1): {best_threshold:.2f}")

    # ── Test set evaluation: default vs tuned ────────────────────
    test_default = metrics_at_threshold(test_labels, test_probs, 0.50)
    test_best    = metrics_at_threshold(test_labels, test_probs, best_threshold)

    # ── Plots ─────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_threshold_curves(results_df, best_threshold)
    plot_confusion_matrices(val_labels, val_probs, test_labels, test_probs,
                             0.50, best_threshold)

    # ── Report ────────────────────────────────────────────────────
    print()
    write_report(results_df, best_row, test_default, test_best, best_threshold)

    # ── Detailed classification report at best threshold ─────────
    print(f"\n  Test set classification report at threshold={best_threshold:.2f}:")
    preds = apply_threshold(test_probs, best_threshold)
    print(classification_report(test_labels, preds, target_names=["Non-Norm", "Norm"]))

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
