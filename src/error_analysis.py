"""
error_analysis.py
=================
Loads the trained DeBERTa model and performs detailed error analysis
on the test set — inspecting ~50 false positives and ~50 false negatives.

HOW TO USE:
-----------
    python src/error_analysis.py
    python src/error_analysis.py --model deberta --n_samples 50

OUTPUTS:
--------
    results/error_analysis/
        ├── false_positives.csv          (predicted Norm, actually Non-Norm)
        ├── false_negatives.csv          (predicted Non-Norm, actually Norm)
        ├── error_summary.txt            (readable report)
        ├── fp_by_source.png
        ├── fn_by_source.png
        ├── fp_confidence_hist.png
        ├── error_sentence_length.png
        └── template_analysis.png
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE, "data")
MODELS_DIR  = os.path.join(BASE, "saved_models")
OUT_DIR     = os.path.join(BASE, "results", "error_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# DATASET  (mirrors transformer_model.py)
# ─────────────────────────────────────────────────────────────────────────────
class NormDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
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
# INFERENCE  — returns predictions + confidence scores
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(model, loader, fp16=True):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            with autocast(enabled=fp16 and DEVICE.type == "cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.tolist())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def detect_template(sentence: str) -> str:
    """Classify sentence template type."""
    s = sentence.strip().lower()
    if s.startswith("in "):
        return "Template-A: In {ctx}, {grp} {behav}"
    first_word = s.split()[0] if s else ""
    # Norm-like patterns: "{group} {behavior} in {context}"
    if " in " in s:
        return "Template-B: {grp} {behav} in {ctx}"
    return "Other"


def word_count(sentence: str) -> int:
    return len(str(sentence).split())


def top_sources(df, n=8):
    return df["source"].value_counts().head(n)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def plot_source_bar(fp_src, fn_src):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # False Positives by source
    fp_src.plot(kind="barh", ax=axes[0], color="#C44E52")
    axes[0].set_title("False Positives by Source\n(Predicted Norm, Actually Non-Norm)", fontsize=11)
    axes[0].set_xlabel("Count")
    axes[0].invert_yaxis()
    for bar in axes[0].patches:
        axes[0].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                     str(int(bar.get_width())), va="center", fontsize=9)

    # False Negatives by source
    fn_src.plot(kind="barh", ax=axes[1], color="#4C72B0")
    axes[1].set_title("False Negatives by Source\n(Predicted Non-Norm, Actually Norm)", fontsize=11)
    axes[1].set_xlabel("Count")
    axes[1].invert_yaxis()
    for bar in axes[1].patches:
        axes[1].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                     str(int(bar.get_width())), va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "errors_by_source.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


def plot_confidence_hist(fp_df, fn_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(fp_df["confidence"], bins=20, color="#C44E52", edgecolor="black", alpha=0.8)
    axes[0].set_title("FP Confidence Distribution\n(Model confidence it was Norm)", fontsize=11)
    axes[0].set_xlabel("Confidence (P(Norm))"); axes[0].set_ylabel("Count")
    axes[0].axvline(fp_df["confidence"].mean(), color="black", linestyle="--",
                    label=f"Mean={fp_df['confidence'].mean():.2f}")
    axes[0].legend()

    axes[1].hist(fn_df["confidence"], bins=20, color="#4C72B0", edgecolor="black", alpha=0.8)
    axes[1].set_title("FN Confidence Distribution\n(Model confidence it was Non-Norm)", fontsize=11)
    axes[1].set_xlabel("Confidence (P(Non-Norm))"); axes[1].set_ylabel("Count")
    axes[1].axvline(fn_df["confidence"].mean(), color="black", linestyle="--",
                    label=f"Mean={fn_df['confidence'].mean():.2f}")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "confidence_distributions.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


def plot_sentence_length(fp_df, fn_df, correct_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = range(0, 90, 5)
    ax.hist(correct_df["word_count"], bins=bins, alpha=0.5, label="Correct", color="#55A868")
    ax.hist(fp_df["word_count"],      bins=bins, alpha=0.7, label="False Positives", color="#C44E52")
    ax.hist(fn_df["word_count"],      bins=bins, alpha=0.7, label="False Negatives", color="#4C72B0")
    ax.set_xlabel("Word Count"); ax.set_ylabel("Number of Sentences")
    ax.set_title("Sentence Length: Correct vs Misclassified")
    ax.legend(); ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "sentence_length_analysis.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


def plot_template_analysis(fp_df, fn_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    fp_tmpl = fp_df["template"].value_counts()
    fn_tmpl = fn_df["template"].value_counts()

    fp_tmpl.plot(kind="bar", ax=axes[0], color="#C44E52", edgecolor="black")
    axes[0].set_title("False Positives — Template Type", fontsize=11)
    axes[0].set_xlabel(""); axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=20)

    fn_tmpl.plot(kind="bar", ax=axes[1], color="#4C72B0", edgecolor="black")
    axes[1].set_title("False Negatives — Template Type", fontsize=11)
    axes[1].set_xlabel(""); axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "template_analysis.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


def plot_cultural_group(fp_df, fn_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    fp_grp = fp_df[fp_df["cultural_group"] != "none"]["cultural_group"].value_counts().head(10)
    fn_grp = fn_df[fn_df["cultural_group"] != "none"]["cultural_group"].value_counts().head(10)

    if not fp_grp.empty:
        fp_grp.plot(kind="barh", ax=axes[0], color="#C44E52")
        axes[0].invert_yaxis()
    axes[0].set_title("Top Cultural Groups in FP", fontsize=11)
    axes[0].set_xlabel("Count")

    if not fn_grp.empty:
        fn_grp.plot(kind="barh", ax=axes[1], color="#4C72B0")
        axes[1].invert_yaxis()
    axes[1].set_title("Top Cultural Groups in FN", fontsize=11)
    axes[1].set_xlabel("Count")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "cultural_group_analysis.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# TEXT REPORT
# ─────────────────────────────────────────────────────────────────────────────
def write_report(fp_df, fn_df, all_preds, all_labels, fp_sample, fn_sample):
    total   = len(all_labels)
    correct = int((all_preds == all_labels).sum())
    acc     = correct / total

    fp_all_count = int(((all_preds == 1) & (all_labels == 0)).sum())
    fn_all_count = int(((all_preds == 0) & (all_labels == 1)).sum())

    lines = []
    lines.append("=" * 70)
    lines.append("  ERROR ANALYSIS REPORT — DeBERTa-v3-base Norm Classifier")
    lines.append("=" * 70)
    lines.append(f"\n  Test Set Size     : {total}")
    lines.append(f"  Overall Accuracy  : {acc:.4f}  ({correct}/{total} correct)")
    lines.append(f"\n  Total Errors      : {total - correct}")
    lines.append(f"    False Positives  : {fp_all_count}  (predicted Norm, actually Non-Norm)")
    lines.append(f"    False Negatives  : {fn_all_count}  (predicted Non-Norm, actually Norm)")
    lines.append(f"\n  Non-Norm Recall   : {(all_labels[all_labels==0] == all_preds[all_labels==0]).mean():.4f}")
    lines.append(f"  Norm Recall       : {(all_labels[all_labels==1] == all_preds[all_labels==1]).mean():.4f}")

    # ── FP breakdown ──────────────────────────────────────────────
    lines.append("\n" + "─" * 70)
    lines.append("  FALSE POSITIVES ANALYSIS  (predicted Norm → actually Non-Norm)")
    lines.append("─" * 70)
    lines.append(f"\n  Total FPs: {fp_all_count}  |  Showing analysis of all {len(fp_df)}\n")

    lines.append("  By Source:")
    for src, cnt in fp_df["source"].value_counts().items():
        pct = cnt / len(fp_df) * 100
        lines.append(f"    {src:<35s} {cnt:>4d}  ({pct:.1f}%)")

    lines.append("\n  By Template Type:")
    for tmpl, cnt in fp_df["template"].value_counts().items():
        pct = cnt / len(fp_df) * 100
        lines.append(f"    {tmpl:<45s} {cnt:>4d}  ({pct:.1f}%)")

    lines.append(f"\n  Avg confidence (P(Norm))   : {fp_df['confidence'].mean():.4f}")
    lines.append(f"  Avg word count             : {fp_df['word_count'].mean():.1f}")

    lines.append(f"\n  Sample FPs (up to {len(fp_sample)}):")
    for i, row in fp_sample.iterrows():
        lines.append(f"\n  [{i}] source={row['source']} | group={row['cultural_group']} | conf={row['confidence']:.3f}")
        lines.append(f"       {row['sentence']}")

    # ── FN breakdown ──────────────────────────────────────────────
    lines.append("\n" + "─" * 70)
    lines.append("  FALSE NEGATIVES ANALYSIS  (predicted Non-Norm → actually Norm)")
    lines.append("─" * 70)
    lines.append(f"\n  Total FNs: {fn_all_count}  |  Showing analysis of all {len(fn_df)}\n")

    lines.append("  By Source:")
    for src, cnt in fn_df["source"].value_counts().items():
        pct = cnt / len(fn_df) * 100
        lines.append(f"    {src:<35s} {cnt:>4d}  ({pct:.1f}%)")

    lines.append("\n  By Template Type:")
    for tmpl, cnt in fn_df["template"].value_counts().items():
        pct = cnt / len(fn_df) * 100
        lines.append(f"    {tmpl:<45s} {cnt:>4d}  ({pct:.1f}%)")

    lines.append(f"\n  Avg confidence (P(Non-Norm)): {fn_df['confidence'].mean():.4f}")
    lines.append(f"  Avg word count              : {fn_df['word_count'].mean():.1f}")

    lines.append(f"\n  Sample FNs (up to {len(fn_sample)}):")
    for i, row in fn_sample.iterrows():
        lines.append(f"\n  [{i}] source={row['source']} | group={row['cultural_group']} | conf={row['confidence']:.3f}")
        lines.append(f"       {row['sentence']}")

    lines.append("\n" + "=" * 70)

    report_text = "\n".join(lines)
    print(report_text)

    path = os.path.join(OUT_DIR, "error_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n  Report saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Error Analysis for Norm Classifier")
    parser.add_argument("--model",     default="deberta", help="Model key (must exist in saved_models/)")
    parser.add_argument("--n_samples", type=int, default=50, help="Max samples to show per error type")
    parser.add_argument("--batch_size",type=int, default=32)
    parser.add_argument("--max_length",type=int, default=128)
    args = parser.parse_args()

    ckpt = os.path.join(MODELS_DIR, f"{args.model}_best")
    if not os.path.isdir(ckpt):
        print(f"ERROR: No saved model found at {ckpt}")
        print("       Train the model first: python src/transformer_model.py --model deberta")
        sys.exit(1)

    # ── Load model & tokenizer ────────────────────────────────────
    print(f"\nDevice : {DEVICE}")
    print(f"Loading model from: {ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model     = AutoModelForSequenceClassification.from_pretrained(ckpt).to(DEVICE)
    model.eval()

    # ── Load test set ─────────────────────────────────────────────
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"Test set: {len(test_df)} rows\n")

    loader = DataLoader(
        NormDataset(test_df, tokenizer, args.max_length),
        batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # ── Run inference ─────────────────────────────────────────────
    print("Running inference...")
    fp16 = DEVICE.type == "cuda"
    all_preds, all_labels, all_probs = run_inference(model, loader, fp16=fp16)

    # ── Enrich test_df with predictions ──────────────────────────
    test_df = test_df.copy()
    test_df["pred"]       = all_preds
    test_df["prob_norm"]  = all_probs[:, 1]       # P(Norm)
    test_df["prob_nonnorm"] = all_probs[:, 0]     # P(Non-Norm)
    test_df["correct"]    = (test_df["pred"] == test_df["label"]).astype(int)
    test_df["word_count"] = test_df["sentence"].apply(word_count)
    test_df["template"]   = test_df["sentence"].apply(detect_template)

    # ── Separate error types ──────────────────────────────────────
    # FP: predicted=1 (Norm), actual=0 (Non-Norm)
    fp_mask = (test_df["pred"] == 1) & (test_df["label"] == 0)
    fp_df   = test_df[fp_mask].copy()
    fp_df["confidence"] = fp_df["prob_norm"]      # confidence in wrong prediction

    # FN: predicted=0 (Non-Norm), actual=1 (Norm)
    fn_mask = (test_df["pred"] == 0) & (test_df["label"] == 1)
    fn_df   = test_df[fn_mask].copy()
    fn_df["confidence"] = fp_df["prob_nonnorm"].reindex(fn_df.index) if False else fn_df["prob_nonnorm"]

    # Sort by confidence descending (most confident mistakes first)
    fp_df = fp_df.sort_values("confidence", ascending=False)
    fn_df = fn_df.sort_values("confidence", ascending=False)

    # Correct predictions (for comparison in length plot)
    correct_df = test_df[test_df["correct"] == 1].copy()

    # ── Save full CSVs ────────────────────────────────────────────
    fp_df.to_csv(os.path.join(OUT_DIR, "false_positives.csv"), index=False)
    fn_df.to_csv(os.path.join(OUT_DIR, "false_negatives.csv"), index=False)
    print(f"  Saved: false_positives.csv ({len(fp_df)} rows)")
    print(f"  Saved: false_negatives.csv ({len(fn_df)} rows)")

    # ── Sampled views for report ──────────────────────────────────
    fp_sample = fp_df.head(args.n_samples)
    fn_sample = fn_df.head(args.n_samples)

    # ── Plots ─────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_source_bar(top_sources(fp_df), top_sources(fn_df))
    plot_confidence_hist(fp_df, fn_df)
    plot_sentence_length(fp_df, fn_df, correct_df)
    plot_template_analysis(fp_df, fn_df)
    plot_cultural_group(fp_df, fn_df)

    # ── Text report ───────────────────────────────────────────────
    print()
    write_report(fp_df, fn_df, all_preds, all_labels, fp_sample, fn_sample)

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
