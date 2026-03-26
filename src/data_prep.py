"""
data_prep.py
============
Builds the final merged dataset for the P3 Norm Classifier project.

Sources:
  NORM (label=1):
    - culturebank_reddit.csv   → actor_behavior, agreement >= 0.7
    - culturebank_tiktok.csv   → actor_behavior, agreement >= 0.7

  NON-NORM (label=0):
    - wikipedia.parquet        → sentence column, ~20-25K sentences
                                 (keyword-filtered to remove cultural/behavioral sentences)

Output:
  data/train.csv, data/val.csv, data/test.csv  (80 / 10 / 10 split)
  data/merged_full.csv                          (complete merged dataset)
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR   = os.path.join(BASE, "data", "raw")
DATA_OUT  = os.path.join(BASE, "data")
os.makedirs(DATA_OUT, exist_ok=True)

REDDIT_PATH     = os.path.join(RAW_DIR, "culturebank_reddit.csv")
TIKTOK_PATH     = os.path.join(RAW_DIR, "culturebank_tiktok.csv")
WIKIPEDIA_PATH  = os.path.join(RAW_DIR, "wikipedia.parquet")

# ─── Config ───────────────────────────────────────────────────────────────────
AGREEMENT_THRESH = 0.70   # CultureBank quality filter

MIN_WORDS = 5
MAX_WORDS = 80

RANDOM_SEED = 42

# Keywords that indicate a sentence is culturally/behaviorally normative.
# Sentences containing any of these are removed from the Wikipedia non-norm set.
NORM_KEYWORDS = [
    "should", "must", "expected to", "it is customary",
    "it is common to", "traditionally", "norm", "etiquette",
    "polite", "rude", "respectful", "greeting", "bow", "tip",
]

# ─── Text cleaning ────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = text.strip()
    return text

def is_valid(text: str) -> bool:
    """Keep sentences with between MIN_WORDS and MAX_WORDS words."""
    if not text:
        return False
    words = text.split()
    return MIN_WORDS <= len(words) <= MAX_WORDS

def is_clean_non_norm(sentence: str) -> bool:
    """Return True if sentence contains no norm-like keywords."""
    sentence_lower = sentence.lower()
    return not any(kw in sentence_lower for kw in NORM_KEYWORDS)

# ─── Load NORM data ───────────────────────────────────────────────────────────
def load_norms() -> pd.DataFrame:
    dfs = []
    for path, source_name in [(REDDIT_PATH, "culturebank_reddit"),
                               (TIKTOK_PATH,  "culturebank_tiktok")]:
        df = pd.read_csv(path)
        print(f"[{source_name}] Raw rows: {len(df)}")

        df = df[df["agreement"] >= AGREEMENT_THRESH].copy()
        print(f"[{source_name}] After agreement >= {AGREEMENT_THRESH}: {len(df)}")

        df["sentence"]       = df["actor_behavior"].apply(clean_text)
        df["cultural_group"] = df["cultural group"].apply(
            lambda x: clean_text(str(x)) if pd.notna(x) else "unknown"
        )
        df["source"] = source_name
        df["label"]  = 1

        df = df[df["sentence"].apply(is_valid)]
        print(f"[{source_name}] After length filter: {len(df)}")

        dfs.append(df[["sentence", "label", "cultural_group", "source"]])

    norms = pd.concat(dfs, ignore_index=True)
    norms = norms.drop_duplicates(subset="sentence")
    print(f"\nTotal NORM rows (deduplicated): {len(norms)}")
    return norms

# ─── Load NON-NORM data ───────────────────────────────────────────────────────
def load_non_norms(target_count: int) -> pd.DataFrame:
    wiki = pd.read_parquet(WIKIPEDIA_PATH)
    print(f"\n[wikipedia] Raw rows: {len(wiki)}")

    wiki["sentence"] = wiki["sentence"].apply(clean_text)

    # Length filter
    wiki = wiki[wiki["sentence"].apply(is_valid)]
    print(f"[wikipedia] After length filter: {len(wiki)}")

    # Keyword filter — remove sentences that look like norms
    wiki = wiki[wiki["sentence"].apply(is_clean_non_norm)]
    print(f"[wikipedia] After keyword filter: {len(wiki)}")

    # Sample to match norm count (target ~20-25K, capped by what's available)
    sample_n = min(target_count, len(wiki))
    wiki = wiki.sample(n=sample_n, random_state=RANDOM_SEED)
    print(f"[wikipedia] Sampled: {len(wiki)}")

    wiki = wiki.drop_duplicates(subset="sentence")
    wiki["label"]          = 0
    wiki["cultural_group"] = "none"
    wiki["source"]         = "wikipedia"

    print(f"\nTotal NON-NORM rows (deduplicated): {len(wiki)}")
    return wiki[["sentence", "label", "cultural_group", "source"]]

# ─── Merge & split ────────────────────────────────────────────────────────────
def build_dataset():
    norms     = load_norms()
    non_norms = load_non_norms(target_count=len(norms))   # 1:1 balance

    # Trim majority class to exact 1:1
    n         = min(len(norms), len(non_norms))
    norms     = norms.sample(n=n, random_state=RANDOM_SEED)
    non_norms = non_norms.sample(n=n, random_state=RANDOM_SEED)

    merged = pd.concat([norms, non_norms], ignore_index=True)
    # Shuffle so norms and non-norms are fully interleaved (no clusters)
    merged = merged.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"\n{'='*50}")
    print(f"MERGED DATASET  ->  {len(merged)} rows")
    print(f"  Norm     (1): {merged['label'].sum()}")
    print(f"  Non-norm (0): {(merged['label'] == 0).sum()}")
    print(f"  Sources: {merged['source'].value_counts().to_dict()}")
    print(f"{'='*50}\n")

    merged.to_csv(os.path.join(DATA_OUT, "merged_full.csv"), index=False)
    print("Saved -> data/merged_full.csv")

    # ── Train / Val / Test split (80 / 10 / 10) ──
    train, temp = train_test_split(
        merged, test_size=0.20, random_state=RANDOM_SEED, stratify=merged["label"]
    )
    val, test = train_test_split(
        temp, test_size=0.50, random_state=RANDOM_SEED, stratify=temp["label"]
    )

    train.to_csv(os.path.join(DATA_OUT, "train.csv"), index=False)
    val.to_csv(  os.path.join(DATA_OUT, "val.csv"),   index=False)
    test.to_csv( os.path.join(DATA_OUT, "test.csv"),  index=False)

    print(f"Train: {len(train)} rows  |  Val: {len(val)} rows  |  Test: {len(test)} rows")
    print("\nSplit label distribution:")
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        vc = split_df["label"].value_counts().sort_index()
        print(f"  {split_name}: norm={vc.get(1,0)}  non-norm={vc.get(0,0)}")

    print("\nData preparation complete.")
    return merged, train, val, test


if __name__ == "__main__":
    build_dataset()
