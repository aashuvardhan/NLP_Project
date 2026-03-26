"""
data_prep.py
============
Builds the final merged dataset for the P3 Norm Classifier project.

Sources:
  NORM (label=1):
    - culturebank_reddit.csv   → actor_behavior, agreement >= 0.7
    - culturebank_tiktok.csv   → actor_behavior, agreement >= 0.7

  NON-NORM (label=0):
    - generic_kb_simplewiki.parquet  → sentence, bert_score >= 0.23
    - generics_kb_best.parquet       → generic_sentence, score >= 0.55

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
SIMPLEWIKI_PATH = os.path.join(RAW_DIR, "generic_kb_simplewiki.parquet")
GENERICS_PATH   = os.path.join(RAW_DIR, "generics_kb_best.parquet")

# ─── Config ───────────────────────────────────────────────────────────────────
AGREEMENT_THRESH  = 0.70   # CultureBank quality filter
BERT_SCORE_THRESH = 0.23   # SimpleWiki quality filter
GENERIC_SCORE_THRESH = 0.55  # GenericsKB best quality filter

MIN_WORDS = 5
MAX_WORDS = 80

RANDOM_SEED = 42

# ─── Text cleaning ────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # collapse multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text)
    # remove non-ASCII junk but keep punctuation
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = text.strip()
    return text

def is_valid(text: str) -> bool:
    """Keep sentences that have between MIN_WORDS and MAX_WORDS words."""
    if not text:
        return False
    words = text.split()
    return MIN_WORDS <= len(words) <= MAX_WORDS

# ─── Load NORM data ───────────────────────────────────────────────────────────
def load_norms() -> pd.DataFrame:
    dfs = []
    for path, source_name in [(REDDIT_PATH, "culturebank_reddit"),
                               (TIKTOK_PATH,  "culturebank_tiktok")]:
        df = pd.read_csv(path)
        print(f"[{source_name}] Raw rows: {len(df)}")

        # quality filter
        df = df[df["agreement"] >= AGREEMENT_THRESH].copy()
        print(f"[{source_name}] After agreement >= {AGREEMENT_THRESH}: {len(df)}")

        # use actor_behavior as the norm sentence
        df["sentence"]       = df["actor_behavior"].apply(clean_text)
        df["cultural_group"] = df["cultural group"].apply(
            lambda x: clean_text(str(x)) if pd.notna(x) else "unknown"
        )
        df["source"]  = source_name
        df["label"]   = 1

        # drop rows where sentence is empty or too short/long
        df = df[df["sentence"].apply(is_valid)]
        print(f"[{source_name}] After length filter: {len(df)}")

        dfs.append(df[["sentence", "label", "cultural_group", "source"]])

    norms = pd.concat(dfs, ignore_index=True)
    norms = norms.drop_duplicates(subset="sentence")
    print(f"\nTotal NORM rows (deduplicated): {len(norms)}")
    return norms

# ─── Load NON-NORM data ───────────────────────────────────────────────────────
def load_non_norms(target_count: int) -> pd.DataFrame:
    dfs = []

    # --- SimpleWiki ---
    sw = pd.read_parquet(SIMPLEWIKI_PATH)
    print(f"\n[simplewiki] Raw rows: {len(sw)}")
    sw = sw[sw["bert_score"] >= BERT_SCORE_THRESH].copy()
    print(f"[simplewiki] After bert_score >= {BERT_SCORE_THRESH}: {len(sw)}")
    sw["sentence"] = sw["sentence"].apply(clean_text)
    sw = sw[sw["sentence"].apply(is_valid)]
    sw["label"]          = 0
    sw["cultural_group"] = "none"
    sw["source"]         = "simplewiki"
    dfs.append(sw[["sentence", "label", "cultural_group", "source"]])
    print(f"[simplewiki] After length filter: {len(sw)}")

    simplewiki_count = len(sw)

    # --- GenericsKB best ---
    remaining = target_count - simplewiki_count
    print(f"\n[generics_best] Need {remaining} more non-norm rows...")

    gb = pd.read_parquet(GENERICS_PATH, columns=["generic_sentence", "score", "source"])
    print(f"[generics_best] Raw rows: {len(gb)}")
    gb = gb[gb["score"] >= GENERIC_SCORE_THRESH].copy()
    print(f"[generics_best] After score >= {GENERIC_SCORE_THRESH}: {len(gb)}")

    gb["sentence"] = gb["generic_sentence"].apply(clean_text)
    gb = gb[gb["sentence"].apply(is_valid)]
    print(f"[generics_best] After length filter: {len(gb)}")

    # Sample to fill the gap (stratified across sources for diversity)
    if len(gb) > remaining:
        gb = gb.groupby("source", group_keys=False).apply(
            lambda x: x.sample(frac=min(1.0, remaining / len(gb)), random_state=RANDOM_SEED)
        )
        # in case we got a bit more or less, trim or top-up simply
        if len(gb) > remaining:
            gb = gb.sample(n=remaining, random_state=RANDOM_SEED)

    gb_out = gb[["sentence"]].copy()
    gb_out["label"]          = 0
    gb_out["cultural_group"] = "none"
    gb_out["source"]         = "generics_kb_best"
    dfs.append(gb_out)
    print(f"[generics_best] Sampled: {len(gb_out)}")

    non_norms = pd.concat(dfs, ignore_index=True)
    non_norms = non_norms.drop_duplicates(subset="sentence")
    print(f"\nTotal NON-NORM rows (deduplicated): {len(non_norms)}")
    return non_norms

# ─── Merge & split ────────────────────────────────────────────────────────────
def build_dataset():
    norms     = load_norms()
    non_norms = load_non_norms(target_count=len(norms))   # aim for 1:1 balance

    # final balance check – trim majority class to exact 1:1
    n = min(len(norms), len(non_norms))
    norms     = norms.sample(n=n, random_state=RANDOM_SEED)
    non_norms = non_norms.sample(n=n, random_state=RANDOM_SEED)

    merged = pd.concat([norms, non_norms], ignore_index=True)
    merged = merged.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"\n{'='*50}")
    print(f"MERGED DATASET  ->  {len(merged)} rows")
    print(f"  Norm     (1): {merged['label'].sum()}")
    print(f"  Non-norm (0): {(merged['label'] == 0).sum()}")
    print(f"  Sources: {merged['source'].value_counts().to_dict()}")
    print(f"{'='*50}\n")

    # Save full merged
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
