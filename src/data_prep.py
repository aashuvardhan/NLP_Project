"""
data_prep.py
============
Builds the final merged dataset for the P3 Norm Classifier project.

Sources:
  NORM (label=1):
    - culturebank_reddit.csv  → first sentence of eval_whole_desc, agreement >= 0.7
    - culturebank_tiktok.csv  → first sentence of eval_whole_desc, agreement >= 0.7

  NON-NORM / Hard Negatives (label=0):
    - ag_news_train.parquet   → factual world/news sentences about people & countries
    - ag_news_test.parquet    → same
    - squad.parquet           → factual passages about human activities (sentence-split)
    - wikipedia.parquet       → broad factual sentences (keyword-filtered)

  All non-norm sources are keyword-filtered to remove sentences that look like norms.

Output:
  data/merged_full.csv
  data/train.csv  (80%)
  data/val.csv    (10%)
  data/test.csv   (10%)
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE, "data", "raw")
DATA_OUT = os.path.join(BASE, "data")
os.makedirs(DATA_OUT, exist_ok=True)

REDDIT_PATH     = os.path.join(RAW_DIR, "culturebank_reddit.csv")
TIKTOK_PATH     = os.path.join(RAW_DIR, "culturebank_tiktok.csv")
AG_TRAIN_PATH   = os.path.join(RAW_DIR, "ag_news_train.parquet")
AG_TEST_PATH    = os.path.join(RAW_DIR, "ag_news_test.parquet")
SQUAD_PATH      = os.path.join(RAW_DIR, "squad.parquet")
WIKIPEDIA_PATH  = os.path.join(RAW_DIR, "wikipedia.parquet")

# ─── Config ───────────────────────────────────────────────────────────────────
AGREEMENT_THRESH = 0.70
MIN_WORDS        = 5
MAX_WORDS        = 80
RANDOM_SEED      = 42

# Sentences with any of these keywords are too norm-like for the non-norm set.
NORM_KEYWORDS = [
    "should", "must", "expected to", "it is customary",
    "it is common to", "traditionally", "norm", "etiquette",
    "polite", "rude", "respectful", "greeting", "bow", "tip",
]

# ─── Text helpers ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def is_valid(text: str) -> bool:
    if not text:
        return False
    return MIN_WORDS <= len(text.split()) <= MAX_WORDS

def is_clean_non_norm(sentence: str) -> bool:
    """True if the sentence contains no norm-like behavioral keywords."""
    sl = sentence.lower()
    return not any(kw in sl for kw in NORM_KEYWORDS)

def split_sentences(text: str) -> list:
    """Split a paragraph into individual sentences."""
    text = clean_text(text)
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in parts if s.strip()]

def clean_ag_text(text: str) -> str:
    """Clean AG News text: remove backslash sequences and source prefixes."""
    # Replace escaped sequences (\\band → ' band')
    text = re.sub(r'\\+', ' ', text)
    # Remove parenthetical source tags like (Reuters), (AP)
    text = re.sub(r'\([A-Z][A-Za-z\s&\.]+\)', '', text)
    # Remove leading "Source - " prefix
    text = re.sub(r'^[A-Z][A-Za-z\s&]+\s*-\s+', '', text)
    return clean_text(text)


# ─── NORM data ────────────────────────────────────────────────────────────────
def build_norm_sentence(row) -> str:
    """
    Construct a natural norm sentence from CultureBank fields:
        "In [context], [cultural_group] [actor_behavior]."

    Why NOT eval_whole_desc?
      eval_whole_desc is GPT-generated boilerplate. Every sentence ends with
      "widely regarded as a common practice within the sampled population" —
      a perfect lexical fingerprint that makes the task trivially easy (>99% acc).

    Why NOT raw actor_behavior?
      actor_behavior is a fragment: "dress casually, often in comfortable clothing"
      — not a grammatical sentence.

    Constructed sentence example:
      context        = "restaurant and service industry settings"
      cultural_group = "Americans"
      actor_behavior = "engage in tipping culture with varying expectations"
      → "In restaurant and service industry settings, Americans engage in
         tipping culture with varying expectations."
    """
    context  = clean_text(str(row.get("context", "")))
    group    = clean_text(str(row.get("cultural group", "")))
    behavior = clean_text(str(row.get("actor_behavior", "")))

    # Fall back gracefully if any field is missing
    if not context or context in ("nan", "unknown"):
        context = ""
    if not group or group in ("nan", "unknown"):
        group = ""
    if not behavior or behavior in ("nan", "unknown"):
        return ""

    # If context already starts with a preposition ("in X", "at X"), don't add "In"
    if context and re.match(r'^(in|at|during|within|across|among)\b', context, re.I):
        prefix = context[0].upper() + context[1:]   # just capitalise it
        if group:
            sentence = f"{prefix}, {group} {behavior}."
        else:
            sentence = f"{prefix}, people {behavior}."
    elif context and group:
        sentence = f"In {context}, {group} {behavior}."
    elif group:
        sentence = f"{group} {behavior}."
    else:
        sentence = f"{behavior}."

    # Capitalise first letter
    sentence = sentence[0].upper() + sentence[1:]
    return sentence


def load_norms() -> pd.DataFrame:
    dfs = []
    for path, source_name in [(REDDIT_PATH, "culturebank_reddit"),
                               (TIKTOK_PATH, "culturebank_tiktok")]:
        df = pd.read_csv(path)
        print(f"[{source_name}] Raw rows: {len(df)}")

        df = df[df["agreement"] >= AGREEMENT_THRESH].copy()
        print(f"[{source_name}] After agreement >= {AGREEMENT_THRESH}: {len(df)}")

        df["sentence"] = df.apply(build_norm_sentence, axis=1)
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


# ─── NON-NORM data (Hard Negatives) ──────────────────────────────────────────
def load_ag_news(target_n: int) -> pd.DataFrame:
    """
    AG News: factual news sentences about world events, people, and countries.
    e.g. "Japan's GDP grew by 2.1% in the third quarter of 2023."
    Combines train + test splits for maximum coverage.
    """
    print(f"\n[ag_news] Loading from local parquets...")
    parts = []
    for path in [AG_TRAIN_PATH, AG_TEST_PATH]:
        if os.path.exists(path):
            parts.append(pd.read_parquet(path))
    ag = pd.concat(parts, ignore_index=True)
    print(f"[ag_news] Raw rows (train+test): {len(ag)}")

    sentences = []
    for text in ag["text"]:
        cleaned = clean_ag_text(str(text))
        for s in split_sentences(cleaned):
            sentences.append(s)

    print(f"[ag_news] Raw sentences extracted: {len(sentences)}")

    df = pd.DataFrame({"sentence": sentences})
    df = df[df["sentence"].apply(is_valid)]
    df = df[df["sentence"].apply(is_clean_non_norm)]
    df = df.drop_duplicates(subset="sentence")
    print(f"[ag_news] After filters + dedup: {len(df)}")

    sample_n = min(target_n, len(df))
    df = df.sample(n=sample_n, random_state=RANDOM_SEED)
    df["label"]          = 0
    df["cultural_group"] = "none"
    df["source"]         = "ag_news"
    print(f"[ag_news] Final sample: {len(df)}")
    return df[["sentence", "label", "cultural_group", "source"]]


def load_squad(target_n: int) -> pd.DataFrame:
    """
    SQuAD context passages split into sentences.
    Passages are Wikipedia-sourced factual descriptions of human/cultural activities.
    e.g. "The United States spends more per capita on healthcare than any other nation."
    Contexts are deduplicated before splitting (many questions share one passage).
    """
    print(f"\n[squad] Loading from local parquet...")
    sq = pd.read_parquet(SQUAD_PATH)
    print(f"[squad] Raw rows: {len(sq)}")

    # Deduplicate contexts — same passage appears for multiple questions
    unique_contexts = sq["context"].drop_duplicates().tolist()
    print(f"[squad] Unique contexts: {len(unique_contexts)}")

    sentences = []
    for ctx in unique_contexts:
        for s in split_sentences(str(ctx)):
            sentences.append(clean_text(s))

    print(f"[squad] Raw sentences from unique contexts: {len(sentences)}")

    df = pd.DataFrame({"sentence": sentences})
    df = df[df["sentence"].apply(is_valid)]
    df = df[df["sentence"].apply(is_clean_non_norm)]
    df = df.drop_duplicates(subset="sentence")
    print(f"[squad] After filters + dedup: {len(df)}")

    sample_n = min(target_n, len(df))
    df = df.sample(n=sample_n, random_state=RANDOM_SEED)
    df["label"]          = 0
    df["cultural_group"] = "none"
    df["source"]         = "squad"
    print(f"[squad] Final sample: {len(df)}")
    return df[["sentence", "label", "cultural_group", "source"]]


def load_wikipedia(target_n: int) -> pd.DataFrame:
    """
    Wikipedia: broad factual sentences across all topics.
    Keyword-filtered to remove any sentences resembling cultural norms (~2-5% removed).
    """
    print(f"\n[wikipedia] Loading from local parquet...")
    wiki = pd.read_parquet(WIKIPEDIA_PATH)
    print(f"[wikipedia] Raw rows: {len(wiki)}")

    wiki["sentence"] = wiki["sentence"].apply(clean_text)
    wiki = wiki[wiki["sentence"].apply(is_valid)]
    wiki = wiki[wiki["sentence"].apply(is_clean_non_norm)]
    wiki = wiki.drop_duplicates(subset="sentence")
    print(f"[wikipedia] After filters + dedup: {len(wiki)}")

    sample_n = min(target_n, len(wiki))
    wiki = wiki.sample(n=sample_n, random_state=RANDOM_SEED)
    wiki["label"]          = 0
    wiki["cultural_group"] = "none"
    wiki["source"]         = "wikipedia"
    print(f"[wikipedia] Final sample: {len(wiki)}")
    return wiki[["sentence", "label", "cultural_group", "source"]]


def load_non_norms(target_count: int) -> pd.DataFrame:
    """
    Combine three hard-negative sources, splitting the target evenly.
    AG News + SQuAD are primary (people/world focus).
    Wikipedia fills remainder as supplementary factual sentences.
    """
    per_source = target_count // 3
    remainder  = target_count % 3   # extra rows go to ag_news

    ag_df   = load_ag_news(per_source + remainder)
    sq_df   = load_squad(per_source)
    wiki_df = load_wikipedia(per_source)

    non_norms = pd.concat([ag_df, sq_df, wiki_df], ignore_index=True)
    non_norms = non_norms.drop_duplicates(subset="sentence")
    print(f"\nTotal NON-NORM rows (deduplicated): {len(non_norms)}")
    return non_norms


# ─── Merge & split ────────────────────────────────────────────────────────────
def build_dataset():
    print("\n" + "="*60)
    print("  P3 Norm Classifier — Data Preparation")
    print("="*60 + "\n")

    norms     = load_norms()
    non_norms = load_non_norms(target_count=len(norms))

    # Trim to exact 1:1 balance
    n         = min(len(norms), len(non_norms))
    norms     = norms.sample(n=n, random_state=RANDOM_SEED)
    non_norms = non_norms.sample(n=n, random_state=RANDOM_SEED)

    merged = pd.concat([norms, non_norms], ignore_index=True)
    # Full shuffle — no clusters of norm/non-norm
    merged = merged.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"\n{'='*50}")
    print(f"MERGED DATASET  ->  {len(merged)} rows")
    print(f"  Norm     (1): {merged['label'].sum()}")
    print(f"  Non-norm (0): {(merged['label'] == 0).sum()}")
    print(f"  Sources: {merged['source'].value_counts().to_dict()}")
    print(f"{'='*50}\n")

    merged.to_csv(os.path.join(DATA_OUT, "merged_full.csv"), index=False)
    print("Saved -> data/merged_full.csv")

    # ── Train / Val / Test split (80 / 10 / 10, stratified) ──
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
