"""
data_prep.py
============
Builds the final merged dataset for the P3 Norm Classifier project.

Sources:
  NORM (label=1):
    - culturebank_reddit.csv           → context + cultural_group + actor_behavior, agreement >= 0.7
    - culturebank_tiktok.csv           → same; KEEP_TOPICS filter removes observational rows
    - normad_etiquette_final_data.csv  → Rule-of-Thumb where Gold Label == 'yes' (~900 rows)
    - NormBank.csv                     → "In a {setting}, people {behavior}." where label in [1,2]
    - cultureatlas.parquet             → positive_sample cultural statements (~3,100 rows)

  NON-NORM / Hard Negatives (label=0):
    - culturebank (agreement <= 0.2)  → HARDEST: same sentence structure as norms but
                                         agreement explicitly marks behavior as non-norm
    - NormBank.csv (label == 0)       → HARDEST: "In a {setting}, people {behavior}." where
                                         behavior is taboo — identical structure to norm
                                         sentences but opposite label
    - stereotype.parquet              → StereoSet anti-stereotype sentences (cultural group + behavior)
    - crows_pairs_anonymized.csv      → CrowS-Pairs sent_less (less-stereotyping sentences)
    - squad.parquet                   → factual human-activity passages (sentence-split)
    - wikipedia.parquet               → structural + general factual sentences

  AG News REMOVED — Reuters/AP formatting was a trivial non-norm signal.

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

REDDIT_PATH       = os.path.join(RAW_DIR, "culturebank_reddit.csv")
TIKTOK_PATH       = os.path.join(RAW_DIR, "culturebank_tiktok.csv")
STEREOSET_PATH    = os.path.join(RAW_DIR, "stereotype.parquet")
CROWS_PATH        = os.path.join(RAW_DIR, "crows_pairs_anonymized.csv")
SQUAD_PATH        = os.path.join(RAW_DIR, "squad.parquet")
WIKIPEDIA_PATH    = os.path.join(RAW_DIR, "wikipedia.parquet")
NORMAD_PATH       = os.path.join(RAW_DIR, "normad_etiquette_final_data.csv")
NORMBANK_PATH     = os.path.join(RAW_DIR, "NormBank.csv")
CULTUREATLAS_PATH = os.path.join(RAW_DIR, "cultureatlas.parquet")

# ─── Config ───────────────────────────────────────────────────────────────────
AGREEMENT_THRESH     = 0.70
HARD_NEG_THRESH      = 0.20   # CultureBank rows at or below this are explicit non-norms
MIN_WORDS            = 5
MAX_WORDS            = 80
RANDOM_SEED          = 42
NORMBANK_NORM_CAP    = 8000   # max normal/expected NormBank rows added as norms
NORMBANK_NEG_CAP     = 8000   # max taboo NormBank rows added as hard negatives

# Sentences with any of these are too norm-like for the non-norm set.
# NOTE: NOT applied to CultureBank hard negatives — those must look like norms.
NORM_KEYWORDS = [
    "should", "must", "expected to", "it is customary",
    "it is common to", "traditionally", "norm", "etiquette",
    "polite", "rude", "respectful", "greeting", "bow", "tip",
]

# Topics to KEEP for norm rows — clear, prescriptive behavioral categories.
# Excluded per CultureBank paper limitations: "Cultural Exchange", "Migration and
# Cultural Adaptation", and "Cultural and Environmental Appreciation" describe
# reactions/observations (culture shock) rather than prescriptive behavioral norms.
# Topic strings match the actual values in the CSV exactly.
KEEP_TOPICS = {
    "Social Norms and Etiquette",
    "Community and Identity",
    "Food and Dining",
    "Communication and Language",
    "Cultural Traditions and Festivals",
    "Miscellaneous",
    "Consumer Behavior",
    "Health and Hygiene",
    "Finance and Economy",
    "Entertainment and Leisure",
    "Education and Technology",
    "Family Dynamics",
    "Social Interactions",
    "Relationships and Marriage",
    "Lifestyles",
    "Family Traditions and Heritage",
    "Household and Daily Life",
    "Drinking and Alcohol",
    "Beauty and Fashion",
    "Safety and Security",
    "Workplace",
    "Work-Life Balance",
    "Religious Practices",
    "Sports and Recreation",
    "Transportation",
    "Time Management and Punctuality",
    "Travelling",
    "Social Infrastructure",
    "Pet and Animal Care",
    "Humor and Storytelling",
    "Dress Codes",
    "Housing and Interior Design",
    "Environmental Adaptation and Sustainability",
}

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
    sl = sentence.lower()
    return not any(kw in sl for kw in NORM_KEYWORDS)

def split_sentences(text: str) -> list:
    text = clean_text(text)
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in parts if s.strip()]


# ─── NORM data ────────────────────────────────────────────────────────────────
def build_norm_sentence(row, template: str) -> str:
    """
    Construct a natural norm sentence from CultureBank fields.
    Two templates break the single-pattern structural shortcut:

    Template A (50%): "In {context}, {group} {behavior}."
    Template B (50%): "{group} {behavior} in {context}."

    Why NOT eval_whole_desc?
      GPT-generated boilerplate ending in "widely regarded as a common practice
      within the sampled population" — a lexical fingerprint that trivialises the task.
    """
    context  = clean_text(str(row.get("context", "")))
    group    = clean_text(str(row.get("cultural group", "")))
    behavior = clean_text(str(row.get("actor_behavior", "")))

    for val in [context, group, behavior]:
        if not val or val in ("nan", "unknown"):
            if val == behavior:
                return ""   # behavior is required; others degrade gracefully

    context  = "" if context  in ("nan", "unknown") else context
    group    = "" if group    in ("nan", "unknown") else group

    if template == "A":
        if context and group:
            # Avoid "In in X" when context already starts with a preposition
            if re.match(r'^(in|at|during|within|across|among)\b', context, re.I):
                prefix = context[0].upper() + context[1:]
                sentence = f"{prefix}, {group} {behavior}."
            else:
                sentence = f"In {context}, {group} {behavior}."
        elif group:
            sentence = f"{group} {behavior}."
        else:
            sentence = f"{behavior}."
    else:  # template B  — "{group} {behavior} in {context}."
        # Strip any leading preposition from context to avoid "in in X" or "in at X"
        ctx_b = re.sub(r'^(in|at|during|within|across|among)\s+', '', context, flags=re.I)
        if ctx_b and group:
            sentence = f"{group} {behavior} in {ctx_b}."
        elif group:
            sentence = f"{group} {behavior}."
        else:
            sentence = f"{behavior}."

    return sentence[0].upper() + sentence[1:]


def load_norms() -> pd.DataFrame:
    dfs = []
    for path, source_name in [(REDDIT_PATH, "culturebank_reddit"),
                               (TIKTOK_PATH, "culturebank_tiktok")]:
        df = pd.read_csv(path)
        print(f"[{source_name}] Raw rows: {len(df)}")

        df = df[df["agreement"] >= AGREEMENT_THRESH].copy()
        print(f"[{source_name}] After agreement >= {AGREEMENT_THRESH}: {len(df)}")

        # Drop observational/culture-shock topics — these describe reactions, not norms
        if "topic" in df.columns:
            before = len(df)
            df = df[df["topic"].isin(KEEP_TOPICS)].copy()
            print(f"[{source_name}] After topic filter: {len(df)}  (removed {before - len(df)} observational rows)")

        # Assign templates 50/50 by row position (deterministic, no shuffle needed)
        df = df.reset_index(drop=True)
        df["_tmpl"] = df.index.map(lambda i: "A" if i % 2 == 0 else "B")
        df["sentence"] = df.apply(lambda r: build_norm_sentence(r, r["_tmpl"]), axis=1)

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
    print(f"\nTotal CultureBank NORM rows (deduplicated): {len(norms)}")
    return norms


def load_normad_norms() -> pd.DataFrame:
    """
    NormAd — Rule-of-Thumb sentences where Gold Label == 'yes'.

    Each Rule-of-Thumb is an already-formed prescriptive norm statement, e.g.:
      "It is respectful to greet everyone present before starting any social interaction."
    These are clean, direct, culturally grounded norms across 75 countries.
    Only 'yes'-labeled rows are used; 'no' rows describe norm violations and
    'neutral' rows are ambiguous.
    """
    print(f"\n[normad] Loading from {NORMAD_PATH}...")
    df = pd.read_csv(NORMAD_PATH)
    print(f"[normad] Raw rows: {len(df)}")

    df = df[df["Gold Label"] == "yes"].copy()
    print(f"[normad] After Gold Label == 'yes': {len(df)}")

    df["sentence"] = df["Rule-of-Thumb"].apply(clean_text)
    df["cultural_group"] = df["Country"].apply(
        lambda x: clean_text(str(x)) if pd.notna(x) else "unknown"
    )
    df["label"]  = 1
    df["source"] = "normad"

    df = df[df["sentence"].apply(is_valid)]
    df = df.drop_duplicates(subset="sentence")
    print(f"[normad] Final rows: {len(df)}")
    return df[["sentence", "label", "cultural_group", "source"]]


def load_normbank_norms() -> pd.DataFrame:
    """
    NormBank — 'normal' and 'expected' rows (label in [1, 2]) from train split.

    Sentence construction: "In a {setting}, people {behavior}."
    This mirrors the CultureBank Template A structure exactly, so NormBank norms
    blend seamlessly with CultureBank norms in the training set.
    Capped at NORMBANK_NORM_CAP rows to prevent NormBank from dominating the norm pool.
    Train split only to avoid test contamination.
    """
    print(f"\n[normbank_norms] Loading from {NORMBANK_PATH}...")
    df = pd.read_csv(NORMBANK_PATH)
    print(f"[normbank_norms] Raw rows: {len(df)}")

    df = df[(df["label"].isin([1, 2])) & (df["split"] == "train")].copy()
    print(f"[normbank_norms] Normal/expected (train split): {len(df)}")

    def build_nb_sentence(row):
        setting  = clean_text(str(row.get("setting",  "")))
        behavior = clean_text(str(row.get("behavior", "")))
        if not behavior or behavior == "nan":
            return ""
        if setting and setting != "nan":
            return f"In a {setting}, people {behavior}."
        return f"People {behavior}."

    df["sentence"] = df.apply(build_nb_sentence, axis=1)
    df = df[df["sentence"].apply(is_valid)]
    df = df.drop_duplicates(subset="sentence")

    if len(df) > NORMBANK_NORM_CAP:
        df = df.sample(n=NORMBANK_NORM_CAP, random_state=RANDOM_SEED)

    df["cultural_group"] = "none"
    df["label"]  = 1
    df["source"] = "normbank"
    print(f"[normbank_norms] Final rows: {len(df)}")
    return df[["sentence", "label", "cultural_group", "source"]]


def load_cultureatlas_norms() -> pd.DataFrame:
    """
    CultureAtlas — positive_sample cultural statements.

    Each positive_sample is a complete, factually grounded sentence about a
    specific cultural practice (dress, customs, rituals, etc.) for a named country.
    e.g. "In some Pashtun cultures, a boy marks his start of adulthood by being
          allowed to wear a turban, which holds special significance."
    Country field maps to cultural_group.
    """
    print(f"\n[cultureatlas] Loading from {CULTUREATLAS_PATH}...")
    df = pd.read_parquet(CULTUREATLAS_PATH)
    print(f"[cultureatlas] Raw rows: {len(df)}")

    df["sentence"] = df["positive_sample"].apply(clean_text)
    df["cultural_group"] = df["country"].apply(
        lambda x: clean_text(str(x)) if pd.notna(x) else "unknown"
    )
    df["label"]  = 1
    df["source"] = "cultureatlas"

    df = df[df["sentence"].apply(is_valid)]
    df = df.drop_duplicates(subset="sentence")
    print(f"[cultureatlas] Final rows: {len(df)}")
    return df[["sentence", "label", "cultural_group", "source"]]


def load_all_norms() -> pd.DataFrame:
    """Merge all norm sources into one deduplicated DataFrame."""
    cb_norms  = load_norms()
    na_norms  = load_normad_norms()
    nb_norms  = load_normbank_norms()
    ca_norms  = load_cultureatlas_norms()

    all_norms = pd.concat([cb_norms, na_norms, nb_norms, ca_norms], ignore_index=True)
    all_norms = all_norms.drop_duplicates(subset="sentence")
    print(f"\n{'='*50}")
    print(f"TOTAL NORM ROWS (all sources, deduplicated): {len(all_norms)}")
    print(f"  {all_norms['source'].value_counts().to_dict()}")
    print(f"{'='*50}\n")
    return all_norms


# ─── NON-NORM data (Hard Negatives) ──────────────────────────────────────────
def load_culturebank_hard_negatives() -> pd.DataFrame:
    """
    CultureBank rows with agreement <= 0.20 become label=0.

    Why this is the most powerful hard negative source:
      - Built with the SAME build_norm_sentence templates as the norm rows.
      - Contains the same cultural group names, behavioral verbs, and "In X, Y does Z."
        sentence structure as the positive class.
      - Yet the CultureBank paper explicitly marks these as non-norms: "if agreement <= 0.2,
        the behavior is NOT the norm for that group."
      - Example: "In restaurants, Americans do not tip service staff." agreement=0 → label=0
        vs. "In restaurants, Americans tip 15-20%." agreement=1 → label=1
      - Forces the model to learn prescriptive vs. descriptive meaning, not shortcuts.

    Topic filter applied: same KEEP_TOPICS as norm rows so that the only difference
    between positive and negative examples is the agreement value, not the topic.
    """
    dfs = []
    for path, source_name in [(REDDIT_PATH, "culturebank_reddit_hardneg"),
                               (TIKTOK_PATH, "culturebank_tiktok_hardneg")]:
        df = pd.read_csv(path)
        df = df[df["agreement"] <= HARD_NEG_THRESH].copy()
        print(f"[{source_name}] Rows with agreement <= {HARD_NEG_THRESH}: {len(df)}")

        if "topic" in df.columns:
            df = df[df["topic"].isin(KEEP_TOPICS)].copy()
            print(f"[{source_name}] After topic filter: {len(df)}")

        df = df.reset_index(drop=True)
        df["_tmpl"] = df.index.map(lambda i: "A" if i % 2 == 0 else "B")
        df["sentence"] = df.apply(lambda r: build_norm_sentence(r, r["_tmpl"]), axis=1)

        df["cultural_group"] = df["cultural group"].apply(
            lambda x: clean_text(str(x)) if pd.notna(x) else "unknown"
        )
        df["source"] = source_name
        df["label"]  = 0

        df = df[df["sentence"].apply(is_valid)]
        dfs.append(df[["sentence", "label", "cultural_group", "source"]])

    hard_negs = pd.concat(dfs, ignore_index=True)
    hard_negs = hard_negs.drop_duplicates(subset="sentence")
    print(f"\nTotal CultureBank hard negatives (deduplicated): {len(hard_negs)}")
    return hard_negs


def load_normbank_hard_negatives() -> pd.DataFrame:
    """
    NormBank taboo rows (label == 0) as hard negatives.

    Sentence construction is IDENTICAL to load_normbank_norms():
      "In a {setting}, people {behavior}."
    The only difference between these and norm sentences is the label.
    e.g. NORM:     "In a restaurant, people pay the bill before leaving."   label=1
         HARD NEG: "In a restaurant, people walk out without paying."       label=0

    This is structurally the hardest possible non-norm — same template, same vocab
    domain, different prescriptive meaning. Forces the model to reason semantically.
    Capped at NORMBANK_NEG_CAP. Train split only.
    """
    print(f"\n[normbank_hardneg] Loading from {NORMBANK_PATH}...")
    df = pd.read_csv(NORMBANK_PATH)

    df = df[(df["label"] == 0) & (df["split"] == "train")].copy()
    print(f"[normbank_hardneg] Taboo rows (train split): {len(df)}")

    def build_nb_sentence(row):
        setting  = clean_text(str(row.get("setting",  "")))
        behavior = clean_text(str(row.get("behavior", "")))
        if not behavior or behavior == "nan":
            return ""
        if setting and setting != "nan":
            return f"In a {setting}, people {behavior}."
        return f"People {behavior}."

    df["sentence"] = df.apply(build_nb_sentence, axis=1)
    df = df[df["sentence"].apply(is_valid)]
    df = df.drop_duplicates(subset="sentence")

    if len(df) > NORMBANK_NEG_CAP:
        df = df.sample(n=NORMBANK_NEG_CAP, random_state=RANDOM_SEED)

    df["cultural_group"] = "none"
    df["label"]  = 0
    df["source"] = "normbank_taboo"
    print(f"[normbank_hardneg] Final rows: {len(df)}")
    return df[["sentence", "label", "cultural_group", "source"]]


def load_stereoset() -> pd.DataFrame:
    """
    StereoSet — anti-stereotype sentences (gold_label=1) combined with context.

    Why this kills the shortcut:
      These sentences contain cultural group names + behavioral content
      but are factual/neutral, NOT prescriptive norms.
      e.g. "I know many people who are Russian. Most of them do not drink alcohol."
           "Many people live in Ethiopia. The people are very thin and good at
            distance running."

    gold_label mapping: 0=stereotype, 1=anti-stereotype, 2=unrelated
    We use ONLY 1 (anti-stereotype) — never stereotypes, never unrelated fillers.
    """
    print(f"\n[stereoset] Loading from local parquet...")
    st = pd.read_parquet(STEREOSET_PATH)
    print(f"[stereoset] Raw rows: {len(st)}")

    rows = []
    for _, row in st.iterrows():
        context = clean_text(str(row["context"]))
        data    = row["sentences"]                  # dict with numpy arrays
        sents   = data["sentence"]
        labels  = data["gold_label"]

        for sent, gold in zip(sents, labels):
            if int(gold) == 0:                      # anti-stereotype only (0=anti, 1=stereo, 2=unrelated)
                full = clean_text(f"{context} {sent}")
                rows.append(full)

    df = pd.DataFrame({"sentence": rows})
    df = df[df["sentence"].apply(is_valid)]
    df = df[df["sentence"].apply(is_clean_non_norm)]
    df = df.drop_duplicates(subset="sentence")

    df["label"]          = 0
    df["cultural_group"] = "none"
    df["source"]         = "stereoset"
    print(f"[stereoset] Anti-stereotype sentences kept: {len(df)}")
    return df[["sentence", "label", "cultural_group", "source"]]


def load_crows_pairs() -> pd.DataFrame:
    """
    CrowS-Pairs — sent_less column only (less-stereotyping sentence).

    Note: sent_less replaces the disadvantaged group with the advantaged group.
    Some sentences still contain negative characterisations — we apply a
    content filter to remove overtly harmful sentences before using them.
    """
    # Only keep sentences that contain a cultural/national/racial group name —
    # sentences without one add no value for closing the cultural group gap.
    CULTURAL_TERMS = re.compile(
        r'\b(American|Japanese|Indian|British|German|Italian|Chinese|Mexican|French|'
        r'Korean|Russian|Australian|Canadian|Brazilian|Nigerian|Arab|Hispanic|'
        r'African|Asian|Latino|Latina|Muslim|Jewish|Christian|Hindu|immigrant|'
        r'white|black|brown|indigenous|native)\b', re.I
    )

    # Remove sentences with overtly harmful or violent language
    BAD_CONTENT = [
        "gang", "gangster", "criminal", "crime", "violent", "attack",
        "murder", "rape", "terrorist", "bomb", "kill", "dead bodies",
        "jail", "prison", "dumb", "stupid", "ignorant", "lazy",
        "thief", "steal", "plotters", "inscrutable", "drug dealer",
        "can't be good", "alcoholic", "don't believe in",
    ]

    def is_safe(text: str) -> bool:
        tl = text.lower()
        return not any(kw in tl for kw in BAD_CONTENT)

    def has_cultural_term(text: str) -> bool:
        return bool(CULTURAL_TERMS.search(text))

    print(f"\n[crows_pairs] Loading from local CSV...")
    cp = pd.read_csv(CROWS_PATH)
    print(f"[crows_pairs] Raw rows: {len(cp)}")

    df = pd.DataFrame({"sentence": cp["sent_less"].apply(clean_text)})
    df = df[df["sentence"].apply(is_valid)]
    df = df[df["sentence"].apply(is_clean_non_norm)]
    df = df[df["sentence"].apply(is_safe)]
    df = df[df["sentence"].apply(has_cultural_term)]   # only keep culturally-grounded ones
    df = df.drop_duplicates(subset="sentence")

    df["label"]          = 0
    df["cultural_group"] = "none"
    df["source"]         = "crows_pairs"
    print(f"[crows_pairs] Sentences kept: {len(df)}")
    return df[["sentence", "label", "cultural_group", "source"]]


def load_squad(target_n: int) -> pd.DataFrame:
    """
    SQuAD context passages split into sentences.
    Factual descriptions of human activities across many domains.
    """
    print(f"\n[squad] Loading from local parquet...")
    sq = pd.read_parquet(SQUAD_PATH)
    print(f"[squad] Raw rows: {len(sq)}")

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
    Wikipedia split into two equal pools:

    Pool A — Structural hard negatives:
      Sentences starting "In [X], [Capital]..." — same template as norm sentences
      but purely factual (dates, events, statistics).
      Forces the model to look beyond sentence structure.

    Pool B — General factual sentences:
      Broad topic diversity (science, geography, history, etc.)
      Prevents topic-shortcut learning.
    """
    print(f"\n[wikipedia] Loading from local parquet...")
    wiki = pd.read_parquet(WIKIPEDIA_PATH)
    print(f"[wikipedia] Raw rows: {len(wiki)}")

    wiki["sentence"] = wiki["sentence"].apply(clean_text)
    wiki = wiki[wiki["sentence"].apply(is_valid)]
    wiki = wiki[wiki["sentence"].apply(is_clean_non_norm)]
    wiki = wiki.drop_duplicates(subset="sentence")
    print(f"[wikipedia] After keyword filter + dedup: {len(wiki)}")

    structural_mask = wiki["sentence"].str.match(r'^In [A-Z][a-z].*,\s+[A-Z][a-z]', na=False)
    pool_a = wiki[structural_mask]
    pool_b = wiki[~structural_mask]
    print(f"[wikipedia] Pool A (structural): {len(pool_a)} | Pool B (general): {len(pool_b)}")

    half  = target_n // 2
    a_n   = min(half, len(pool_a))
    b_n   = min(target_n - a_n, len(pool_b))

    sampled = pd.concat([
        pool_a.sample(n=a_n, random_state=RANDOM_SEED),
        pool_b.sample(n=b_n, random_state=RANDOM_SEED),
    ], ignore_index=True)

    sampled["label"]          = 0
    sampled["cultural_group"] = "none"
    sampled["source"]         = "wikipedia"
    print(f"[wikipedia] Final sample: {len(sampled)}  (structural={a_n}, general={b_n})")
    return sampled[["sentence", "label", "cultural_group", "source"]]


def load_non_norms(target_count: int) -> pd.DataFrame:
    """
    Non-norm budget (priority order, all fixed sources used in full):
      CultureBank hard neg  — agreement <= 0.2, same template as norms
      NormBank taboo        — label==0, same "In a {setting}, people {behavior}." template
      StereoSet             — cultural group + behavior, not prescriptive
      CrowS-Pairs           — cultural group + behavior, less-stereotyping
      SQuAD                 — fills ~30% of remainder
      Wikipedia             — fills ~70% of remainder (50% structural, 50% general)
    """
    cb_hard_df = load_culturebank_hard_negatives()
    nb_hard_df = load_normbank_hard_negatives()
    stereo_df  = load_stereoset()
    crows_df   = load_crows_pairs()

    filled    = len(cb_hard_df) + len(nb_hard_df) + len(stereo_df) + len(crows_df)
    remaining = max(0, target_count - filled)
    sq_n      = remaining // 3
    wiki_n    = remaining - sq_n

    sq_df   = load_squad(sq_n)
    wiki_df = load_wikipedia(wiki_n)

    non_norms = pd.concat(
        [cb_hard_df, nb_hard_df, stereo_df, crows_df, sq_df, wiki_df],
        ignore_index=True
    )
    non_norms = non_norms.drop_duplicates(subset="sentence")
    print(f"\nTotal NON-NORM rows (deduplicated): {len(non_norms)}")
    return non_norms


# ─── Merge & split ────────────────────────────────────────────────────────────
def build_dataset():
    print("\n" + "="*60)
    print("  P3 Norm Classifier — Data Preparation")
    print("="*60 + "\n")

    norms     = load_all_norms()
    non_norms = load_non_norms(target_count=len(norms))

    # Trim to exact 1:1 balance
    n         = min(len(norms), len(non_norms))
    norms     = norms.sample(n=n, random_state=RANDOM_SEED)
    non_norms = non_norms.sample(n=n, random_state=RANDOM_SEED)

    merged = pd.concat([norms, non_norms], ignore_index=True)
    # Full shuffle — no clusters
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
