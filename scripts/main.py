import pyarrow.json as paj
import pandas as pd
from config import DATA_DIR

# Load metadata
meta_table = paj.read_json(DATA_DIR / "metadata_Software.jsonl.gz")
meta_df = meta_table.to_pandas()

print(meta_df.shape)
print(meta_df.columns.tolist())

# Quick quality check
print(meta_df[['title', 'description', 'features', 'categories', 'store']].isnull().mean())

# Load your already-filtered train_df
train_df = pd.read_csv(DATA_DIR / "train.csv")

# --- Build metadata text per item ---
def build_meta_text(row):
    parts = []
    
    if pd.notna(row.get('title')) and row['title']:
        parts.append(row['title'])
    
    # features is a list like ["Works on Windows", "Requires 4GB RAM"]
    if isinstance(row.get('features'), list) and row['features']:
        parts.append(". ".join(row['features']))
    
    # description is also a list
    if isinstance(row.get('description'), list) and row['description']:
        parts.append(" ".join(row['description'])[:500])  # cap it
    
    # categories like [["Software", "Utilities"]]
    if isinstance(row.get('categories'), list) and row['categories']:
        flat_cats = [c for sublist in row['categories'] for c in sublist]
        parts.append(", ".join(flat_cats))
    
    if pd.notna(row.get('store')) and row['store']:
        parts.append(f"by {row['store']}")
    
    return " | ".join(parts)

meta_df['meta_text'] = meta_df.apply(build_meta_text, axis=1)

# --- Build review text per item (from train only — no leakage) ---
train_df["review_text"] = (
    train_df["title"].fillna("") + ". " + train_df["text"].fillna("")
).str.strip()

review_texts = (
    train_df.groupby("parent_asin")["review_text"]
    .apply(lambda x: " [SEP] ".join(x.iloc[:5]))
    .str[:1500]
    .reset_index()
)
review_texts.columns = ["parent_asin", "review_text"]

# --- Merge metadata + reviews ---
item_texts = review_texts.merge(
    meta_df[['parent_asin', 'meta_text']],
    on='parent_asin',
    how='left'    # keep all items even if metadata is missing
)

# Combine: metadata first (authoritative), then reviews (social proof)
item_texts['combined_text'] = (
    item_texts['meta_text'].fillna('') + 
    " [SEP] " + 
    item_texts['review_text'].fillna('')
).str.strip().str[:2000]   # MiniLM budget

print(f"Items with metadata:    {item_texts['meta_text'].notna().sum():,}")
print(f"Items without metadata: {item_texts['meta_text'].isna().sum():,}")
print("\nSample combined text:")
print(item_texts['combined_text'].iloc[0])