"""
16 test items (0.13%) appeared exclusively as held-out interactions
with no training signal. These users were excluded from evaluation — standard
practice in leave-one-out protocols.

```python
import pandas as pd
from config import DATA_DIR


train_df = pd.read_csv(DATA_DIR / "train.csv")  # user_id, parent_asin, rating, timestamp
test_df = pd.read_csv(DATA_DIR / "test.csv")  # user_id, parent_asin, rating, timestamp

train_items = set(train_df['parent_asin'].unique())
test_items  = set(test_df['parent_asin'].unique())

truly_missing = test_items - train_items

print(f"Total test items:                {len(test_items):,}")
print(f"Test items also in train:        {len(test_items & train_items):,}")
print(f"Test items NOT in train at all:  {len(truly_missing):,}")
```

```bash
Total test items:                12,646
Test items also in train:        12,630
Test items NOT in train at all:  16
```
> 16 test items (0.13%) appeared exclusively as held-out interactions with no training signal. These users were excluded from evaluation — standard practice in leave-one-out protocols.


This file outputs. A few embeddings files. Here is a list of them

| File | Used by | Contents |
|---|---|---|
| `item_embeddings.csv` | C++ | `parent_asin, d0, d1, ... d383` |
| `item_embedding_index.csv` | C++ | `parent_asin, embedding_idx` — maps asin → row |
| `item_embeddings.npy` | Python eval | raw matrix for Cornac / recall@K scripts |
| `checkpoints/chunk_N.npy` | Recovery | resume if process crashes mid-way |

"""

import pyarrow.json as paj
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from config import DATA_DIR, EMBEDDINGS_DIR
import os


# Load Reviews
table = paj.read_json(DATA_DIR / "reviews_Software.jsonl.gz")
df = table.to_pandas()

# Filter them
df = df.sort_values(['user_id', 'timestamp'])

user_counts = df.groupby('user_id').size()
df = df[df['user_id'].isin(user_counts[user_counts >= 5].index)]

item_counts = df.groupby('parent_asin').size()
df = df[df['parent_asin'].isin(item_counts[item_counts >= 5].index)]

# re-filter users after item filtering
user_counts = df.groupby('user_id').size()
df = df[df['user_id'].isin(user_counts[user_counts >= 5].index)]

# For leave one out split
df['rank']     = df.groupby('user_id')['timestamp'].rank(method='first', ascending=True)
df['max_rank'] = df.groupby('user_id')['rank'].transform('max')

train_df = df[df['rank'] < df['max_rank']] 
test_df  = df[df['rank'] == df['max_rank']]

# Load metadata
meta_table = paj.read_json(DATA_DIR / "metadata_Software.jsonl.gz")
meta_slim = meta_table.to_pandas()[
    ['parent_asin', 'title', 'description', 'features', 'categories', 'store', 'details']
].copy()  # .copy() immediately to avoid SettingWithCopyWarning


def build_meta_text(row):
    parts = []

    if pd.notna(row['title']) and row['title']:
        parts.append(str(row['title']))

    if isinstance(row['features'], list) and row['features']:
        parts.append(". ".join(row['features']))

    if isinstance(row['description'], list) and row['description']:
        # capped at 500 chars to leave budget for other fields
        # full combined_text is capped at 2000 chars (MiniLM's ~512 token limit)
        parts.append(" ".join(row['description'])[:500])

    if isinstance(row['categories'], list) and row['categories']:
        # categories = [["Software", "Utilities", "Disk Tools"]] — a list of lists
        # this flattens it to ["Software", "Utilities", "Disk Tools"]
        # isinstance guard handles rare cases where a sub-entry is a plain string
        flat = [c for sub in row['categories'] for c in (sub if isinstance(sub, list) else [sub])]
        parts.append(", ".join(flat))

    if pd.notna(row['store']) and row['store']:
        parts.append(f"by {row['store']}")

    if pd.notna(row['details']) and row['details']:
        # capped at 300 chars — key-value pairs, diminishing returns after a few
        parts.append(str(row['details'])[:300])

    return " | ".join(parts)


meta_slim['meta_text'] = meta_slim.apply(build_meta_text, axis=1)

# build review text for embedding generation
train_df = train_df.copy()
train_df['review_text'] = (
    train_df['title'].fillna('') + ". " + train_df['text'].fillna('')
).str.strip()

review_texts = (
    train_df.groupby('parent_asin')['review_text']
    .apply(lambda x: " [SEP] ".join(x.iloc[:5]))   # max 5 reviews per item
    .str[:1500]
    .reset_index()
)
review_texts.columns = ['parent_asin', 'review_text']

# Build item universe and merge review with metadata
all_items = pd.DataFrame({
    'parent_asin': pd.concat([
        train_df['parent_asin'],
        meta_slim['parent_asin']
    ]).unique()
})

item_texts = all_items \
    .merge(review_texts, on='parent_asin', how='left') \
    .merge(meta_slim[['parent_asin', 'meta_text']], on='parent_asin', how='left')

item_texts['combined_text'] = (
    item_texts['meta_text'].fillna('') +
    " [SEP] " +
    item_texts['review_text'].fillna('')
).str.strip().str[:2000]

# Categorise text source for diagnostics
item_texts['text_source'] = 'both'
item_texts.loc[item_texts['review_text'].isna(), 'text_source'] = 'metadata_only'
item_texts.loc[item_texts['meta_text'].isna(),   'text_source'] = 'reviews_only'

print(item_texts['text_source'].value_counts())
print(f"\nItems to embed:       {len(item_texts):,}")
print(f"  With metadata:      {item_texts['meta_text'].notna().sum():,}")
print(f"  Without metadata:   {item_texts['meta_text'].isna().sum():,}")

# Drop the 16 unevaluable test items
embeddable = set(item_texts['parent_asin'])
test_clean = test_df[['user_id', 'parent_asin']][
    test_df['parent_asin'].isin(embeddable)
]

# Save lean CSVs for C++
train_df[['user_id', 'parent_asin', 'rating', 'timestamp']].to_csv(DATA_DIR / "train.csv", index=False)
test_clean.to_csv(DATA_DIR / "test.csv", index=False)

# 7. Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")   # 384-dim, fast on CPU

BATCH_SIZE     = 128
CHUNK_SIZE     = 5_000
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

texts = item_texts['combined_text'].tolist()
asins = item_texts['parent_asin'].tolist()

all_embeddings = []

for chunk_idx, start in enumerate(range(0, len(texts), CHUNK_SIZE)):
    ckpt        = CHECKPOINT_DIR / f"chunk_{chunk_idx}.npy"
    chunk_texts = texts[start : start + CHUNK_SIZE]

    if os.path.exists(ckpt):
        emb = np.load(ckpt)
        print(f"Chunk {chunk_idx}: loaded from checkpoint ({emb.shape[0]} items)")
    else:
        emb = model.encode(
            chunk_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # unit vectors → cosine = dot product in C++
        )
        emb = emb.astype(np.float32)
        np.save(ckpt, emb)
        print(f"Chunk {chunk_idx}: embedded {emb.shape[0]} items")

    all_embeddings.append(emb)

# 8. Save outputs
embeddings = np.vstack(all_embeddings)   # (n_items, 384)
print(f"\nFinal embedding matrix: {embeddings.shape}")

emb_df = pd.DataFrame(embeddings, columns=[f"d{i}" for i in range(embeddings.shape[1])])
emb_df.insert(0, "parent_asin", asins)
emb_df.to_csv(EMBEDDINGS_DIR / "item_embeddings.csv", index=False)

np.save(EMBEDDINGS_DIR / "item_embeddings.npy", embeddings)

pd.DataFrame({'parent_asin': asins, 'embedding_idx': range(len(asins))}) \
  .to_csv(EMBEDDINGS_DIR / "item_embedding_index.csv", index=False)

print("Done.")
print(f"  item_embeddings.csv       → {embeddings.shape[0]} items × {embeddings.shape[1]} dims")
print(f"  item_embedding_index.csv  → asin to row number mapping for C++")
