import pandas as pd
from pathlib import Path
from config import DATA_DIR
import pyarrow.json as paj

# Load reviews
import psutil, os

process = psutil.Process(os.getpid())
print(f"Memory: {process.memory_info().rss / 1e9:.2f} GB")
# print("Loading data")
# df = pd.read_json(DATA_DIR / 'reviews_Software.jsonl', 
#                   lines=True)

table = paj.read_json(DATA_DIR / "reviews_Software.jsonl.gz")
print(table.schema)
# print(table)
df = table.to_pandas()
print(f"Memory: {process.memory_info().rss / 1e9:.2f} GB")

print("Data loaded")

print(df)

# df = df[['user_id', 'parent_asin', 'rating', 'timestamp']]

print("=== Basic Stats ===")
print(f"Total interactions: {len(df):,}")
print(f"Unique users: {df['user_id'].nunique():,}")
print(f"Unique items: {df['parent_asin'].nunique():,}")
print(f"Sparsity: {1 - len(df) / (df['user_id'].nunique() * df['parent_asin'].nunique()):.4%}")

print("\n=== Interaction Distribution ===")
user_counts = df.groupby('user_id').size()
print(f"Avg interactions per user: {user_counts.mean():.1f}")
print(f"Median: {user_counts.median():.1f}")
print(f"90th percentile: {user_counts.quantile(0.9):.0f}")
print(f"99th percentile: {user_counts.quantile(0.99):.0f}")
print(f"Max: {user_counts.max()}")

print("\n=== User Buckets ===")
print(f"Users with 1 interaction:   {(user_counts == 1).sum():,}")
print(f"Users with 2-4 interactions:{((user_counts >= 2) & (user_counts < 5)).sum():,}")
print(f"Users with 5-20:            {((user_counts >= 5) & (user_counts <= 20)).sum():,}")
print(f"Users with 20+:             {(user_counts > 20).sum():,}")

print("\n=== Rating Distribution ===")
print(df['rating'].value_counts().sort_index())

print("\n=== Timestamp Range ===")
df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
print(f"Earliest: {df['date'].min()}")
print(f"Latest: {df['date'].max()}")

print("\n=== Item Interaction Distribution ===")
item_counts = df.groupby('parent_asin').size()
print(f"Avg interactions per item: {item_counts.mean():.1f}")
print(f"Items with < 5 interactions: {(item_counts < 5).sum():,}")
print(f"Items with 20+ interactions: {(item_counts >= 20).sum():,}")