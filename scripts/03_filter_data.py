import pandas as pd
import pyarrow.json as paj
from config import DATA_DIR

table = paj.read_json(DATA_DIR / "reviews_Software.jsonl.gz")
df = table.to_pandas()

# 1. Sort by timestamp — critical for leave-one-out correctness
df = df.sort_values(["user_id", "timestamp"])

# 2. Keep only users with 5+ interactions
#    (users with 1-2 interactions make poor train/test splits)
user_counts = df.groupby("user_id").size()
valid_users = user_counts[user_counts >= 5].index
df = df[df["user_id"].isin(valid_users)]

# 3. Keep items that appear at least 5 times
#    (cold items will never be retrieved anyway)
item_counts = df.groupby("parent_asin").size()
valid_items = item_counts[item_counts >= 5].index
df = df[df["parent_asin"].isin(valid_items)]

# 4. Re-filter users (some may now have < 5 after item filtering)
user_counts = df.groupby("user_id").size()
valid_users = user_counts[user_counts >= 5].index
df = df[df["user_id"].isin(valid_users)]

print(f"Users:         {df['user_id'].nunique():,}")
print(f"Items:         {df['parent_asin'].nunique():,}")
print(f"Interactions:  {len(df):,}")

# Last interaction per user = test, everything before = train
df["rank"] = df.groupby("user_id")["timestamp"].rank(method="first", ascending=True)
df["max_rank"] = df.groupby("user_id")["rank"].transform("max")

test_df = df[df["rank"] == df["max_rank"]]  # last item per user
train_df = df[df["rank"] < df["max_rank"]]  # all prior items

print(f"Train interactions: {len(train_df):,}")
print(f"Test  interactions: {len(test_df):,}")

# Save
train_df[["user_id", "parent_asin", "rating", "timestamp"]].to_csv(
    DATA_DIR / "train.csv", index=False
)
test_df[["user_id", "parent_asin"]].to_csv(DATA_DIR / "test.csv", index=False)
