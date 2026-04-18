"""
AI Generated
"""

import pandas as pd

# Use the rating-only files, much smaller
urls = {
    "Software": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/0core/rating_only/Software.csv.gz",
    "Video_Games": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/0core/rating_only/Video_Games.csv.gz",
    "Musical_Instruments": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/0core/rating_only/Musical_Instruments.csv.gz",
    "Sports_and_Outdoors": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/0core/rating_only/Sports_and_Outdoors.csv.gz",
    "All Beauty": "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/All_Beauty.csv.gz",
}

for name, url in urls.items():
    try:
        df = pd.read_csv(
            url,
            header=None,
            names=["user", "item", "rating", "timestamp"],
            dtype={"user": str, "item": str, "rating": float, "timestamp": str},
            low_memory=False,
        )
        sizes = df.groupby("user").size()
        print(f"\n{name}")
        print(f"  Users: {df['user'].nunique():,}")
        print(f"  Items: {df['item'].nunique():,}")
        print(f"  Interactions: {len(df):,}")
        print(f"  Avg per user: {sizes.mean():.1f}")
        print(f"  Median per user: {sizes.median():.1f}")
        print(f"  Users with 5+ interactions: {(sizes >= 5).sum():,}")
    except Exception as e:
        print(f"{name}: failed — {e}")
