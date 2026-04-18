"""
Downloads the Dataset
"""


from pathlib import Path
import subprocess

from config import DATA_DIR


files = {
    "reviews": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Software.jsonl.gz",
    "metadata": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Software.jsonl.gz",
}

for name, url in files.items():
    output = DATA_DIR / f"{name}_Software.jsonl.gz"
    if output.exists():
        print(f"{name} already exists, skipping")
        continue
    print(f"Downloading {name}...")
    subprocess.run(["curl", "-#", "-o", str(output), url], check=True)
    print(f"Saved to {output}")
