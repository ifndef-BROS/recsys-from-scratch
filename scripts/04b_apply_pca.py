import numpy as np
import csv
from pathlib import Path
import os

DATA_DIR = Path("../data") if os.path.exists("../data/embeddings") else Path("data")

print("1. Loading 384D embeddings...")
emb = np.load(DATA_DIR / "embeddings" / "item_embeddings.npy")

print("2. Performing PCA from scratch using Linear Algebra...")
mean = np.mean(emb, axis=0)
centered = emb - mean
cov_matrix = np.cov(centered, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
sorted_idx = np.argsort(eigenvalues)[::-1]
top_64_eigenvectors = eigenvectors[:, sorted_idx][:, :64]
emb_64 = np.dot(centered, top_64_eigenvectors)

print("3. Normalizing vectors...")
norms = np.linalg.norm(emb_64, axis=1, keepdims=True)
emb_64 = emb_64 / np.clip(norms, a_min=1e-9, a_max=None)

print("4. Saving NPY...")
np.save(DATA_DIR / "embeddings" / "item_embeddings.npy", emb_64)

print("5. Mapping ASINs and Saving perfectly formatted CSV...")
# Read the ASINs from the index file
asin_map = {}
with open(DATA_DIR / "embeddings" / "item_embedding_index.csv", "r") as f:
    reader = csv.reader(f)
    next(reader) # skip header
    for row in reader:
        if len(row) >= 2:
            asin_map[int(row[1])] = row[0]

# Write the new CSV with Header and ASINs
with open(DATA_DIR / "embeddings" / "item_embeddings.csv", "w", newline="") as f:
    writer = csv.writer(f)
    # Write header: parent_asin, d0, d1...
    writer.writerow(["parent_asin"] +[f"d{i}" for i in range(64)])
    
    # Write rows: ASIN, val0, val1...
    for i in range(len(emb_64)):
        asin = asin_map[i]
        # Format floats to 6 decimal places to keep file size manageable <-- OLD
        # Now, changed to .8e because self similar test case was failing in case of kd tree
        # the hypothesis is that the case fails because two very similar items have almost zero distance between them
        row_data = [asin] +[f"{val:.8e}" for val in emb_64[i]]
        writer.writerow(row_data)

print("Done! You are ready to run C++!")