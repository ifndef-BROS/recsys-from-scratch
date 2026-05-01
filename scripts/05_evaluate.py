"""
05_evaluate.py

Evaluates recommendation results against ground truth.
Computes Recall@K and NDCG@K for cosine sort and Adaptive MMR outputs.
Also runs Cornac baselines for comparison.

Usage:
    uv run python 05_evaluate.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config import DATA_DIR

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_k(recommended: list, ground_truth: str, k: int) -> float:
    """1 if ground truth appears in top-k recommendations, else 0."""
    return float(ground_truth in recommended[:k])

def ndcg_at_k(recommended: list, ground_truth: str, k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at K.
    For a single relevant item, NDCG@K = 1/log2(rank+1) if item in top-K.
    Normalised by ideal DCG (item at rank 1) = 1/log2(2) = 1.
    """
    for i, asin in enumerate(recommended[:k]):
        if asin == ground_truth:
            return 1.0 / np.log2(i + 2)  # rank is 0-indexed, +2 for log2
    return 0.0

def evaluate(results_df: pd.DataFrame,
             ground_truth: dict,
             ks: list = [5, 10, 20]) -> pd.DataFrame:
    """
    Evaluates a results dataframe against ground truth.

    results_df columns: user_id, rec_1, rec_2, ..., rec_K
    ground_truth: dict user_id → held-out asin

    Returns a dataframe of metrics per K.
    """
    rec_cols = [c for c in results_df.columns if c.startswith('rec_')]

    rows = []
    for k in ks:
        recalls, ndcgs = [], []
        for _, row in results_df.iterrows():
            user_id = row['user_id']
            if user_id not in ground_truth:
                continue
            gt_asin      = ground_truth[user_id]
            recommended  = [row[c] for c in rec_cols[:k] if pd.notna(row[c])]
            recalls.append(recall_at_k(recommended, gt_asin, k))
            ndcgs.append(ndcg_at_k(recommended, gt_asin, k))

        rows.append({
            'K':        k,
            'Recall@K': np.mean(recalls),
            'NDCG@K':   np.mean(ndcgs),
            'n_users':  len(recalls)
        })

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Load ground truth
# ---------------------------------------------------------------------------

test_df      = pd.read_csv(DATA_DIR / 'test.csv')
ground_truth = dict(zip(test_df['user_id'], test_df['parent_asin']))

print(f"Ground truth users: {len(ground_truth):,}")

# ---------------------------------------------------------------------------
# Evaluate C++ outputs
# ---------------------------------------------------------------------------

results = {}

for name, path in [
    ('cosine_sort',   DATA_DIR / 'results_cosine.csv'),
    ('adaptive_mmr',  DATA_DIR / 'results_mmr.csv'),
]:
    if not Path(path).exists():
        print(f"[SKIP] {path} not found")
        continue

    df      = pd.read_csv(path)
    metrics = evaluate(df, ground_truth)
    results[name] = metrics
    print(f"\n── {name} ──")
    print(metrics.to_string(index=False))

# ---------------------------------------------------------------------------
# Cornac baselines
# ---------------------------------------------------------------------------

try:
    import cornac
    from cornac.eval_methods import BaseMethod
    from cornac.metrics import Recall, NDCG

    print("\n── Cornac baselines ──")

    # Build cornac dataset from train
    train_df = pd.read_csv(DATA_DIR / 'train.csv')

    # Cornac needs numeric user/item ids
    users  = {u: i for i, u in enumerate(train_df['user_id'].unique())}
    items  = {a: i for i, a in enumerate(train_df['parent_asin'].unique())}

    train_data = [
        (users[r.user_id], items[r.parent_asin], r.rating)
        for _, r in train_df.iterrows()
        if r.user_id in users and r.parent_asin in items
    ]

    test_data = [
        (users[r.user_id], items[r.parent_asin], 1.0)
        for _, r in test_df.iterrows()
        if r.user_id in users and r.parent_asin in items
    ]

    eval_method = BaseMethod.from_splits(
        train_data=train_data,
        test_data=test_data,
        rating_threshold=1.0,
        exclude_unknowns=True,
        verbose=False
    )

    metrics   = [Recall(k=10), NDCG(k=10)]
    baselines = [
        cornac.models.MostPop(),                          # popularity
        cornac.models.ItemKNN(k=50, similarity='cosine'), # item-item CF
        cornac.models.BPR(k=64, max_iter=200),            # BPR
    ]

    cornac.Experiment(
        eval_method=eval_method,
        models=baselines,
        metrics=metrics,
        user_based=True
    ).run()

except ImportError:
    print("[SKIP] cornac not installed — run: uv add cornac")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print("\n══════════════════════════════════════════════════")
print("  Summary @ K=10")
print("══════════════════════════════════════════════════")
print(f"{'Method':<20} {'Recall@10':>10} {'NDCG@10':>10}")
print("─" * 42)

for name, metrics_df in results.items():
    row = metrics_df[metrics_df['K'] == 10].iloc[0]
    print(f"{name:<20} {row['Recall@K']:>10.4f} {row['NDCG@K']:>10.4f}")
    

# At the end of 05_evaluate.py — append to experiment log
import json
from datetime import datetime

log_entry = {
    'timestamp':    datetime.now().isoformat(),
    'dim':          384,          # change to 64 after PCA
    'weight_scheme': 'signed',    # or 'normalised'
    'metrics': {
        name: metrics_df[metrics_df['K'] == 10].iloc[0].to_dict()
        for name, metrics_df in results.items()
    }
}

log_path = DATA_DIR / 'experiment_log.jsonl'
with open(log_path, 'a') as f:
    f.write(json.dumps(log_entry) + '\n')

print(f"\nLogged to {log_path}")