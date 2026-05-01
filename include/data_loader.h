/**
 * @file    data_loader.h
 * @brief   In-memory loading of embedding and interaction data for inference.
 *
 * Parses the three CSV artifacts produced by the Python preprocessing pipeline
 * (scripts/04_item_embedding.py) into STL structures consumed by the retrieval
 * and ranking stages.
 *
 * Data flow:
 * ┌─────────────────────────────┬──────────────────────────────────────────┐
 * │ File                        │ Loaded into                              │
 * ├─────────────────────────────┼──────────────────────────────────────────┤
 * │ item_embeddings.csv         │ vector<array<float, DIM>>                │
 * │ item_embedding_index.csv    │ unordered_map asin ↔ row                 │
 * │ train.csv                   │ unordered_map user → vector<Interaction> │
 * │ test.csv                    │ unordered_map user → ground truth asin   │
 * └─────────────────────────────┴──────────────────────────────────────────┘
 *
 * Assumptions:
 *   - train.csv rows are ordered by timestamp ascending (guaranteed by
 *     03_filter_data.py). Insertion order encodes recency — do not re-sort.
 *   - All embeddings are L2-normalised. Dot product == cosine similarity.
 *   - DIM must match the dimensionality used during embedding generation.
 *     Change it here if PCA was applied (e.g. 384 → 64).
 *
 * @author  anshulbadhani
 * @date    19/4/2026
 */

#pragma once

#include <array>
#include <string>
#include <vector>
#include <unordered_map>

using std::vector,
    std::string,
    std::unordered_map,
    std::array;

    

/**
 * Dimensionality of item embedding vectors.
 *
 * Default: 384 (all-MiniLM-L6-v2 output dimension).
 * Set to 64 if PCA compression was applied in 04_item_embedding.py.
 * This value must be consistent with the CSV produced by the Python pipeline.
 */
constexpr int DIM = 64;

/**
 * @brief A single rated interaction between a user and an item.
 *
 * Loaded from train.csv. The timestamp field is intentionally omitted —
 * row insertion order already encodes chronology, and the MVP does not
 * apply recency weighting.
 */
struct Interaction {
    string asin;
    float rating;
};

/**
 * @brief To make the code more readable and short
 */
typedef array<float, DIM> embedding_t;

/**
 * @brief Loads item embedding vectors and the bidirectional ASIN ↔ row index.
 *
 * Reads two files:
 *   1. @p idx_path  — lightweight index (asin, row_idx). Loaded first to
 *                     pre-size the embedding matrix and avoid reallocation.
 *   2. @p emb_path  — dense float matrix (asin, d0 … d{DIM-1}).
 *
 * Both lookup directions are populated so that:
 *   - The KD-tree can resolve a row number back to a human-readable ASIN.
 *   - User history (stored as ASINs) can be mapped to row indices at
 *     inference time without a linear scan.
 *
 * @param emb_path      Path to item_embeddings.csv
 * @param idx_path      Path to item_embedding_index.csv
 * @param embeddings    Output: row-indexed embedding matrix
 * @param asin_to_idx   Output: ASIN → row number
 * @param idx_to_asin   Output: row number → ASIN
 *
 * @throws std::runtime_error if either file cannot be opened.
 * @throws std::out_of_range  if an ASIN in emb_path is absent from idx_path.
 */
void load_embeddings(
    const string &emb_path,
    const string &idx_path,
    vector<embedding_t>& embeddings,
    unordered_map<string, int>& asin_to_idx,
    unordered_map<int, string>& idx_to_asin
);


/**
 * @brief Loads user interaction histories from the training split.
 *
 * Each row in train.csv contributes one Interaction to the corresponding
 * user's history vector. Vectors grow via push_back, preserving the
 * chronological order guaranteed by the Python preprocessing step.
 *
 * Users with fewer than 5 interactions are excluded upstream (in
 * 03_filter_data.py) and will not appear here.
 *
 * @param path          Path to train.csv
 * @param user_history  Output: user_id → ordered list of rated interactions
 *
 * @throws std::runtime_error if the file cannot be opened.
 */
void load_train(
    const string& path,
    unordered_map<string, vector<Interaction>>& user_history
);

/**
 * @brief Loads the leave-one-out ground truth for evaluation.
 *
 * Each entry is the single held-out item for a user — the last interaction
 * chronologically, withheld from training. Used exclusively by the evaluator
 * to compute Recall@K and NDCG@K. Never fed into the recommendation pipeline.
 *
 * 16 users whose held-out item had no training signal were removed upstream.
 * See scripts/README.md for details.
 *
 * @param path          Path to test.csv
 * @param ground_truth  Output: user_id → held-out ASIN
 *
 * @throws std::runtime_error if the file cannot be opened.
 */
void load_test(
    const string& path,
    unordered_map<string, string>& ground_truth
);