/**
 * @file    pipeline.h
 * @brief   End-to-end recommendation pipeline.
 *
 * Wires together all components into a single inference pass:
 *
 *   train.csv + embeddings
 *       │
 *       ▼
 *   User embedding  (weighted average of item vectors)
 *       │
 *       ▼
 *   KD-tree query   (top-N candidates from embedding space)
 *       │
 *       ▼
 *   Bloom filter    (exclude already-seen items)
 *       │
 *       ▼
 *   Ranker          (cosine sort or Adaptive MMR)
 *       │
 *       ▼
 *   Top-K output    (ranked list of item row indices)
 *
 * Two modes:
 *   single_user() — recommend for one user by ID
 *   all_users()   — run inference for every user in test set,
 *                   save results to CSV for evaluation
 *
 * @author  anshulbadhani
 * @date    30/04/2026
 */

#pragma once

#include "data_loader.h"
#include "kdtree.h"
#include "ranker.h"
#include "bloom_filter.h"
#include "user_embedding.h"

#include <string>
#include <vector>
#include <unordered_map>

using std::vector,
    std::string,
    std::unordered_map;

/**
 * @brief Full pipeline configuration.
 */
struct PipelineConfig {
    int candidate_pool = 500;    ///< How many candidates to fetch from KD-tree
                                 ///< before Bloom filtering. Larger = better
                                 ///< recall, slower query. Rule of thumb: 5-10x K.
    int top_k = 10;              ///< Final number of recommendations to return.
    MMRConfig mmr = {};          ///< Adaptive MMR hyperparameters.
    float bloom_fp_rate = 0.01f; ///< Bloom filter false positive rate.
};

/**
 * @brief Recommendations for a single user.
 */
struct UserRecommendations {
    string user_id;
    vector<string> cosine_asins; ///< Top-K by cosine sort
    vector<string> mmr_asins;    ///< Top-K by Adaptive MMR
};

/**
 * @brief End-to-end recommendation pipeline.
 *
 * Built once, queried many times. All heavy structures (KD-tree, embeddings)
 * are constructed at initialisation and reused across queries.
 *
 * Typical usage:
 * @code
 *   Pipeline pipeline(embeddings, asin_to_idx, idx_to_asin,
 *                     user_history, config);
 *
 *   // Single user
 *   auto recs = pipeline.single_user("AHZZYDN7XZXJRETMPWW4RRD4PS2Q");
 *
 *   // All test users → saved to CSV
 *   pipeline.all_users(ground_truth, "cosine_results.csv", "mmr_results.csv");
 * @endcode
 */
class Pipeline {
public:
    /**
     * @brief Constructs and initialises the full pipeline.
     *
     * Builds the KD-tree and Ranker at construction time — O(n log² n).
     * Per-query work is user embedding + KD-tree query + Bloom filter + ranking.
     *
     * @param embeddings    Item embedding matrix.
     * @param asin_to_idx   ASIN → row index.
     * @param idx_to_asin   Row index → ASIN.
     * @param user_history  Training interactions per user.
     * @param config        Pipeline hyperparameters.
     */
    Pipeline(
        const vector<embedding_t> &embeddings,
        const unordered_map<string, int> &asin_to_idx,
        const unordered_map<int, string> &idx_to_asin,
        const unordered_map<string, vector<Interaction>> &user_history,
        const PipelineConfig &config = {});

    /**
     * @brief Generates recommendations for a single user.
     *
     * @param user_id  User to generate recommendations for.
     * @return         Ranked recommendations (cosine + MMR).
     *                 Returns empty lists if user not found in train.
     */
    UserRecommendations single_user(const string &user_id) const;

    /**
     * @brief Runs inference for all users in ground_truth, saves to CSV.
     *
     * Output format (one row per user):
     *   user_id, rec_1, rec_2, ..., rec_K
     *
     * Saves two files — one for cosine sort, one for Adaptive MMR.
     * These are consumed by the Python evaluation script.
     *
     * @param ground_truth      Test set — user_id → held-out ASIN.
     * @param cosine_out_path   Output path for cosine sort results.
     * @param mmr_out_path      Output path for Adaptive MMR results.
     */
    void all_users(
        const unordered_map<string, string> &ground_truth,
        const string &cosine_out_path,
        const string &mmr_out_path) const;

private:
    // ── Owned components ───────────────────────────────────────────────────
    const vector<embedding_t> &embeddings_;
    const unordered_map<string, int> &asin_to_idx_;
    const unordered_map<int, string> &idx_to_asin_;
    const unordered_map<string, vector<Interaction>> &user_history_;
    PipelineConfig config_;
    KDTree tree_;
    Ranker ranker_;

    /**
     * @brief Core inference for one user — shared by single_user and all_users.
     *
     * Steps:
     *   1. Compute user embedding (weighted average of item vectors)
     *   2. Query KD-tree for candidate_pool nearest items
     *   3. Build Bloom filter from user's train history
     *   4. Filter candidates through Bloom filter
     *   5. Rank with cosine sort and Adaptive MMR
     *   6. Convert row indices → ASINs
     *
     * @param user_id  User to run inference for.
     * @return         Recommendations or empty if user not in train.
     */
    UserRecommendations infer(const string &user_id) const;

    /**
     * @brief Converts a vector of RankedItems to ASIN strings.
     *
     * @param items  Ranked items with row indices.
     * @param k      Maximum number of ASINs to return.
     * @return       Ordered list of ASIN strings.
     */
    vector<string> to_asins(
        const vector<RankedItem> &items,
        int k) const;
};