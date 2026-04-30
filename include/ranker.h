/**
 * @file    ranker.h
 * @brief   Candidate ranking with cosine sort and Adaptive MMR.
 *
 * Two ranking strategies are implemented, both operating on the candidate
 * set returned by the KD-tree after Bloom filter exclusion:
 *
 * 1. Cosine Sort
 *    Ranks candidates by cosine similarity to the user vector.
 *    Since embeddings are unit-normalised, cosine similarity = dot product.
 *    O(n log n) — sort candidates by descending similarity.
 *
 * 2. Adaptive MMR (novel)
 *    Extends Maximal Marginal Relevance (Carbonell & Goldstein, 1998) with
 *    a position-dependent λ that decays exponentially as the list grows:
 *
 *      λ(pos) = λ_max · exp(-decay · pos)
 *
 *      score(i, pos) = λ(pos) · sim(user, item_i)
 *                    - (1 - λ(pos)) · max_j ∈ selected sim(item_i, item_j)
 *
 *    Early positions exploit relevance (λ ≈ 1.0).
 *    Later positions explore diversity (λ → 0.0).
 *
 *    Special cases (unified interface):
 *      decay=0, λ_max=1.0 → cosine sort
 *      decay=0, λ_max=0.5 → standard MMR
 *      decay=0.2, λ_max=1.0 → Adaptive MMR (default)
 *
 *    Complexity: O(n·k) — for each of k positions, scan n candidates.
 *
 * Comparative analysis:
 *   Cosine sort  : O(n log n), pure relevance, ignores inter-item similarity
 *   Standard MMR : O(n·k),    fixed tradeoff, same λ at every position
 *   Adaptive MMR : O(n·k),    position-aware, exploits then explores
 *
 * Reference:
 *   Carbonell, J. & Goldstein, J. (1998). The use of MMR, diversity-based
 *   reranking for reordering documents and producing summaries. SIGIR.
 *
 * @author  anshulbadhani
 * @date    30/04/2026
 */

#pragma once

#include "data_loader.h"
#include "kdtree.h"
#include <vector>
#include <string>

using std::vector;

/**
 * @brief A single ranked recommendation.
 */
struct RankedItem {
    int row;        ///< Row index in embedding matrix
    float score;    ///< Final ranking score (higher=better)
};

/**
 * @brief Configuration for the Adaptive MMR ranker.
 *
 * Setting decay_rate=0 and lambda_max=1.0 recovers cosine sort.
 * Setting decay_rate=0 and lambda_max=0.5 recovers standard MMR.
 */
struct MMRConfig {
    float lambda_max = 1.0f;    ///< Starting exploitation weight in [0, 1]
    float decay_rate = 0.2f;    ///< Exponential decay rate per position
};

/**
 * @brief Ranks KD-tree candidates for a given user embedding.
 *
 * Typical usage:
 * @code
 *   Ranker ranker(embeddings);
 *
 *   // Cosine sort
 *   auto results = ranker.cosine_sort(user_vec, candidates, K);
 *
 *   // Adaptive MMR
 *   MMRConfig cfg{.lambda_max=1.0f, .decay_rate=0.2f};
 *   auto results = ranker.adaptive_mmr(user_vec, candidates, K, cfg);
 * @endcode
 */
class Ranker {
public:
    /**
     * @brief Constructs the ranker with access to the item embedding matrix.
     *
     * @param embeddings  Row-indexed item embedding matrix.
     */
    explicit Ranker(const vector<embedding_t>& embeddings);

    /**
     * @brief Ranks candidates by cosine similarity to the user vector.
     * 
     * Since all embeddings are unit normalized, cosine similarity = dot product
     * no division required
     * 
     * Complexity: O(n·DIM + n log n)
     *   n·DIM  — compute similarity for each candidate
     *   n log n — sort by similarity
     * 
     * @param user_vec   Normalised user embedding.
     * @param candidates KD-tree results (after Bloom filter).
     * @param k          Number of items to return.
     * @return           Top-k items sorted by descending similarity.
     */
    vector<RankedItem> cosine_sort(
        const embedding_t& user_vec,
        const vector<KNNResult>& candidates,
        int k
    ) const;

    /**
     * @brief Ranks candidates using Adaptive MMR.
     *
     * Iteratively selects the next item that maximises:
     *
     *   score(i, pos) = λ(pos) · sim(user, item_i)
     *                 - (1 - λ(pos)) · max_{j ∈ selected} sim(item_i, item_j)
     *
     * where λ(pos) = lambda_max · exp(-decay_rate · pos)
     *
     * The decay causes the list to transition from exploitation (high λ,
     * pure relevance) to exploration (low λ, high diversity) as positions
     * increase. This is analogous to the exploration-exploitation tradeoff
     * in multi-armed bandits, applied statically to a ranked list.
     *
     * Complexity: O(n·k·DIM)
     *   For each of k positions: scan n candidates, compute similarity
     *   to user and to all already-selected items.
     *
     * @param user_vec   Normalised user embedding.
     * @param candidates KD-tree results (after Bloom filter).
     * @param k          Number of items to return.
     * @param config     MMR hyperparameters (lambda_max, decay_rate).
     * @return           Top-k items sorted by descending MMR score.
     */
    vector<RankedItem> adaptive_mmr(
        const embedding_t& user_vec,
        const vector<KNNResult>& candidates,
        int k,
        const MMRConfig& config = MMRConfig{}
    ) const;

private:
    /// Item embedding matrix — not owned by the ranker
    const std::vector<embedding_t>& embeddings_;

    /**
     * @brief Computes dot product between two unit-normalised vectors.
     *
     * For unit vectors: dot product == cosine similarity.
     * Range: [-1, 1]. Higher is more similar.
     *
     * @return Cosine similarity in [-1, 1].
     */
    float dot(const embedding_t& a, const embedding_t& b) const;
};