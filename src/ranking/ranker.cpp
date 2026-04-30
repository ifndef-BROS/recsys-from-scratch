/**
 * @file    ranker.cpp
 * @brief   Implementation of cosine sort and Adaptive MMR ranking.
 *
 * Cosine sort:
 *   Straightforward — compute dot product for each candidate, sort descending.
 *   Serves as the baseline ranker and as a special case of Adaptive MMR.
 *
 * Adaptive MMR:
 *   Greedy iterative selection. At each position:
 *     1. Compute λ(pos) = lambda_max * exp(-decay * pos)
 *     2. For each remaining candidate, compute MMR score:
 *          λ(pos) * sim(user, item) - (1-λ(pos)) * max_sim_to_selected
 *     3. Select the candidate with the highest MMR score
 *     4. Add to results, remove from candidates
 *
 *   The max_sim_to_selected term requires computing similarity between
 *   the candidate and every already-selected item. This is the O(n·k)
 *   cost — unavoidable for exact MMR.
 *
 *   Initialisation of max_sim_to_selected:
 *     Set to -1.0f (minimum possible cosine similarity) so the first
 *     selected item is chosen purely by relevance regardless of λ.
 *     This ensures position 0 is always the most relevant item.
 */

#include "ranker.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <functional>

using std::vector,
    std::sort,
    std::min,
    std::exp,
    std::numeric_limits;

Ranker::Ranker(const vector<embedding_t>& embeddings): embeddings_(embeddings) {}

vector<RankedItem> Ranker::cosine_sort(
    const embedding_t& user_vec,
    const vector<KNNResult>& candidates,
    int k
) const {
    // Compute cosine similarity for every candidate
    vector<RankedItem> scored;
    scored.reserve(candidates.size());

    for (const auto& c: candidates)
        scored.push_back({c.row, dot(user_vec, embeddings_[c.row])});

    // Sort descending by similarity — most relevant first
    sort(scored.begin(), scored.end(), [](const RankedItem& a, const RankedItem& b) {
        return a.score > b.score;
    });

    // Return top-k
    if (static_cast<int>(scored.size()) > k)
        scored.resize(k);

    return scored;
}

vector<RankedItem> Ranker::adaptive_mmr(
    const embedding_t& user_vec,
   const vector<KNNResult>& candidates,
   int k,
   const MMRConfig& config
) const {
    if (candidates.empty()) return {};

    int n = static_cast<int>(candidates.size());
    k = min(k, n);

    // Pre-compute user similarity for all candidates — reused every iteration
    // Avoids recomputing dot(user, item) O(k) times per candidate
    vector<float> user_sim(n);
    for (int i = 0; i < n; i++) {
        user_sim[i] = dot(user_vec, embeddings_[candidates[i].row]);
    }

    // Track which candidates are still available
    vector<bool> selected(n, false);

    // For each candidate, track its maximum similarity to any selected item
    // Initialised to -1.0 — minimum cosine similarity — so first selection
    // is driven purely by relevance regardless of lambda
    vector<float> max_sim_to_selected(n, -1.0f);
    
    vector<RankedItem> results;
    results.reserve(k);

    for (int pos = 0; pos < k; pos++) {
        // Compute position-dependent lambda
        // Decays exponentially — early positions exploit relevance,
        // later positions explore diversity
        // At pos=0: lambda = lambda_max (pure relevance)
        // At pos=K: lambda → 0          (pure diversity)
        float lambda = config.lambda_max * exp(-config.decay_rate * static_cast<float>(pos));

        // Find best candidate by MMR score 
        int   best_idx   = -1;
        float best_score = numeric_limits<float>::lowest();

        for (int i = 0; i < n; i++) {
            if (selected[i]) continue;

            // MMR score = relevance term - diversity penalty
            // relevance : how similar is this item to the user
            // penalty   : how similar is this item to already-selected items
            //             (high penalty → item is redundant)
            float mmr_score = lambda * user_sim[i]
                            - (1.0f - lambda) * max_sim_to_selected[i];

            if (mmr_score > best_score) {
                best_score = mmr_score;
                best_idx   = i;
            }
        }

        if (best_idx == -1) break;  // no candidates left

        // Select best candidate
        selected[best_idx] = true;
        results.push_back({candidates[best_idx].row, best_score});

        // Update max_sim_to_selected for remaining candidates
        // Only the newly selected item can increase any candidate's
        // max_sim_to_selected — no need to recompute from scratch
        const embedding_t& selected_emb = embeddings_[candidates[best_idx].row];
        for (int i = 0; i < n; i++) {
            if (selected[i]) continue;
            float sim = dot(embeddings_[candidates[i].row], selected_emb);
            if (sim > max_sim_to_selected[i])
                max_sim_to_selected[i] = sim;
        }
    }

    return results;
}

float Ranker::dot(const embedding_t& a, const embedding_t& b) const {
    float sum = 0.0f;
    for (int d = 0; d < DIM; d++)
        sum += a[d] * b[d];
    return sum;
}