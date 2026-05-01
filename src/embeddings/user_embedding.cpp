/**
 * @file    user_embedding.cpp
 * @brief   Implementation of user embedding computation.
 *
 * Weighted average approach:
 *   - Weight per item = rating / 5.0, mapping [1,5] → [0.2, 1.0]
 *   - A 1-star item still contributes (weight 0.2) but less than 5-star (1.0)
 *   - Final vector is L2-normalised to unit length
 *
 * Why weighted average over simple average:
 *   A user who rated item A 5 stars and item B 1 star has a clear preference.
 *   Simple average treats both equally — weighted average pulls the user
 *   vector toward item A's direction in embedding space.
 */

#include "user_embedding.h"
#include <cmath>
#include <numeric>
#include <iostream>

using std::vector,
    std::array,
    std::string,
    std::unordered_map,
    std::cout,
    std::sqrt,
    std::abs,
    std::cerr;

/**
 * @brief L2-normalises a vector in place.
 *
 * If the vector is all zeros (user had no valid items), it is left unchanged.
 * The caller checks for zero vectors before using the result.
 *
 * @param user_vector  Vector to normalise. (can be any vector)
 */
static void l2_normalize(embedding_t& user_vector) {
    float norm = 0.0f;
    for (float v: user_vector) norm += v * v;
    norm = sqrt(norm);

    if (norm < 1e-9f) return; // zero vector will give undefined or inf norm
    for (float& v : user_vector) v /= norm;
}

static bool is_zero_vector(const embedding_t& vec) {
    for (float v : vec)
        if (abs(v) > 1e-6f) return false;
    return true;
}

/**
 * @brief Checks whether a vector is all zeros.
 *
 * Used to detect users with no valid embedded items.
 */
embedding_t compute_user_embedding(
    const vector<Interaction>& history,
    const vector<embedding_t>& embeddings,
    const unordered_map<string, int>& asin_to_idx
) {
    embedding_t user_vector{0};
    float total_weight = 0.0f;
    int skipped = 0;

    for (const auto& interaction: history) {
        auto it = asin_to_idx.find(interaction.asin);
        if (it == asin_to_idx.end()) {
            ++skipped;
            continue;
        }

        // Weight = rating / 5.0 → maps [1, 5] to [0.2, 1.0]
        // A 1-star review still contributes — it places the user
        // near that item's region, but with less conviction than 5 stars.
        // float weight = interaction.rating / 5.0f;
        float weight = interaction.rating - 3.0f;  
        int   row    = it->second;

        for (int d = 0; d < DIM; d++)
            user_vector[d] += weight * embeddings[row][d];

        total_weight += weight;
    }

    if (skipped > 0)
        std::cerr << "[user_embedding] skipped " << skipped
                  << " items not in embedding index\n";

    // Normalise by total weight to get weighted average
    // (not just weighted sum — scale matters for normalisation)
    float abs_weight = 0.0f;
    for (const auto& inter : history) {
        auto it = asin_to_idx.find(inter.asin);
        if (it == asin_to_idx.end()) continue;
        abs_weight += std::abs(inter.rating - 3.0f);
    }

    if (abs_weight > 1e-9f)
        for (float& v : user_vector) v /= abs_weight;

    l2_normalize(user_vector);
    return user_vector;
}

unordered_map<string, embedding_t> compute_all_user_embeddings(
    const unordered_map<string,vector<Interaction>>& user_history,
    const vector<embedding_t>& embeddings, // item embeddings
    const unordered_map<string, int>& asin_to_idx
) {
    unordered_map<string, embedding_t> user_embeddings;
    user_embeddings.reserve(user_history.size()); // what reserve does?

    int skipped_users = 0;

    for (const auto& [user_id, history]: user_history) {
        embedding_t vec = compute_user_embedding(history, embeddings, asin_to_idx);

        // Skip users whose history had no valid embedded items
        if (is_zero_vector(vec)) {
            ++skipped_users;
            continue;
        }
        user_embeddings[user_id] = vec;
    }

    if (skipped_users > 0)
        std::cerr << "[user_embedding] skipped " << skipped_users
                  << " users with no valid embeddings\n";

    std::cout << "[user_embedding] computed " << user_embeddings.size()
              << " user embeddings\n";

    return user_embeddings;
}