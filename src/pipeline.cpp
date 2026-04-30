/**
 * @file    pipeline.cpp
 * @brief   Implementation of the end-to-end recommendation pipeline.
 *
 * Per-query complexity:
 *   User embedding : O(h · DIM)     h = history length
 *   KD-tree query  : O(log n · DIM) best, O(n · DIM) worst (high dims)
 *   Bloom filter   : O(h · k)       h = history, k = hash functions
 *   Cosine sort    : O(c · DIM + c log c)  c = candidate pool size
 *   Adaptive MMR   : O(c · K · DIM)        K = top-k
 *
 * Total per user: dominated by KD-tree query at high dimensionality.
 * Applying PCA to 64 dims before building significantly reduces this.
 */

#include "pipeline.h"

#include <fstream>
#include <iostream>
#include <chrono>

using namespace std;

Pipeline::Pipeline(
    const vector<embedding_t>&                                  embeddings,
    const unordered_map<string, int>&                      asin_to_idx,
    const unordered_map<int, string>&                      idx_to_asin,
    const unordered_map<string, vector<Interaction>>& user_history,
    const PipelineConfig&                                            config
)
    : embeddings_   (embeddings)
    , asin_to_idx_  (asin_to_idx)
    , idx_to_asin_  (idx_to_asin)
    , user_history_ (user_history)
    , config_       (config)
    , tree_         (embeddings)       // build KD-tree once at startup
    , ranker_       (embeddings)       // initialise ranker
{
    cout << "[pipeline] initialised"
              << "  candidate_pool=" << config_.candidate_pool
              << "  top_k="          << config_.top_k
              << "  mmr_lambda="     << config_.mmr.lambda_max
              << "  mmr_decay="      << config_.mmr.decay_rate
              << "\n";
}

UserRecommendations Pipeline::single_user(const string& user_id) const {
    auto recs = infer(user_id);

    // Pretty print for interactive use
    cout << "\n[pipeline] recommendations for " << user_id << "\n";

    cout << "  Cosine sort:\n";
    for (int i = 0; i < static_cast<int>(recs.cosine_asins.size()); i++)
        cout << "    " << (i+1) << ". " << recs.cosine_asins[i] << "\n";

    cout << "  Adaptive MMR:\n";
    for (int i = 0; i < static_cast<int>(recs.mmr_asins.size()); i++)
        cout << "    " << (i+1) << ". " << recs.mmr_asins[i] << "\n";

    return recs;
}

void Pipeline::all_users(
    const unordered_map<string, string>& ground_truth,
    const string& cosine_out_path,
    const string& mmr_out_path
) const {
    ofstream cosine_f(cosine_out_path);
    ofstream mmr_f(mmr_out_path);

    if (!cosine_f.is_open())
        throw runtime_error("Cannot open: " + cosine_out_path);
    if (!mmr_f.is_open())
        throw runtime_error("Cannot open: " + mmr_out_path);

    // Header: user_id, rec_1, rec_2, ..., rec_K
    cosine_f << "user_id";
    mmr_f    << "user_id";
    for (int i = 1; i <= config_.top_k; i++) {
        cosine_f << ",rec_" << i;
        mmr_f    << ",rec_" << i;
    }
    cosine_f << "\n";
    mmr_f    << "\n";

    int processed  = 0;
    int skipped    = 0;
    int total      = static_cast<int>(ground_truth.size());

    auto t_start = chrono::steady_clock::now();

    for (const auto& [user_id, gt_asin] : ground_truth) {
        auto recs = infer(user_id);

        if (recs.cosine_asins.empty()) {
            ++skipped;
            continue;
        }

        // Write cosine results
        cosine_f << user_id;
        for (auto& asin : recs.cosine_asins) cosine_f << "," << asin;
        cosine_f << "\n";

        // Write MMR results
        mmr_f << user_id;
        for (auto& asin : recs.mmr_asins) mmr_f << "," << asin;
        mmr_f << "\n";

        ++processed;

        // Progress every 10K users
        if (processed % 10000 == 0) {
            auto elapsed = chrono::steady_clock::now() - t_start;
            float secs   = chrono::duration<float>(elapsed).count();
            cout << "[pipeline] " << processed << " / " << total
                      << "  (" << secs << "s  "
                      << static_cast<float>(processed) / secs << " users/s)\n";
        }
    }

    auto elapsed = chrono::steady_clock::now() - t_start;
    float secs   = chrono::duration<float>(elapsed).count();

    cout << "[pipeline] done."
              << "  processed=" << processed
              << "  skipped="   << skipped
              << "  time="      << secs << "s"
              << "  avg="       << secs / processed * 1000.0f << "ms/user\n";
}

UserRecommendations Pipeline::infer(const string& user_id) const {
    UserRecommendations recs;
    recs.user_id = user_id;

    // 1. Look up user history
    auto it = user_history_.find(user_id);
    if (it == user_history_.end()) {
        cerr << "[pipeline] user not found in train: " << user_id << "\n";
        return recs;
    }
    const auto& history = it->second;

    // 2. Compute user embedding
    embedding_t user_vec = compute_user_embedding(history, embeddings_, asin_to_idx_);

    // Skip users whose history had no valid embedded items
    bool is_zero = true;
    for (float v : user_vec) if (abs(v) > 1e-6f) { is_zero = false; break; }
    if (is_zero) {
        cerr << "[pipeline] zero user embedding: " << user_id << "\n";
        return recs;
    }

    // 3. KD-tree query
    // Fetch more candidates than needed — Bloom filter will reduce the pool
    auto candidates = tree_.query(user_vec, config_.candidate_pool);

    // 4. Bloom filter — exclude seen items
    BloomFilter seen(static_cast<int>(history.size()), config_.bloom_fp_rate);
    for (const auto& inter : history)
        seen.insert(inter.asin);

    vector<KNNResult> unseen;
    unseen.reserve(candidates.size());
    for (const auto& c : candidates) {
        const string& asin = idx_to_asin_.at(c.row);
        if (!seen.probably_seen(asin))
            unseen.push_back(c);
    }

    if (unseen.empty()) {
        cerr << "[pipeline] all candidates filtered for user: " << user_id << "\n";
        return recs;
    }

    // 5. Rank
    auto cosine_ranked = ranker_.cosine_sort(user_vec, unseen, config_.top_k);
    auto mmr_ranked    = ranker_.adaptive_mmr(user_vec, unseen, config_.top_k, config_.mmr);

    // 6. Convert row indices → ASINs
    recs.cosine_asins = to_asins(cosine_ranked, config_.top_k);
    recs.mmr_asins    = to_asins(mmr_ranked,    config_.top_k);

    return recs;
}

vector<string> Pipeline::to_asins(
    const vector<RankedItem>& items,
    int                            k
) const {
    vector<string> asins;
    asins.reserve(min(k, static_cast<int>(items.size())));
    for (int i = 0; i < k && i < static_cast<int>(items.size()); i++)
        asins.push_back(idx_to_asin_.at(items[i].row));
    return asins;
}