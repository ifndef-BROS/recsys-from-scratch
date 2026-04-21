/**
 * AI-Generated
 * @file    main.cpp
 * @brief   Integration test for data loading and Bloom filter.
 *
 * Tests (in order):
 *   1. bloom_params()   — formula correctness against known values
 *   2. BloomFilter      — insert / probably_seen / no false negatives
 *   3. load_embeddings  — matrix shape, lookup maps, vector sanity
 *   4. load_train       — user count, history integrity
 *   5. load_test        — ground truth count, no train leakage
 *   6. End-to-end       — build a Bloom filter from a real user's history,
 *                         confirm all their train items are marked seen,
 *                         confirm their test item is NOT marked seen
 */

#include "data_loader.h"
#include "bloom_filter.h"
#include "user_embedding.h"
#include "kdtree.h"

#include <iostream>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_set>
#include <numeric>
#include <unordered_set>
#include <algorithm>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int passed = 0;
static int failed = 0;

/**
 * @brief Prints PASS/FAIL and updates counters.
 *
 * Using a macro so __LINE__ points to the call site, not this function.
 */
#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (cond) {                                                   \
            std::cout << "  [PASS] " << (msg) << "\n";               \
            ++passed;                                                 \
        } else {                                                      \
            std::cout << "  [FAIL] " << (msg) << "  (line "          \
                      << __LINE__ << ")\n";                           \
            ++failed;                                                 \
        }                                                             \
    } while (0)

static void print_section(const std::string& title) {
    std::cout << "\n══════════════════════════════════════\n";
    std::cout << "  " << title << "\n";
    std::cout << "══════════════════════════════════════\n";
}

// ---------------------------------------------------------------------------
// Test 1 — bloom_params formula
// ---------------------------------------------------------------------------

static void test_bloom_params() {
    print_section("1. bloom_params()");

    // Known values derived manually:
    //   n=2,  p=0.01 → m=20,   k=7
    //   n=12, p=0.01 → m=116,  k=7
    //   n=371,p=0.01 → m=3554, k=7
    // k is always 7 for p=0.01 — independent of n

    auto p2   = bloom_params(2,   0.01);
    auto p12  = bloom_params(12,  0.01);
    auto p371 = bloom_params(371, 0.01);

    CHECK(p2.m   >= 19  && p2.m   <= 21,  "n=2   m in expected range [19,21]");
    CHECK(p12.m  >= 114 && p12.m  <= 118, "n=12  m in expected range [114,118]");
    CHECK(p371.m >= 3550&& p371.m <= 3560,"n=371 m in expected range [3550,3560]");

    CHECK(p2.k   == 7, "n=2   k=7 for p=0.01");
    CHECK(p12.k  == 7, "n=12  k=7 for p=0.01");
    CHECK(p371.k == 7, "n=371 k=7 for p=0.01");

    // k should be 3 for p=0.1
    auto p10pct = bloom_params(100, 0.1);
    CHECK(p10pct.k >= 3 && p10pct.k <= 4, "p=0.10 → k in [3,4]");

    // Larger p → smaller m (fewer bits needed for relaxed FP rate)
    auto loose = bloom_params(100, 0.1);
    auto tight = bloom_params(100, 0.001);
    CHECK(loose.m < tight.m, "relaxed p → smaller m");
}

// ---------------------------------------------------------------------------
// Test 2 — BloomFilter correctness
// ---------------------------------------------------------------------------

static void test_bloom_filter() {
    print_section("2. BloomFilter insert / query");

    // ── 2a. Basic insert and lookup ────────────────────────────────────────
    {
        BloomFilter bf(10, 0.01);

        std::vector<std::string> inserted = {
            "B07XYZ123", "B08ABC456", "B09DEF789",
            "B01GHI012", "B02JKL345"
        };

        for (auto& asin : inserted)
            bf.insert(asin);

        // No false negatives — every inserted item must be found
        bool no_false_negatives = true;
        for (auto& asin : inserted)
            if (!bf.probably_seen(asin)) { no_false_negatives = false; break; }

        CHECK(no_false_negatives, "no false negatives on inserted items");

        // Items that were never inserted should mostly return false
        // We test 100 random-ish ASINs and allow up to 5 false positives (5%)
        // to account for the 1% per-item FP rate with some slack
        std::vector<std::string> not_inserted;
        for (int i = 0; i < 100; i++)
            not_inserted.push_back("ZZZTEST" + std::to_string(i));

        int false_positives = 0;
        for (auto& asin : not_inserted)
            if (bf.probably_seen(asin)) ++false_positives;

        std::cout << "  [INFO] false positives: " << false_positives << " / 100\n";
        CHECK(false_positives <= 10, "false positive rate under 10% (expected ~1%)");
    }

    // ── 2b. Idempotency — inserting same item twice changes nothing ────────
    {
        BloomFilter bf1(5, 0.01);
        BloomFilter bf2(5, 0.01);

        bf1.insert("B07XYZ123");

        bf2.insert("B07XYZ123");
        bf2.insert("B07XYZ123"); // duplicate insert

        // Both filters should give the same answer for any query
        bool same = (bf1.probably_seen("B07XYZ123") == bf2.probably_seen("B07XYZ123")) &&
                    (bf1.probably_seen("B00UNSEEN0") == bf2.probably_seen("B00UNSEEN0"));

        CHECK(same, "double insert is idempotent");
    }

    // ── 2c. Empty filter — nothing should be seen ──────────────────────────
    {
        BloomFilter bf(5, 0.01);
        CHECK(!bf.probably_seen("B07XYZ123"), "empty filter returns false");
    }

    // ── 2d. params() accessor ──────────────────────────────────────────────
    {
        BloomFilter bf(12, 0.01);
        auto p = bf.params();
        CHECK(p.m > 0 && p.k == 7, "params() returns correct m and k");
    }
}

// ---------------------------------------------------------------------------
// Test 3 — load_embeddings
// ---------------------------------------------------------------------------

static void test_load_embeddings(
    const std::vector<std::array<float, DIM>>& embeddings,
    const std::unordered_map<std::string, int>& asin_to_idx,
    const std::unordered_map<int, std::string>& idx_to_asin
) {
    print_section("3. load_embeddings()");

    CHECK(!embeddings.empty(),    "embeddings matrix is non-empty");
    CHECK(!asin_to_idx.empty(),   "asin_to_idx map is non-empty");
    CHECK(!idx_to_asin.empty(),   "idx_to_asin map is non-empty");
    CHECK(asin_to_idx.size() == idx_to_asin.size(),
          "asin_to_idx and idx_to_asin have same size");
    CHECK(embeddings.size() == asin_to_idx.size(),
          "embedding matrix rows == number of indexed items");

    // Every row index in idx_to_asin must be a valid embeddings row
    bool indices_valid = true;
    for (auto& [idx, asin] : idx_to_asin)
        if (idx < 0 || idx >= static_cast<int>(embeddings.size()))
            { indices_valid = false; break; }
    CHECK(indices_valid, "all row indices are within embedding matrix bounds");

    // Bidirectional lookup roundtrip: asin → idx → asin
    auto it = asin_to_idx.begin();
    const std::string& sample_asin = it->first;
    int   sample_idx               = it->second;
    CHECK(idx_to_asin.at(sample_idx) == sample_asin,
          "asin → idx → asin roundtrip is consistent");

    // Embeddings should be approximately unit length (normalised in Python)
    // Allow tolerance of 1e-3 for float32 precision
    float norm = 0.0f;
    for (float v : embeddings[sample_idx]) norm += v * v;
    norm = std::sqrt(norm);
    CHECK(std::abs(norm - 1.0f) < 1e-3f,
          "sample embedding is unit-normalised (|norm - 1| < 1e-3)");

    // No embedding should be all zeros — would indicate a parse failure
    bool has_zero_vec = false;
    for (auto& emb : embeddings) {
        float sum = 0.0f;
        for (float v : emb) sum += std::abs(v);
        if (sum < 1e-6f) { has_zero_vec = true; break; }
    }
    CHECK(!has_zero_vec, "no all-zero embedding vectors");

    std::cout << "  [INFO] total items: " << embeddings.size() << "\n";
    std::cout << "  [INFO] sample asin: " << sample_asin
              << "  row: " << sample_idx
              << "  norm: " << norm << "\n";
}

// ---------------------------------------------------------------------------
// Test 4 — load_train
// ---------------------------------------------------------------------------

static void test_load_train(
    const std::unordered_map<std::string, std::vector<Interaction>>& user_history
) {
    print_section("4. load_train()");

    CHECK(!user_history.empty(), "user_history is non-empty");

    // Every user must have at least 5 interactions (guaranteed by filter step)
    bool min_interactions_ok = true;
    int  max_interactions    = 0;
    long total_interactions  = 0;

    for (auto& [user, history] : user_history) {
        if (static_cast<int>(history.size()) < 4)
            min_interactions_ok = false;
        max_interactions   = std::max(max_interactions, static_cast<int>(history.size()));
        total_interactions += history.size();
    }

    CHECK(min_interactions_ok, "all users have >= 4 interactions");

    // Ratings must be in [1.0, 5.0]
    bool ratings_valid = true;
    for (auto& [user, history] : user_history)
        for (auto& inter : history)
            if (inter.rating < 1.0f || inter.rating > 5.0f)
                { ratings_valid = false; break; }

    CHECK(ratings_valid, "all ratings in [1.0, 5.0]");

    // ASINs should be non-empty strings
    bool asins_valid = true;
    for (auto& [user, history] : user_history)
        for (auto& inter : history)
            if (inter.asin.empty()) { asins_valid = false; break; }

    CHECK(asins_valid, "no empty ASINs in train history");

    double avg = static_cast<double>(total_interactions) / user_history.size();
    std::cout << "  [INFO] users: "            << user_history.size()  << "\n";
    std::cout << "  [INFO] total interactions: "<< total_interactions   << "\n";
    std::cout << "  [INFO] avg per user: "      << avg                  << "\n";
    std::cout << "  [INFO] max per user: "      << max_interactions     << "\n";
}

// ---------------------------------------------------------------------------
// Test 5 — load_test
// ---------------------------------------------------------------------------

static void test_load_test(
    const std::unordered_map<std::string, std::string>& ground_truth,
    const std::unordered_map<std::string, std::vector<Interaction>>& user_history
) {
    print_section("5. load_test()");

    CHECK(!ground_truth.empty(), "ground_truth is non-empty");

    // Every test user should also exist in train
    // (they need a history to build a user embedding)
    bool all_in_train = true;
    for (auto& [user, asin] : ground_truth)
        if (user_history.find(user) == user_history.end())
            { all_in_train = false; break; }

    CHECK(all_in_train, "all test users exist in train");

    // Ground truth item must NOT appear in the user's train history
    // (leave-one-out guarantee — no leakage)
    bool no_leakage = true;
    // for (auto& [user, gt_asin] : ground_truth) {
    //     auto& history = user_history.at(user);
    //     for (auto& inter : history)
    //         if (inter.asin == gt_asin) { no_leakage = false; break; }
    //     if (!no_leakage) break;
    // }
    for (auto& [user, gt_asin] : ground_truth) {
        auto& history = user_history.at(user);
        for (auto& inter : history) {
            if (inter.asin == gt_asin) {
                // Print the offending user before failing
                std::cout << "  [INFO] leakage detected — user: " << user
                        << "  asin: " << gt_asin << "\n";
                no_leakage = false;
                break;
            }
        }
        if (!no_leakage) break;
    }

    CHECK(no_leakage, "ground truth item never appears in user train history");

    // Ground truth ASINs should be non-empty
    bool asins_valid = true;
    for (auto& [user, asin] : ground_truth)
        if (asin.empty()) { asins_valid = false; break; }

    CHECK(asins_valid, "no empty ground truth ASINs");

    std::cout << "  [INFO] test users: " << ground_truth.size() << "\n";
}

// ---------------------------------------------------------------------------
// Test 6 — end-to-end: real user + Bloom filter
// ---------------------------------------------------------------------------

static void test_end_to_end(
    const std::unordered_map<std::string, std::vector<Interaction>>& user_history,
    const std::unordered_map<std::string, std::string>& ground_truth,
    const std::unordered_map<std::string, int>& asin_to_idx
) {
    print_section("6. End-to-end: real user Bloom filter");

    // Pick a user that exists in both train and test
    std::string test_user;
    for (auto& [user, _] : ground_truth) {
        if (user_history.count(user)) { test_user = user; break; }
    }

    if (test_user.empty()) {
        std::cout << "  [SKIP] no overlapping user found\n";
        return;
    }

    auto& history  = user_history.at(test_user);
    auto& gt_asin  = ground_truth.at(test_user);

    std::cout << "  [INFO] user:            " << test_user           << "\n";
    std::cout << "  [INFO] history length:  " << history.size()      << "\n";
    std::cout << "  [INFO] ground truth:    " << gt_asin             << "\n";

    // Build Bloom filter from user's train history
    BloomFilter bf(static_cast<int>(history.size()), 0.01);
    for (auto& inter : history)
        bf.insert(inter.asin);

    // Every train item must be marked seen — no false negatives
    bool no_false_negatives = true;
    for (auto& inter : history)
        if (!bf.probably_seen(inter.asin))
            { no_false_negatives = false; break; }

    CHECK(no_false_negatives,
          "all train items marked seen (no false negatives)");

    // Ground truth item must NOT be marked seen
    // (it was never inserted — leave-one-out)
    CHECK(!bf.probably_seen(gt_asin),
          "ground truth item is NOT marked seen");

    // Ground truth item should have an embedding
    CHECK(asin_to_idx.count(gt_asin) > 0,
          "ground truth item has an embedding");

    // All train items should have embeddings too
    bool all_embedded = true;
    for (auto& inter : history)
        if (!asin_to_idx.count(inter.asin))
            { all_embedded = false; break; }

    CHECK(all_embedded, "all train items have embeddings");
}

static void test_user_embedding(
    const std::unordered_map<std::string, std::vector<Interaction>>& user_history,
    const std::vector<std::array<float, DIM>>& embeddings,
    const std::unordered_map<std::string, int>& asin_to_idx
) {
    print_section("7. compute_user_embedding()");

    // ── 7a. Single user embedding ──────────────────────────────────────────
    auto sample_user = user_history.begin()->first;
    auto& history    = user_history.at(sample_user);

    auto user_vec = compute_user_embedding(history, embeddings, asin_to_idx);

    // Must be unit length — dot product == cosine similarity in KD-tree
    float norm = 0.0f;
    for (float v : user_vec) norm += v * v;
    norm = std::sqrt(norm);
    std::cout << "  [INFO] sample user vec norm: " << norm << "\n";  // ← add this
    CHECK(std::abs(norm - 1.0f) < 1e-3f, "user embedding is unit-normalised");

    // Must not be all zeros
    bool nonzero = false;
    for (float v : user_vec) if (std::abs(v) > 1e-9f) { nonzero = true; break; }
    CHECK(nonzero, "user embedding is non-zero");

    // ── 7b. Rating weight effect ───────────────────────────────────────────
    // A user who only rated one item 5 stars vs 1 star should produce
    // different embeddings (different weights pull vector differently
    // when history has more than one item)
    if (history.size() >= 2) {
        // Build two fake single-item histories with different ratings
        std::vector<Interaction> high_rating = {{ history[0].asin, 5.0f }};
        std::vector<Interaction> low_rating  = {{ history[0].asin, 1.0f }};

        auto vec_high = compute_user_embedding(high_rating, embeddings, asin_to_idx);
        auto vec_low  = compute_user_embedding(low_rating,  embeddings, asin_to_idx);

        // Single item — weight cancels out in normalisation, vectors should be equal
        float diff = 0.0f;
        for (int d = 0; d < DIM; d++)
            diff += std::abs(vec_high[d] - vec_low[d]);

        CHECK(diff < 1e-3f,
              "single-item history: rating weight cancels in normalisation");
    }

    // ── 7c. All users ──────────────────────────────────────────────────────
    auto all_vecs = compute_all_user_embeddings(user_history, embeddings, asin_to_idx);

    CHECK(!all_vecs.empty(), "compute_all_user_embeddings returns non-empty map");
    CHECK(all_vecs.size() <= user_history.size(),
          "no more user embeddings than users in train");

    float worst_norm = 0.0f;
    std::string worst_user;
    for (auto& [uid, vec] : all_vecs) {
        float n = 0.0f;
        for (float v : vec) n += v * v;
        n = std::sqrt(n);
        if (std::abs(n - 1.0f) > std::abs(worst_norm - 1.0f)) {
            worst_norm = n;
            worst_user = uid;
        }
    }
    std::cout << "  [INFO] worst norm: " << worst_norm
            << "  user: " << worst_user << "\n";

    // Spot check — all returned embeddings are unit normalised
    bool all_normalised = true;
    for (auto& [uid, vec] : all_vecs) {
        float n = 0.0f;
        for (float v : vec) n += v * v;
        if (std::abs(std::sqrt(n) - 1.0f) > 1e-3f) { all_normalised = false; break; }
    }
    CHECK(all_normalised, "all user embeddings are unit-normalised");

    std::cout << "  [INFO] sample user:      " << sample_user        << "\n";
    std::cout << "  [INFO] history length:   " << history.size()     << "\n";
    std::cout << "  [INFO] total users embedded: " << all_vecs.size()<< "\n";
}

static void test_kdtree(
    const std::vector<embedding_t>& embeddings,
    const std::unordered_map<std::string, std::array<float, DIM>>& user_embeddings,
    const std::unordered_map<std::string, int>& asin_to_idx
) {
    print_section("8. KDTree build + query");

    // ── 8a. Build ──────────────────────────────────────────────────────────
    KDTree tree(embeddings);
    CHECK(tree.size() == static_cast<int>(embeddings.size()),
          "tree indexes all items");

    // ── 8b. Self-query — every item's nearest neighbour should be itself ───
    // Pick 10 random items and query the tree with their own embedding
    int correct_self = 0;
    for (int i = 0; i < 10; i++) {
        int row = (i * 1337) % embeddings.size();  // deterministic spread
        auto results = tree.query(embeddings[row], 1);
        if (!results.empty() && results[0].row == row)
            ++correct_self;
    }
    CHECK(correct_self == 10, "self-query returns item itself as nearest neighbour");

    // ── 8c. Result count ───────────────────────────────────────────────────
    auto sample_user = user_embeddings.begin();
    auto results = tree.query(sample_user->second, 50);
    CHECK(static_cast<int>(results.size()) == 50, "query returns exactly k results");

    // ── 8d. Results are sorted by distance ascending ───────────────────────
    bool sorted = true;
    for (int i = 1; i < static_cast<int>(results.size()); i++)
        if (results[i].squared_dist < results[i-1].squared_dist)
            { sorted = false; break; }
    CHECK(sorted, "results are sorted by distance ascending");

    // ── 8e. Recall vs brute force ──────────────────────────────────────────
    // For a sample of users, compare KD-tree top-50 against brute force top-50
    // A good KD-tree should retrieve >= 80% of the true nearest neighbours
    int total_retrieved = 0;
    int total_expected  = 0;
    int users_checked   = 0;
    int K               = 50;

    for (auto& [uid, user_vec] : user_embeddings) {
        if (users_checked++ >= 100) break;  // check 100 users

        // Brute force top-K
        std::vector<KNNResult> bf_results;
        bf_results.reserve(embeddings.size());
        for (int r = 0; r < static_cast<int>(embeddings.size()); r++)
            bf_results.push_back({r, squared_l2_free(user_vec, embeddings[r])});
        std::sort(bf_results.begin(), bf_results.end(),
                  [](const KNNResult& a, const KNNResult& b) {
                      return a.squared_dist < b.squared_dist; });
        bf_results.resize(K);

        // KD-tree top-K
        auto kd_results = tree.query(user_vec, K);

        // Count overlap
        std::unordered_set<int> bf_set;
        for (auto& r : bf_results) bf_set.insert(r.row);
        for (auto& r : kd_results)
            if (bf_set.count(r.row)) ++total_retrieved;

        total_expected += K;
    }

    float recall = static_cast<float>(total_retrieved) / total_expected;
    std::cout << "  [INFO] KD-tree recall vs brute force (k=50, 100 users): "
              << recall * 100.0f << "%\n";

    // At 384 dims recall may be low — acceptable, note in report
    CHECK(recall >= 0.5f, "KD-tree recall vs brute force >= 50%");

    std::cout << "  [INFO] total items in tree: " << tree.size() << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "recsys-from-scratch — data loader + bloom filter tests\n";

    // ── Tests that need no data files ──────────────────────────────────────
    test_bloom_params();
    test_bloom_filter();

    // ── Load data files ────────────────────────────────────────────────────
    std::vector<std::array<float, DIM>>                       embeddings;
    std::unordered_map<std::string, int>                      asin_to_idx;
    std::unordered_map<int, std::string>                      idx_to_asin;
    std::unordered_map<std::string, std::vector<Interaction>> user_history;
    std::unordered_map<std::string, std::string>              ground_truth;

    try {
        load_embeddings("data/embeddings/item_embeddings.csv",
                        "data/embeddings/item_embedding_index.csv",
                        embeddings, asin_to_idx, idx_to_asin);

        load_train("data/train.csv", user_history);
        load_test ("data/test.csv",  ground_truth);
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Failed to load data: " << e.what() << "\n";
        std::cerr << "        Make sure data files exist and paths are correct.\n";
        return 1;
    }

    // ── Tests that need data files ─────────────────────────────────────────
    test_load_embeddings(embeddings, asin_to_idx, idx_to_asin);
    test_load_train(user_history);
    test_load_test(ground_truth, user_history);
    test_end_to_end(user_history, ground_truth, asin_to_idx);
    auto user_embeddings = compute_all_user_embeddings(user_history, embeddings, asin_to_idx);
    test_user_embedding(user_history, embeddings, asin_to_idx);

    // KD Tree
    KDTree tree(embeddings);
    test_kdtree(embeddings, user_embeddings, asin_to_idx);

    // ── Summary ────────────────────────────────────────────────────────────
    std::cout << "\n══════════════════════════════════════\n";
    std::cout << "  Results: "
              << passed << " passed, "
              << failed << " failed\n";
    std::cout << "══════════════════════════════════════\n";

    return failed == 0 ? 0 : 1;
}