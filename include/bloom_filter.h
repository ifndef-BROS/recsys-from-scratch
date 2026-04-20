/**
 * @file    bloom_filter.h
 * @brief   Space-efficient probabilistic set for seen-item filtering.
 *
 * Used during recommendation to exclude items a user has already interacted
 * with. Built per-user from their training history, queried against KD-tree
 * candidates before ranking.
 *
 * Properties:
 *   - False negatives : IMPOSSIBLE — a seen item is always identified as seen.
 *   - False positives : possible at rate p (default 1%). An unseen item may
 *                       rarely be excluded. Acceptable for recommendation.
 *
 * Parameters are computed automatically from history length and target p:
 *   m = -( n * ln(p) ) / ln(2)²    (bit array size)
 *   k =  ( m / n )    * ln(2)      (number of hash functions)
 *
 * For p=0.01, k=7 regardless of n. At n=12 (99th percentile user),
 * m=116 bits — the entire filter fits in 2 × uint64_t words.
 *
 * Hash strategy: double hashing — generates k independent positions from
 * two base hashes (FNV-1a + djb2) with no external dependencies.
 *   position_i = (h1 + i * h2) % m
 *
 * @author  anshulbadhani
 * @date    2026
 */

#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <cstdint>

using std::vector,
    std::string;

/**
 * @brief Optimal Bloom filter parameters for a given workload.
 * @see   bloom_params()
 */
struct BloomParams {
    int m; ///< Total number of bits in the filter
    int k; ///< Number of hash functions (independent bit positions per item)
};

/**
 * @brief Per-user Bloom filter for O(k) seen-item lookup.
 *
 * Sized dynamically to each user's history length — a user with 2 interactions
 * gets a 19-bit filter, a user with 371 gets a 3554-bit filter.
 *
 * Typical usage:
 * @code
 *   BloomFilter bf(user_history.size(), 0.01);
 *   for (auto& interaction : user_history)
 *       bf.insert(interaction.asin);
 *
 *   if (!bf.probably_seen(candidate_asin))
 *       shortlist.push_back(candidate_asin);
 * @endcode
 */
class BloomFilter {
public:
    /**
     * @brief Constructs a Bloom filter sized for n items at false positive rate p.
     *
     * @param n  Number of items to insert (user history length).
     * @param p  Target false positive rate in (0, 1). Default: 0.01 (1%).
     *
     * @throws std::invalid_argument if n <= 0 or p outside (0, 1).
     */
    BloomFilter(int n, double p=0.01);

    /**
     * @brief Inserts an item into the filter.
     *
     * Sets k bit positions derived from the item's ASIN. Insertion is
     * idempotent — inserting the same item twice has no additional effect.
     *
     * @param asin  Amazon item identifier to mark as seen.
     */
    void insert(const string& asin);

    /**
     * @brief Queries whether an item was probably inserted.
     *
     * @param asin  Item to check.
     * @return true  if the item was probably inserted (may be a false positive).
     * @return false if the item was definitely NOT inserted.
     */
    bool probably_seen(const string& asin) const;

    /**
     * @brief Returns the parameters computed for this filter instance.
     *
     * Useful for logging and report validation.
     */
    BloomParams params() const { return params_; }

private:
    BloomParams      params_;  ///< Computed m and k for this instance
    vector<uint64_t> bits_;    ///< Backing bit array (ceil(m/64) words)

    /**
     * @brief FNV-1a hash — first base hash (h1).
     *
     * Fast, good avalanche behaviour on short strings like ASINs.
     * See: http://www.isthe.com/chongo/tech/comp/fnv/
     */
    uint64_t fnv1a(const string& s) const;

    /**
     * @brief djb2 hash — second base hash (h2).
     *
     * Used with FNV-1a in double hashing to simulate k independent hashes:
     *   position_i = (h1 + i * h2) % m
     *
     * Keeps k hash functions without k separate implementations.
     */
    uint64_t djb2(const string& s) const;

    /**
     * @brief Sets bit at position pos in the backing array.
     * @param pos  Bit index in [0, m).
     */
    void set_bit(int pos);

    /**
     * @brief Tests bit at position pos in the backing array.
     * @param pos  Bit index in [0, m).
     * @return true if the bit is set.
     */
    bool get_bit(int pos) const;
};

/**
 * @brief Computes optimal Bloom filter parameters for n items at rate p.
 *
 * Derivation:
 *   m = ceil( -( n * ln(p) ) / ln(2)² )
 *   k = round( ( m / n ) * ln(2) )
 *
 * Example outputs (p = 0.01):
 *   n=2   → m=20,   k=7
 *   n=12  → m=116,  k=7
 *   n=371 → m=3554, k=7
 *
 * Note: k=7 is constant for p=0.01 regardless of n.
 *
 * @param n  Number of items to insert. Must be > 0.
 * @param p  False positive rate in (0, 1).
 * @return   BloomParams with computed m and k.
 */
BloomParams bloom_params(int n, double p = 0.01);