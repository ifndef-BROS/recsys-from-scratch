/**
 * @file    bloom_filter.cpp
 * @brief   Implementation of BloomFilter declared in bloom_filter.h
 *
 * Bit storage:
 *   The m-bit array is backed by ceil(m/64) uint64_t words.
 *   Bit i lives at: words[i/64], position (i % 64).
 *   Using 64-bit words keeps the array compact and makes set/get
 *   a single bitwise operation each.
 *
 * Double hashing:
 *   Rather than implementing k independent hash functions, we derive
 *   k positions from two base hashes (FNV-1a, djb2):
 *     pos_i = (h1 + i * h2) % m,  for i in [0, k)
 *   This is standard practice — Kirsch & Mitzenmacher (2006) proved
 *   this causes no asymptotic degradation in false positive rate.
 *   Reference: https://dl.acm.org/doi/10.1007/11841036_42
 */

#include "bloom_filter.h"
#include <stdexcept>
#include <iostream>

using std::cout,
    std::ceil,
    std::round,
    std::max,
    std::log,
    std::invalid_argument;

// non class function
BloomParams bloom_params(int n, double p) {
    if (n <= 0)      throw invalid_argument("bloom_params: n must be > 0");
    if (p <= 0 || p >= 1) throw invalid_argument("bloom_params: p must be in (0,1)");

    double ln2 = log(2.0);
    int m = static_cast<int>(ceil(-(n * log(p)) / (ln2 * ln2)));
    int k = static_cast<int>(round((static_cast<double>(m) / n) * ln2));

    // k must be at least 1 — degenerate case where m/n is very small
    k = max(k, 1);

    return {m, k};
}

BloomFilter::BloomFilter(int n, double p)
    : params_(bloom_params(n, p)),
      // ceil(m / 64) words — each word stores 64 bits, all initialised to 0
      // 64 because I am working on a 64 bit machine and it is faster than larger values and max value which can give the same speed (for smaller values like 32)
      bits_(static_cast<size_t>(ceil(params_.m / 64.0)), 0ULL)
{
    cout << "[bloom_filter] m=" << params_.m
         << " bits  k=" << params_.k
         << "  backing=" << bits_.size() << " uint64 words\n";
}

void BloomFilter::insert(const string& asin) {
    uint64_t h1 = fnv1a(asin);
    uint64_t h2 = djb2(asin);

    // for all hash functions
    for (int i = 0; i < params_.k; i++) {
        // Double hashing: derive k positions from two base hashes
        // pos = (int)(h1 + (int)i * h2) % (int)params_.m
        int pos = static_cast<int>((h1 + static_cast<uint64_t>(i) * h2) % static_cast<uint64_t>(params_.m));
        set_bit(pos);
    }
}

bool BloomFilter::probably_seen(const string& asin) const {
    uint64_t h1 = fnv1a(asin);
    uint64_t h2 = djb2(asin);

    // for all hash functions
    for (int i = 0; i < params_.k; i++) {
        int pos = static_cast<int>((h1 + static_cast<uint64_t>(i) * h2) % static_cast<uint64_t>(params_.m));
        // If any bit is 0 → item was definitely not inserted
        if (!get_bit(pos)) return false;
    }
    // All k bits set → item was probably inserted
    return true;
}

void BloomFilter::set_bit(int pos) {
    // pos / 64 → which uint64_t word
    // pos % 64 → which bit within that word
    bits_[pos / 64] |= (1ULL << (pos % 64));
}

bool BloomFilter::get_bit(int pos) const {
    return (bits_[pos / 64] >> (pos % 64)) & 1ULL;
}

uint64_t BloomFilter::fnv1a(const std::string& s) const {
    // FNV-1a 64-bit — standard constants from the FNV spec
    uint64_t hash  = 14695981039346656037ULL; // FNV offset basis
    uint64_t prime = 1099511628211ULL;         // FNV prime

    for (unsigned char c : s) {
        hash ^= c;
        hash *= prime;
    }
    return hash;
}

uint64_t BloomFilter::djb2(const std::string& s) const {
    // djb2 — Dan Bernstein's classic string hash
    // chosen as h2 because it has different bit mixing than FNV-1a,
    // reducing correlation between the two base hashes
    uint64_t hash = 5381;
    for (unsigned char c : s)
        hash = ((hash << 5) + hash) + c; // hash * 33 + c
    return hash;
}