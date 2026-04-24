/**
 * @file    kdtree.cpp
 * @brief   Implementation of KD-tree nearest neighbour search.
 *
 * Build strategy:
 *   1. Find highest-variance dimension (Welford's, cache-friendly gather)
 *   2. Sort points by that dimension — O(n log n)
 *   3. Split on median → balanced tree, O(log n) depth
 *   4. Recurse until node has <= LEAF_SIZE points
 *
 * Search strategy:
 *   1. Traverse to the leaf the query point falls in (near subtree first)
 *   2. Check all leaf points, push candidates onto max-heap of size k
 *   3. Backtrack — check if splitting hyperplane is closer than heap's worst
 *   4. If yes → search far subtree. If no → prune it.
 *
 * Why max-heap of size k:
 *   We want to evict the worst candidate when a better one is found.
 *   Max-heap keeps the worst result at top → O(log k) eviction.
 *   At the end we drain and reverse-sort for ascending distance output.
 *
 * Row-major vs column-major tradeoff:
 *   Row-major (current): query distance computation is sequential (Yes)
 *                        variance gather is random access (No)
 *   Column-major:        variance gather is sequential (Yes)
 *                        query distance computation is random access (No)
 *   Query runs O(n_users × k × log n) times vs build runs once.
 *   Row-major wins overall.
 */

#include "kdtree.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

KDTree::KDTree(const std::vector<embedding_t>& embeddings)
    : embeddings_(embeddings)
{
    // Build index over all item rows
    std::vector<int> all_indices(embeddings.size());
    std::iota(all_indices.begin(), all_indices.end(), 0);  // 0, 1, 2, ..., n-1

    root_ = build(std::move(all_indices));

    std::cout << "[kdtree] built over " << embeddings.size() << " items"
              << "  leaf_size=" << LEAF_SIZE << "\n";
}


std::unique_ptr<KDTree::Node> KDTree::build(std::vector<int> indices) {
    auto node = std::make_unique<Node>();

    // Base case — small enough to search exhaustively at query time
    if (static_cast<int>(indices.size()) <= LEAF_SIZE) {
        node->indices   = std::move(indices);
        node->split_dim = -1;
        node->split_val = 0.0f;
        return node;
    }

    // Find best split dimension — highest variance separates points most
    node->split_dim = highest_variance_dim(indices);

    // Sort by split dimension — O(n log n)
    // After sort, median element gives a balanced split
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return embeddings_[a][node->split_dim] < embeddings_[b][node->split_dim];
    });

    // Split on median — guarantees balanced tree → O(log n) depth
    int median      = static_cast<int>(indices.size()) / 2;
    node->split_val = embeddings_[indices[median]][node->split_dim];

    // Partition: left gets [0, median), right gets [median, n)
    std::vector<int> left_idx (indices.begin(), indices.begin() + median);
    std::vector<int> right_idx(indices.begin() + median, indices.end());

    node->left  = build(std::move(left_idx));
    node->right = build(std::move(right_idx));

    return node;
}

int KDTree::highest_variance_dim(const std::vector<int>& indices) const {
    int   best_dim = 0;
    float best_var = -1.0f;
    int   n        = static_cast<int>(indices.size());

    // Contiguous buffer — Welford pass reads this sequentially
    // Allocated once per call, reused across all DIM iterations
    std::vector<float> dim_vals(n);

    for (int d = 0; d < DIM; d++) {

        // Phase 1: Gather 
        // Pack all values for dim d into contiguous buffer.
        // Access pattern: embeddings_[random_row][d] — unavoidably random.
        // Separating this from compute means the Welford pass is fully sequential.
        for (int i = 0; i < n; i++)
            dim_vals[i] = embeddings_[indices[i]][d];

        // Phase 2: Welford's online mean + variance (single pass) 
        //
        // Standard two-pass formula:
        //   pass 1: mean = Σx / n
        //   pass 2: var  = Σ(x - mean)² / n
        //
        // Welford's recurrence (Welford 1962, Knuth TAOCP Vol.2 §4.2.2):
        //   delta  = x - mean_prev
        //   mean  += delta / (i + 1)
        //   delta2 = x - mean_new        ← uses UPDATED mean, key to stability
        //   M2    += delta * delta2
        //   var    = M2 / n
        //
        // For unit-normalised embeddings in [-1,1], numerical stability is
        // not a practical concern — Welford's is used for its single-pass
        // property (n iterations vs 2n) and principled general correctness.
        float mean = 0.0f;
        float M2   = 0.0f;

        for (int i = 0; i < n; i++) {
            float x      = dim_vals[i];
            float delta  = x - mean;
            mean        += delta / static_cast<float>(i + 1);
            float delta2 = x - mean;     // note: mean is already updated
            M2          += delta * delta2;
        }

        float var = M2 / static_cast<float>(n);

        if (var > best_var) {
            best_var = var;
            best_dim = d;
        }
    }

    return best_dim;
}

std::vector<KNNResult> KDTree::query(const embedding_t& query, int k) const {
    // Max-heap — worst result (largest distance) at top
    // When heap is full and we find a better candidate:
    //   pop worst → push new → O(log k)
    std::priority_queue<KNNResult, std::vector<KNNResult>, KNNComparator> heap;

    search(root_.get(), query, k, heap);

    // Drain heap into vector and sort ascending (closest first)
    std::vector<KNNResult> results;
    results.reserve(heap.size());
    while (!heap.empty()) {
        results.push_back(heap.top());
        heap.pop();
    }
    std::sort(results.begin(), results.end(), [](const KNNResult& a, const KNNResult& b) {
        return a.squared_dist < b.squared_dist;
    });

    return results;
}

void KDTree::search(
    const Node* node,
    const embedding_t& query,
    int k,
    std::priority_queue<KNNResult, std::vector<KNNResult>, KNNComparator>& heap
) const {
    if (node == nullptr) return;

    // Leaf node — exhaustive check 
    if (node->is_leaf()) {
        for (int idx : node->indices) {
            float dist = squared_l2(query, embeddings_[idx]);

            if (static_cast<int>(heap.size()) < k) {
                // Heap not full — always add
                heap.push({idx, dist});
            } else if (dist < heap.top().squared_dist) {
                // Better than current worst — evict worst, add new
                heap.pop();
                heap.push({idx, dist});
            }
        }
        return;
    }

    // Internal node — traverse closer subtree first 
    // The subtree on the same side as the query is more likely to contain
    // nearest neighbours — searching it first fills the heap faster,
    // making pruning of the far subtree more likely to trigger
    float diff      = query[node->split_dim] - node->split_val;
    const Node* near = (diff <= 0) ? node->left.get()  : node->right.get();
    const Node* far  = (diff <= 0) ? node->right.get() : node->left.get();

    search(near, query, k, heap);

    // Pruning
    // The closest point in the far subtree is at least |diff| away
    // along the split dimension. If diff² >= current worst distance,
    // no point in the far subtree can beat our current k results → prune.
    //
    // At high dims (384) this rarely triggers — most far subtrees are
    // searched, degrading toward O(n). PCA to 64 dims restores pruning.
    float dist_to_plane = diff * diff;
    bool  heap_full     = static_cast<int>(heap.size()) >= k;
    bool  can_prune     = heap_full && (dist_to_plane >= heap.top().squared_dist);

    if (!can_prune)
        search(far, query, k, heap);
}