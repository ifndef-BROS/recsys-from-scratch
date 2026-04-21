/**
 * @file    kdtree.cpp
 * @brief   Implementation of KD-tree nearest neighbour search.
 *
 * Build strategy:
 *   1. Find highest-variance dimension among current points
 *   2. Sort points by that dimension
 *   3. Split on median — left gets lower half, right gets upper half
 *   4. Recurse until node has <= LEAF_SIZE points
 *
 * Search strategy:
 *   1. Traverse to the leaf the query point falls in
 *   2. Check all leaf points, push candidates onto max-heap of size k
 *   3. Backtrack — for each internal node, check if the opposite
 *      subtree's splitting hyperplane is closer than heap's worst result
 *   4. If yes, search that subtree too. If no, prune it.
 *
 * Pruning condition:
 *   dist_to_hyperplane² = (query[split_dim] - split_val)²
 *   If dist_to_hyperplane² >= heap.top().squared_dist → prune
 *
 * Note on high dimensionality:
 *   At 384 dims, dist_to_hyperplane is almost always < heap.top().squared_dist
 *   because points are spread across many dimensions. Pruning rarely fires.
 *   Apply PCA before building to restore meaningful pruning.
 */

#include "kdtree.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <cassert>
#include <memory>

using std::vector,
    std::move,
    std::unique_ptr,
    std::make_unique,
    std::sort,
    std::cout,
    std::priority_queue,
    std::iota;

KDTree::KDTree(const vector<embedding_t>& embeddings) : embeddings_(embeddings) {
    // Build index over all rows
    vector<int> all_indices(embeddings.size());
    iota(all_indices.begin(), all_indices.end(), 0); // fill 0, 1, 2... n-1

    root_ = build(move(all_indices)); // understand this and xvalues in cpp

    cout << "[kdtree] built over " << embeddings.size() << " items"
              << "  leaf_size=" << LEAF_SIZE << "\n";
}

unique_ptr<KDTree::Node> KDTree::build(vector<int> indices) {
    auto node = make_unique<Node>();

    // Base case — small enough to search exhaustively
    if (static_cast<int>(indices.size()) <= LEAF_SIZE) {
        node->indices = move(indices);
        node->split_dim = -1;
        node->split_val = 0.0f;
        return node;
    }

    // Find dimension with highest variance — best split axis
    node->split_dim = highest_variance_dim(indices);

    // Sort indices by the split dimension
    sort(indices.begin(), indices.end(), [&](int a, int b) {
        return embeddings_[a][node->split_dim] < embeddings_[b][node->split_dim];
    });

    // Split on median — left gets lower half, right gets upper half
    // Median index ensures balanced tree → O(log n) depth
    int median     = static_cast<int>(indices.size()) / 2;
    node->split_val = embeddings_[indices[median]][node->split_dim];

    // Partition into left and right
    vector<int> right_idx(indices.begin() + median, indices.end());
    vector<int> left_idx (indices.begin(), indices.begin() + median);

    node->left  = build(move(left_idx));
    node->right = build(move(right_idx));

    return node;
}

int KDTree::highest_variance_dim(const vector<int>& indices) const {
    int   best_dim  = 0;
    float best_var  = -1.0f;

    for (int d = 0; d < DIM; d++) {
        // Compute mean along dim d
        float mean = 0.0f;
        for (int idx : indices)
            mean += embeddings_[idx][d];
        mean /= static_cast<float>(indices.size());

        // Compute variance along dim d
        float var = 0.0f;
        for (int idx : indices) {
            float diff = embeddings_[idx][d] - mean;
            var += diff * diff;
        }

        if (var > best_var) {
            best_var = var;
            best_dim = d;
        }
    }

    return best_dim;
}

vector<KNNResult> KDTree::query(const embedding_t& query, int k) const {
    // Max-heap — worst result (largest distance) at top
    // Lets us efficiently check if a new candidate beats the current worst
    priority_queue<KNNResult, vector<KNNResult>, KNNComparator> heap;

    search(root_.get(), query, k, heap);

    // Drain heap into sorted vector (closest first)
    vector<KNNResult> results;
    results.reserve(heap.size());
    while (!heap.empty()) {
        results.push_back(heap.top());
        heap.pop();
    }
    sort(results.begin(), results.end(), [](const KNNResult& a, const KNNResult& b) {
        return a.squared_dist < b.squared_dist;
    });

    return results;
}

void KDTree::search(
    const Node* node,
    const embedding_t& query,
    int k,
    priority_queue<KNNResult, vector<KNNResult>, KNNComparator>& heap
) const {
    if (node == nullptr) return;

    // Leaf node — check all points exhaustively 
    if (node->is_leaf()) {
        for (int idx : node->indices) {
            float dist = squared_l2(query, embeddings_[idx]);

            if (static_cast<int>(heap.size()) < k) {
                // Heap not full yet — always add
                heap.push({idx, dist});
            } else if (dist < heap.top().squared_dist) {
                // Better than current worst — replace it
                heap.pop();
                heap.push({idx, dist});
            }
        }
        return;
    }

    // Internal node — traverse closer subtree first 
    float diff = query[node->split_dim] - node->split_val;

    // The subtree on the same side as the query point is more likely
    // to contain nearest neighbours — search it first
    const Node* near_node = (diff <= 0) ? node->left.get()  : node->right.get();
    const Node* far_node  = (diff <= 0) ? node->right.get() : node->left.get();

    search(near_node, query, k, heap);

    // Pruning check
    // Distance from query to the splitting hyperplane (squared)
    // If this is >= our current worst result, the far subtree cannot
    // contain a closer point — prune it entirely
    float dist_to_plane = diff * diff;

    bool heap_full       = static_cast<int>(heap.size()) >= k;
    bool plane_too_far   = heap_full && (dist_to_plane >= heap.top().squared_dist);

    if (!plane_too_far)
        search(far_node, query, k, heap);
}


float KDTree::squared_l2(const embedding_t& a, const embedding_t& b) const {
    float dist = 0.0f;
    for (int d = 0; d < DIM; d++) {
        float diff = a[d] - b[d];
        dist += diff * diff;
    }
    return dist;
}