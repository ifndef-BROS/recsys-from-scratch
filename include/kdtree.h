/**
 * @file    kdtree.h
 * @brief   KD-tree for approximate nearest neighbour search in embedding space.
 *
 * Recursively partitions the embedding space by splitting on the dimension
 * of highest variance at each node. This maximises separation between points
 * at each split, improving pruning during search.
 *
 * Complexity:
 *   Build : O(n log² n)  — O(log n) levels, O(n log n) sort per level
 *   Query : O(log n) best case, O(n) worst case
 *
 * Curse of dimensionality:
 *   At 384 dims, pruning rarely triggers — query approaches O(n).
 *   Apply PCA to 64 dims before building to recover meaningful speedup.
 *   Changing DIM in data_loader.h is sufficient — no other code changes needed.
 *
 * Split strategy:
 *   Highest-variance dimension at each node — maximises point separation.
 *   Variance computed via Welford's online algorithm (single pass, numerically
 *   stable). Values gathered into a contiguous buffer first for cache-friendly
 *   sequential access during the Welford pass.
 *
 * Cache behaviour:
 *   Gather pass  : unavoidably random (arbitrary row indices in 21MB matrix)
 *   Welford pass : fully sequential (contiguous dim_vals buffer)
 *   Query pass   : sequential per distance computation (row-major optimal)
 *   Row-major storage is chosen to optimise query over build — query runs
 *   far more often than build in practice.
 *
 * Leaf size:
 *   Recursion stops when a node contains <= LEAF_SIZE points.
 *   These are checked exhaustively during query.
 *   LEAF_SIZE=10 is a good default — tune based on dimensionality.
 *
 * @author  anshulbadhani
 * @date    2026
 */

#pragma once

#include "data_loader.h"
#include <vector>
#include <array>
#include <string>
#include <queue>
#include <memory>

/// Leaf nodes with <= LEAF_SIZE points are searched exhaustively.
/// Larger values → shallower tree, more exhaustive search at leaves.
constexpr int LEAF_SIZE = 10;

/**
 * @brief A single query result — item row index and distance to query point.
 *
 * Distance is squared L2 for efficiency (avoids sqrt during search).
 * Since embeddings are unit-normalised:
 *   squared_l2 = 2 - 2 * cosine_similarity
 * So minimising squared L2 is equivalent to maximising cosine similarity.
 */
struct KNNResult {
    int   row;           ///< Row index in the embedding matrix
    float squared_dist;  ///< Squared L2 distance to query point
};

/// Max-heap comparator — largest distance at top, so we can evict worst easily
struct KNNComparator {
    bool operator()(const KNNResult& a, const KNNResult& b) {
        return a.squared_dist < b.squared_dist;
    }
};


/**
 * @brief Computes squared L2 distance between two embedding vectors.
 *
 * Exposed as a free function so tests can use it for brute-force comparison
 * without instantiating a KDTree.
 *
 * For unit-normalised vectors:
 *   ||a - b||² = 2 - 2 * dot(a, b)
 * Minimising squared L2 == maximising cosine similarity.
 *
 * Avoids sqrt — valid for comparison since sqrt is monotonically increasing.
 */
inline float squared_l2(const embedding_t& a, const embedding_t& b) {
    float dist = 0.0f;
    for (int d = 0; d < DIM; d++) {
        float diff = a[d] - b[d];
        dist += diff * diff;
    }
    return dist;
}

/**
 * @brief KD-tree over the item embedding matrix.
 *
 * Built once at startup from item_embeddings. Queried once per user
 * to retrieve top-K nearest items in embedding space.
 *
 * Typical usage:
 * @code
 *   KDTree tree(embeddings);
 *   auto results = tree.query(user_vec, 100);  // top-100 candidates
 *   // filter through Bloom filter, rank, return top-K
 * @endcode
 */
class KDTree {
public:
    /**
     * @brief Builds the KD-tree from the item embedding matrix.
     *
     * @param embeddings  Row-indexed item embedding matrix (from load_embeddings).
     */
    explicit KDTree(const std::vector<embedding_t>& embeddings);

    /**
     * @brief Returns the k nearest items to the query vector.
     *
     * Results are sorted by ascending squared L2 distance (closest first).
     * Since embeddings are unit-normalised, this is equivalent to
     * descending cosine similarity.
     *
     * @param query  Query vector (user embedding). Must be unit-normalised.
     * @param k      Number of nearest neighbours to return.
     * @return       Up to k results sorted by distance ascending.
     */
    std::vector<KNNResult> query(const embedding_t& query, int k) const;

    /**
     * @brief Returns the total number of items indexed in the tree.
     */
    int size() const { return static_cast<int>(embeddings_.size()); }

private:

    /**
     * @brief A single node in the KD-tree.
     *
     * Internal nodes store a split dimension and split value.
     * Leaf nodes store the indices of their points directly.
     *
     * Note: unique_ptr children mean every tree traversal follows pointers
     * to non-contiguous heap allocations — unavoidable cache miss per node.
     * A flat array-based tree would be more cache-friendly but significantly
     * more complex to implement with variable-size subtrees.
     */
    struct Node {
        int   split_dim;  ///< Dimension used to split (highest variance)
        float split_val;  ///< Median value along split_dim

        /// Point indices — non-empty only at leaf nodes
        std::vector<int> indices;

        /// Child nodes — null for leaves
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;

        bool is_leaf() const { return left == nullptr; }
    };


    /// Reference to the embedding matrix — not owned by the tree
    const std::vector<embedding_t>& embeddings_;

    /// Root of the tree
    std::unique_ptr<Node> root_;


    /**
     * @brief Recursively builds a subtree over the given point indices.
     *
     * @param indices  Row indices of points in this subtree.
     * @return         Owning pointer to the root of this subtree.
     */
    std::unique_ptr<Node> build(std::vector<int> indices);

    /**
     * @brief Finds the dimension of highest variance among the given points.
     *
     * Two-phase approach for cache efficiency:
     *   1. Gather: pack dim values into contiguous buffer (random access — unavoidable)
     *   2. Welford: single-pass mean+variance on contiguous buffer (sequential)
     *
     * Welford's algorithm reference:
     *   Welford, B.P. (1962). Technometrics 4(3): 419–420.
     *   Also: Knuth, TAOCP Vol.2, Section 4.2.2.
     *
     * Numerical stability note:
     *   For unit-normalised embeddings in [-1, 1], catastrophic cancellation
     *   is not a practical concern. Welford's is used for its single-pass
     *   property (n iterations vs 2n) and as a principled general implementation.
     *
     * @param indices  Row indices to scan.
     * @return         Dimension index with highest variance.
     */
    int highest_variance_dim(const std::vector<int>& indices) const;


    /**
     * @brief Recursively searches a subtree for nearest neighbours.
     *
     * Uses a max-heap of size k to track the best candidates found so far.
     * Prunes branches whose closest possible point is farther than the
     * current k-th best distance.
     *
     * Pruning condition:
     *   dist_to_plane² = (query[split_dim] - split_val)²
     *   If dist_to_plane² >= heap.top().squared_dist → prune far subtree
     *
     * @param node   Current node being searched.
     * @param query  Query vector.
     * @param k      Number of neighbours to find.
     * @param heap   Max-heap of current best results (size <= k).
     */
    void search(
        const Node* node,
        const embedding_t& query,
        int k,
        std::priority_queue<KNNResult, std::vector<KNNResult>, KNNComparator>& heap
    ) const;
};