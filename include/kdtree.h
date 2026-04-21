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
 *   Changing DIM in config.h is sufficient — no other code changes needed.
 *
 * Split strategy:
 *   At each node, scan all points to find the dimension with highest
 *   variance. Split on the median value of that dimension. Left subtree
 *   receives points below median, right subtree receives points above.
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
#include <string>
#include <array>
#include <memory>
#include <queue>

using std::vector,
    std::string,
    std::array,
    std::queue;

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
    int row;            ///< Row index in the embedding matrix
    float squared_dist; ///< Squared L2 distance to query point
};

/// Min-heap comparator — smallest distance at top
struct KNNComparator {
    bool operator()(const KNNResult& a, const KNNResult& b) {
        return a.squared_dist > b.squared_dist;
    }
};

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
    explicit KDTree(const vector<embedding_t>& embeddings);

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
    vector<KNNResult> query(const embedding_t& query, int k) const;
    
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
     */
    struct Node {
        int   split_dim;    ///< Dimension used to split (highest variance)
        float split_val;    ///< Median value along split_dim

        /// Point indices at this node (non-empty only for leaves)
        std::vector<int> indices;

        /// Child nodes (null for leaves)
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;

        bool is_leaf() const { return left == nullptr && right == nullptr; }
    };

    /// Reference to the embedding matrix — not owned by the tree
    const vector<embedding_t>& embeddings_;

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
     * Scans all points × all dims — O(n × DIM).
     * Called once per node during build.
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

    /**
     * @brief Computes squared L2 distance between two embedding vectors.
     *
     * Squared distance avoids sqrt — valid for comparison since sqrt is
     * monotonically increasing.
     *
     * For unit-normalised vectors:
     *   ||a - b||² = 2 - 2 * dot(a, b)
     *
     * @return Squared L2 distance.
     */
    float squared_l2(const embedding_t& a, const embedding_t& b) const;
};

// Free function for brute force comparison in tests
inline float squared_l2_free(const embedding_t& a, const embedding_t& b) {
    float dist = 0.0f;
    for (int d = 0; d < DIM; d++) {
        float diff = a[d] - b[d];
        dist += diff * diff;
    }
    return dist;
}