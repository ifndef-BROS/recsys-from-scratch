## Reviewing Amazon Reviews Dataset sub category
```Software
  Users: 2,589,467
  Items: 89,247
  Interactions: 4,828,481
  Avg per user: 1.9
  Median per user: 1.0
  Users with 5+ interactions: 157,078
```

```Video_Games
  Users: 2,766,657
  Items: 137,250
  Interactions: 4,555,501
  Avg per user: 1.6
  Median per user: 1.0
  Users with 5+ interactions: 113,809
```

```Musical_Instruments
  Users: 1,762,680
  Items: 213,572
  Interactions: 2,975,552
  Avg per user: 1.7
  Median per user: 1.0
  Users with 5+ interactions: 82,396
```

```Sports_and_Outdoors
  Users: 10,331,142
  Items: 1,587,220
  Interactions: 19,349,404
  Avg per user: 1.9
  Median per user: 1.0
  Users with 5+ interactions: 616,968
```

Since software has the highest 5+ interactions and lesser number of items, we will go with this. As we have bettwe interaction data and lesser items means less time it will take to generate embeddings

## Output of 02_check_data.py

```bash
Memory: 0.11 GB
rating: double
title: string
text: string
images: list<item: struct<small_image_url: string, medium_image_url: string, large_image_url: string, attach (... 19 chars omitted)
  child 0, item: struct<small_image_url: string, medium_image_url: string, large_image_url: string, attachment_type:  (... 7 chars omitted)
      child 0, small_image_url: string
      child 1, medium_image_url: string
      child 2, large_image_url: string
      child 3, attachment_type: string
asin: string
parent_asin: string
user_id: string
timestamp: int64
helpful_vote: int64
verified_purchase: bool
Memory: 2.78 GB
Data loaded
         rating                                     title  ... helpful_vote verified_purchase
0           1.0                                   malware  ...            0             False
1           5.0                               Lots of Fun  ...            0              True
2           5.0                         Light Up The Dark  ...            0              True
3           4.0                                  Fun game  ...            0              True
4           4.0  I am not that good at it but my kids are  ...            0              True
...         ...                                       ...  ...          ...               ...
4880176     5.0                                       Gog  ...            0              True
4880177     1.0                           WORST GAME EVER  ...            1              True
4880178     5.0                                 better!!!  ...            2              True
4880179     5.0         It Has Everything I Need And More  ...            0              True
4880180     5.0                                  Huge fan  ...            0              True

[4880181 rows x 10 columns]
=== Basic Stats ===
Total interactions: 4,880,181
Unique users: 2,589,466
Unique items: 89,246
Sparsity: 99.9979%

=== Interaction Distribution ===
Avg interactions per user: 1.9
Median: 1.0
90th percentile: 3
99th percentile: 12
Max: 371

=== User Buckets ===
Users with 1 interaction:   1,743,452
Users with 2-4 interactions:685,758
Users with 5-20:            152,592
Users with 20+:             7,664

=== Rating Distribution ===
rating
1.0     695854
2.0     239253
3.0     419356
4.0     857082
5.0    2668636
Name: count, dtype: int64

=== Timestamp Range ===
Earliest: 1999-03-15 04:02:39
Latest: 2023-09-11 02:13:11.515000

=== Item Interaction Distribution ===
Avg interactions per item: 54.7
Items with < 5 interactions: 53,048
Items with 20+ interactions: 14,515
```

## How I handled the memory issues with WSL when loading the dataset
When loading the dataset. WSL started to crash. After a lot of debugging. I finally figured it out that the lack of memory is the issue. It is crashing due to the large size of the data set. Which is about 2 GB (with all the python stuff and other things running inside wsl). So, I changed the WSL memory limit to 8 GB RAM and 4 GB Swap. It should have fixed the issue. But it didn't. I still do not know why it isn't running. But it does run when I use `pyarrow` instead of `pandas`. I loaded the json data as pyarrow table and converted it into pandas dataframe later as I am more familiar with pandas. The dataset loads and there are no memory issues or laptop overheating. I looked into pyarrow and understood why it is more efficient than vanilla pandas.

## On filtering the data
Not all data points are useful for us. The users and items which are useful to us are the ones with the most amount of interactions so our model can generalize well. So, we are:
- sorting users by timestamps of interactions
- Kept users with 5+ interactions as lesser the number of interactions, poorer the quality of training and testing set for a user
- Items with more than 5 unique ratings. For our MVP we are purging less popular items, for which we might have to handle the cold start scenario 

## Designing the Bloom Filter
The bloom filter has the following parameters:
- `m`: The size of bit array
- `k`: Number of hash functions
We have two more parameters at a higher level of abstraction, which would let us decide the optimal value of `m` and `k` for each user with unique habits.
- `n`: Number of average items the user looks at
- `p`: The acceptable false positive rate which we are willing to tolerate (probability of interpretation of not seen items as seen)

According to our stats:
```
Avg:  1.9 interactions
90th: 3
99th: 12
Max:  371
```
`n` = 2. But this only covers the average user and not everyone. Hence, it would be a better choice to take `n` = 12

Now, given `n` and `p` we can calculate the values of `m` and `k` (refer to the literature below)
```
m = -( n × ln(p) ) / (ln(2))²

k =  ( m / n ) × ln(2)
```
based on these formulae and assuming `p=0.01` or 1% we get `k=7`

### References for Bloom Filters
- [im2005b.pdf](https://www.eecs.harvard.edu/~michaelm/postscripts/im2005b.pdf)
- [/tmp/CS-2002-10.dvi - CS-2002-10.pdf](https://cdn.dal.ca/content/dam/dalhousie/pdf/faculty/computerscience/technical-reports/CS-2002-10.pdf)
- [1804.04777v2.pdf](https://arxiv.org/pdf/1804.04777)
- [Less hashing, same performance: Building a better Bloom filter - rsa2008.pdf](https://www.eecs.harvard.edu/~michaelm/postscripts/rsa2008.pdf)

### References for KDTree
- [Algorithms for calculating variance - Wikipedia](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance). This page has some rather interesting ways to calculate variance. I found it while looking for cache friendly ways to calculate variance. But found numerically stable ones instead.

### Ranker
The key implementation detail worth noting for your report is the max_sim_to_selected incremental update at the bottom of the MMR loop. Instead of recomputing similarity from all selected items for all candidates at every iteration — which would be O(n·k²·DIM) — you only update using the newly selected item, keeping it at O(n·k·DIM). This is the standard MMR optimisation and worth one sentence in the complexity analysis.


## Failing tests
```bash
dev@2a1a8724a8f7:/app$ mkdir -p build && cd build && cmake .. && make
-- Configuring done
-- Generating done
-- Build files have been written to: /app/build
[ 25%] Building CXX object CMakeFiles/main.dir/src/main.cpp.o
[ 50%] Linking CXX executable bin/main
[100%] Built target main
dev@2a1a8724a8f7:/app/build$ cd ..
dev@2a1a8724a8f7:/app$ ./build/bin/main
recsys-from-scratch — data loader + bloom filter tests

══════════════════════════════════════
  1. bloom_params()
══════════════════════════════════════
  [PASS] n=2   m in expected range [19,21]
  [PASS] n=12  m in expected range [114,118]
  [PASS] n=371 m in expected range [3550,3560]
  [PASS] n=2   k=7 for p=0.01
  [PASS] n=12  k=7 for p=0.01
  [PASS] n=371 k=7 for p=0.01
  [PASS] p=0.10 → k in [3,4]
  [PASS] relaxed p → smaller m

══════════════════════════════════════
  2. BloomFilter insert / query
══════════════════════════════════════
[bloom_filter] m=96 bits  k=7  backing=2 uint64 words
  [PASS] no false negatives on inserted items
  [INFO] false positives: 0 / 100
  [PASS] false positive rate under 10% (expected ~1%)
[bloom_filter] m=48 bits  k=7  backing=1 uint64 words
[bloom_filter] m=48 bits  k=7  backing=1 uint64 words
  [PASS] double insert is idempotent
[bloom_filter] m=48 bits  k=7  backing=1 uint64 words
  [PASS] empty filter returns false
[bloom_filter] m=116 bits  k=7  backing=2 uint64 words
  [PASS] params() returns correct m and k
[data_loader] embeddings : 89251 items × 384 dims
[data_loader] train      : 150198 users
[data_loader] test       : 150198 users

══════════════════════════════════════
  3. load_embeddings()
══════════════════════════════════════
  [PASS] embeddings matrix is non-empty
  [PASS] asin_to_idx map is non-empty
  [PASS] idx_to_asin map is non-empty
  [PASS] asin_to_idx and idx_to_asin have same size
  [PASS] embedding matrix rows == number of indexed items
  [PASS] all row indices are within embedding matrix bounds
  [PASS] asin → idx → asin roundtrip is consistent
  [PASS] sample embedding is unit-normalised (|norm - 1| < 1e-3)
  [PASS] no all-zero embedding vectors
  [INFO] total items: 89251
  [INFO] sample asin: B0751Q59HC  row: 89250  norm: 1

══════════════════════════════════════
  4. load_train()
══════════════════════════════════════
  [PASS] user_history is non-empty
  [PASS] all users have >= 5 interactions
  [PASS] all ratings in [1.0, 5.0]
  [PASS] no empty ASINs in train history
  [INFO] users: 150198
  [INFO] total interactions: 1164101
  [INFO] avg per user: 7.75044
  [INFO] max per user: 295

══════════════════════════════════════
  5. load_test()
══════════════════════════════════════
  [PASS] ground_truth is non-empty
  [PASS] all test users exist in train
  [FAIL] ground truth item never appears in user train history  (line 301)
  [PASS] no empty ground truth ASINs
  [INFO] test users: 150198

══════════════════════════════════════
  6. End-to-end: real user Bloom filter
══════════════════════════════════════
  [INFO] user:            AHZZYDN7XZXJRETMPWW4RRD4PS2Q
  [INFO] history length:  7
  [INFO] ground truth:    B075B2RMT2
[bloom_filter] m=68 bits  k=7  backing=2 uint64 words
  [PASS] all train items marked seen (no false negatives)
  [PASS] ground truth item is NOT marked seen
  [PASS] ground truth item has an embedding
  [PASS] all train items have embeddings

══════════════════════════════════════
  Results: 33 passed, 1 failed
══════════════════════════════════════
```
1 test is failing: maybe because of data duplication in the dataset (same millisecond timestamp issue) causing data leak. Investigate dataset and update 03_filtering.py


### Evaluation
- The first script without any optimizations took
[pipeline] 10000 / 146980  (1167.79s  8.56321 users/s)

- The first script with max compiler optimizations took
[pipeline] 3800 / 146980  (132.91s  28.5907 users/s)

- After reducing the number of candidates to rank after kdtree from 500 to 100
[pipeline] 600 / 146980  (18.9958s  31.5859 users/s)

- After replacing l2_squared with SIMD dot product
[pipeline] 146900 / 146980  (2270.07s  64.7117 users/s)
[pipeline] done.  processed=146980  skipped=0  time=2271.29s  avg=15.453ms/user
Ground truth users: 146,980
── cosine_sort ──
 K  Recall@K   NDCG@K  n_users
 5  0.010484 0.006770   146980
10  0.016846 0.008808   146980
20  0.016846 0.008808   146980 // we only save top 20 recs that is why it is same
── adaptive_mmr ──
 K  Recall@K   NDCG@K  n_users
 5  0.009729 0.006577   146980
10  0.011798 0.007240   146980
20  0.011798 0.007240   146980 // we only save top 20 recs that is why it is same

- After OMP
[pipeline] inferred 600 / 146980  (2.46439s  243.468 users/s)
[pipeline] done.  processed=146980  skipped=0  time=640.556s  avg=4.35812ms/user
── cosine_sort ──
 K  Recall@K   NDCG@K  n_users
 5  0.010484 0.006770   146980
10  0.016846 0.008808   146980
20  0.016846 0.008808   146980

── adaptive_mmr ──
 K  Recall@K   NDCG@K  n_users
 5  0.009729 0.006577   146980
10  0.011798 0.007240   146980
20  0.011798 0.007240   146980