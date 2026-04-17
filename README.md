# recsys-from-scratch
We are attempting to make a recommendation system in C++. We are going to submit this as our Design and Analysis of Algorithms project.

We are making a history based recsys. Given the time we have to implement this. This would work extremely well for us. As updating and generating the user embeddings would be very easy and for the given constraints (time, compute, and no external libraries) this works really well. But the cons being, order of interactions will be ignored. Eg: "User bought charger before phone" and "User bought phone before charger" will be same for our system. But this is a trade off we can make. One actual concern might be users with longer history of interaction. As the avg will get very noisy. But for the MVP we will be ignoring that constraint.

One more important choice is, what are we making a recommnedation system for? Honestly, there are many choices. Like songs, movies, products, [GitHub Repositories](https://github.com/anshulbadhani/github_recsys) or something else? We are making it for recommending which item the user might be interested in buying based on previous interactions.

Input: set of items user has previously rated/interacted with
Output: ranked list of K items the user hasn't seen, ordered by predicted relevance

Evaluation: you hold out the last item each user interacted with, try to retrieve it in your top-K, and measure Recall@K and NDCG@K. aka `leave one out` evaluation. (We will get baselines to compare with)

Dataset source: https://mcauley.ucsd.edu/data/amazon_2023/ [3]


## Proposed Architecture
- Item tower — MiniLM offline, once, in Python
- User tower — weighted average of item embeddings, at inference, in C++
- Retrieval — KD-tree in C++
- Filtering — Bloom filter in C++
- Ranking — MMR or cosine sort in C++ (maybe LinUCB if we could manage the online side of things)


## Evaluation
- Basic ones
    - Random – recommend random unseen items
    - Popularity-based – recommend the most interacted-with items globally
    - Most recent item – recommend items similar to the last item the user interacted with. One nearest neighbor in embedding space
- Standard ones
    - [[1205.2618] BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618)
    - ItemKNN – find items similar to what the user has interacted with using item-item cosine similarity
- Upper bound reference (just for citing, not feasible to implement from scratch)
    - SASRec — sequential transformer model, state of the art on most Amazon category benchmarks. 

Will use `Cornac` or `implicit` to benchmark and not re-invent the wheel

## Papers
- [[1205.2618] BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618) [1]
- [[1708.05031] Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)
- [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/) [2]
- [[2006.11632] Embedding-based Retrieval in Facebook Search](https://arxiv.org/abs/2006.11632)
- [[1602.01585] Ups and Downs: Modeling the Visual Evolution of Fashion Trends with One-Class Collaborative Filtering](https://arxiv.org/abs/1602.01585)
- [[1808.09781] Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781) [4]
- [[1603.09320] Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320) [5]
- [1807.05614v2.pdf](https://arxiv.org/pdf/1807.05614)

## Running the project
```bash
docker-compose build
docker-compose up -d
docker-compose exec recsys bash
```
```bash
mkdir -p build && cd build && cmake .. && make
./bin/main
```