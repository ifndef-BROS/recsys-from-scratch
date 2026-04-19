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