# Bug logs
This file include bug encountered while the development. Does not contain trivial issues like: `unordered_map<int, string>` instead of `unordered_map<string, int>`. *This bug report was written by Claude while debugging*.

### Bug 1 – Deduplication after filtering in `03_filter_data.py`
**File:** `scripts/03_filter_data.py`
**Symptom:** Test — `ground truth item never appears in user train history` — FAIL. Same item appearing in both train and test for some users.
**Cause:** Users who reviewed the same item twice at the same timestamp had that item land in both splits. Deduplication was placed after filtering, meaning interaction counts were computed on duplicate rows.
```python
# wrong — dedup after filter, counts are inflated
df = df[df['user_id'].isin(user_counts >= 5)]
df = df.drop_duplicates(subset=['user_id', 'parent_asin'], keep='last')

# correct — dedup first, then count
df = df.drop_duplicates(subset=['user_id', 'parent_asin'], keep='last')
df = df[df['user_id'].isin(user_counts >= 5)]
```
**Fix:** Move `drop_duplicates` to step 0, before any filtering or sorting.

## Bug 2 – Failing test for valid embeddings

**File:** `src/embeddings/user_embedding.cpp`

**Symptom:** User embedding norm was 0.659 instead of 1.0. All user embeddings failed unit-normalisation check.

**Cause:** `l2_normalize` took `embedding_t` by value instead of by reference. The function normalised a local copy and discarded it — the original vector in the caller was never modified.

```cpp
// WRONG — normalises a copy, original unchanged
static void l2_normalize(embedding_t user_vector)

// CORRECT — normalises in place
static void l2_normalize(embedding_t& user_vector)
```

**Why it compiled silently:** C++ pass-by-value is valid syntax. The compiler has no way to know the intent was in-place modification. No warning is generated.

**Detected by:** Test 7a — unit-normalisation check on sample user vector. Diagnosed by printing the norm value (0.659) which ruled out zero vector and NaN, pointing to normalisation running on the wrong object.

**Fix:** Add `&` to the parameter in the function signature.