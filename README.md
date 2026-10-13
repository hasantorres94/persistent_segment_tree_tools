# persistent_segment_tree_tools

A persistent segment tree over an integer array that supports O(log n) range queries and O(log n) point updates, with every update returning a new version whose prior state remains queryable.

```python
from persistent_segment_tree_tools import PersistentSegmentTree

v0 = PersistentSegmentTree([3, 1, 4, 1, 5], lambda a, b: a + b, 0)
print(v0.query(0, 5))   # 14

v1 = v0.update(2, 10)
print(v0.query(0, 5))   # 14  (unchanged)
print(v1.query(0, 5))   # 20
```

## Why this exists

When you need to answer range queries (sums, maxima, minima) against the state of an array *as it was at a specific point in time*, a mutable segment tree is not enough — updating it destroys the old state, and copying the whole array per update costs O(n). A persistent segment tree shares structure between versions, so each update allocates only the O(log n) nodes on the path from the root to the updated leaf. This trades higher constant factors and O(n log n) total memory for the ability to query any historical version in O(log n).

The operation and identity are passed at construction as plain callables, so the same tree supports sum, max, min, or any associative commutative binary operation without a dispatch lookup on the query path.

## Edge cases

- An update with an index outside `[0, n)` returns the same tree object unchanged; no error is raised and no allocation occurs. Range queries clip their bounds to `[0, n)`, so an out-of-bounds update never affects a later query.
- An empty array yields a tree whose `query` always returns the identity and whose `length` is 0.
- The identity must be the true neutral element of your operation: `combine(identity, x) == x`. For `max` over non-negative data, `0` works; over arbitrary integers, use a sufficiently negative value.
- Ranges are half-open: `query(1, 3)` covers indices 1 and 2, matching Python slicing.
