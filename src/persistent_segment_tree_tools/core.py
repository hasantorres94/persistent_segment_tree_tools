from __future__ import annotations

from typing import Callable, List, Optional


class _Node:
    __slots__ = ("left", "right", "value")

    def __init__(
        self,
        left: Optional["_Node"],
        right: Optional["_Node"],
        value: int,
    ) -> None:
        self.left = left
        self.right = right
        self.value = value


class PersistentSegmentTree:
    """A persistent segment tree for integer arrays.

    A persistent segment tree represents an array under an associative,
    commutative operation. The key property is structural sharing: every
    ``update`` returns a new ``PersistentSegmentTree`` object whose root
    shares every node that the update did not touch with the previous
    version. An update touches O(log n) nodes, so a version costs O(log n)
    time and space. A range query on any version costs O(log n) time.

    Design choices and their reasons:

    1. The operation and identity are fixed at construction time and passed
       as plain callables. This is more flexible than an enum and avoids
       any string-comparison dispatch on the query hot path.

    2. We build the initial tree over the *half-open* range [0, n). Using
       half-open intervals everywhere makes the ``end`` argument of a query
       behave like Python slicing: ``query(1, 3)`` sums elements at indices
       1 and 2. Mixing conventions is a common source of off-by-one errors.

    3. The identity must be the true identity of the operation:
       ``combine(identity, x) == x``. The build step relies on this when a
       node covers an empty sub-range after a point update falls outside
       it; the identity is stored there so that queries still compose
       correctly.

    Edge case: if a point update at index ``i`` is applied to a version and
    ``i`` is outside ``[0, n)``, the update silently returns a structurally
    shared copy of the same root without raising. This is deliberate: it
    matches the semantics of a versioned map where an out-of-bounds key is
    simply not present, and it keeps the update path allocation-free for
    that case. Range queries clip their bounds to ``[0, n)``, so an
    out-of-bounds update never corrupts a query result.
    """

    def __init__(
        self,
        data: List[int],
        combine: Callable[[int, int], int],
        identity: int,
    ) -> None:
        if not isinstance(data, list):
            raise TypeError("data must be a list")
        if not callable(combine):
            raise TypeError("combine must be callable")
        self._combine = combine
        self._identity = identity
        self._n = len(data)
        if self._n == 0:
            root: Optional[_Node] = None
        else:
            root = self._build(data, 0, self._n)
        self._root = root

    def _build(self, data: List[int], lo: int, hi: int) -> _Node:
        if hi - lo == 1:
            return _Node(None, None, data[lo])
        mid = (lo + hi) // 2
        left = self._build(data, lo, mid)
        right = self._build(data, mid, hi)
        return _Node(left, right, self._combine(left.value, right.value))

    def update(self, index: int, value: int) -> "PersistentSegmentTree":
        """Return a new version with ``data[index]`` replaced by ``value``.

        If ``index`` is outside ``[0, n)``, the returned tree shares the
        same root as ``self``; no allocation occurs.
        """
        if index < 0 or index >= self._n:
            return self
        new = PersistentSegmentTree.__new__(PersistentSegmentTree)
        new._combine = self._combine
        new._identity = self._identity
        new._n = self._n
        new._root = self._update(self._root, 0, self._n, index, value)
        return new

    def _update(
        self,
        node: Optional[_Node],
        lo: int,
        hi: int,
        index: int,
        value: int,
    ) -> _Node:
        if hi - lo == 1:
            return _Node(None, None, value)
        mid = (lo + hi) // 2
        # We must read node.left / node.right even though the recursion
        # path only descends into one side: the untouched child is shared
        # by reference into the new node, which is what gives the update
        # its O(log n) space bound.
        if index < mid:
            assert node is not None
            left = self._update(node.left, lo, mid, index, value)
            right = node.right
        else:
            assert node is not None
            left = node.left
            right = self._update(node.right, mid, hi, index, value)
        return _Node(left, right, self._combine(left.value, right.value))

    def query(self, start: int, end: int) -> int:
        """Aggregate over ``data[start:end]`` (half-open).

        Bounds are clipped to ``[0, n)``. If ``start >= end`` after
        clipping, the identity is returned.
        """
        if self._n == 0:
            return self._identity
        start = max(0, start)
        end = min(self._n, end)
        if start >= end:
            return self._identity
        return self._query(self._root, 0, self._n, start, end)

    def _query(
        self,
        node: Optional[_Node],
        lo: int,
        hi: int,
        start: int,
        end: int,
    ) -> int:
        if start >= hi or end <= lo:
            return self._identity
        if start <= lo and hi <= end:
            assert node is not None
            return node.value
        mid = (lo + hi) // 2
        assert node is not None
        return self._combine(
            self._query(node.left, lo, mid, start, end),
            self._query(node.right, mid, hi, start, end),
        )

    @property
    def length(self) -> int:
        return self._n
