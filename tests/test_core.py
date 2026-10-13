import unittest

from persistent_segment_tree_tools import PersistentSegmentTree


class TestBuildAndQuery(unittest.TestCase):
    def setUp(self):
        self.data = [3, 1, 4, 1, 5, 9, 2, 6]
        self.tree = PersistentSegmentTree(
            self.data, lambda a, b: a + b, 0
        )

    def test_full_range_sum(self):
        self.assertEqual(self.tree.query(0, 8), 31)

    def test_prefix_sum(self):
        self.assertEqual(self.tree.query(0, 3), 8)

    def test_suffix_sum(self):
        self.assertEqual(self.tree.query(5, 8), 17)

    def test_single_element(self):
        self.assertEqual(self.tree.query(4, 5), 5)

    def test_length(self):
        self.assertEqual(self.tree.length, 8)


class TestEdgeBounds(unittest.TestCase):
    def setUp(self):
        self.tree = PersistentSegmentTree(
            [3, 1, 4, 1, 5], lambda a, b: a + b, 0
        )

    def test_negative_start_clipped(self):
        self.assertEqual(self.tree.query(-3, 2), 4)

    def test_end_beyond_length_clipped(self):
        self.assertEqual(self.tree.query(2, 100), 10)

    def test_empty_range_returns_identity(self):
        self.assertEqual(self.tree.query(3, 3), 0)

    def test_inverted_range_returns_identity(self):
        self.assertEqual(self.tree.query(4, 1), 0)


class TestEmptyArray(unittest.TestCase):
    def test_query_returns_identity(self):
        tree = PersistentSegmentTree([], lambda a, b: a + b, 0)
        self.assertEqual(tree.query(0, 0), 0)
        self.assertEqual(tree.query(-5, 10), 0)

    def test_length_is_zero(self):
        tree = PersistentSegmentTree([], lambda a, b: a + b, 0)
        self.assertEqual(tree.length, 0)


class TestSingleElement(unittest.TestCase):
    def test_query(self):
        tree = PersistentSegmentTree([42], lambda a, b: a + b, 0)
        self.assertEqual(tree.query(0, 1), 42)
        self.assertEqual(tree.query(0, 0), 0)


class TestUpdate(unittest.TestCase):
    def setUp(self):
        self.data = [1, 2, 3, 4, 5]
        self.v0 = PersistentSegmentTree(self.data, lambda a, b: a + b, 0)

    def test_update_creates_new_version(self):
        v1 = self.v0.update(2, 10)
        self.assertIsNot(self.v0, v1)
        self.assertEqual(self.v0.query(0, 5), 15)
        self.assertEqual(v1.query(0, 5), 22)

    def test_old_version_unchanged_after_multiple_updates(self):
        v1 = self.v0.update(0, 100)
        v2 = v1.update(4, 100)
        self.assertEqual(self.v0.query(0, 5), 15)
        self.assertEqual(v1.query(0, 5), 114)
        self.assertEqual(v2.query(0, 5), 209)

    def test_update_out_of_bounds_returns_same_reference(self):
        v1 = self.v0.update(-1, 99)
        v2 = self.v0.update(5, 99)
        self.assertIs(self.v0, v1)
        self.assertIs(self.v0, v2)

    def test_update_then_query_affected_range(self):
        v1 = self.v0.update(3, 40)
        self.assertEqual(v1.query(0, 3), 6)
        self.assertEqual(v1.query(3, 4), 40)
        self.assertEqual(v1.query(0, 5), 51)


class TestMaxOperation(unittest.TestCase):
    def test_max_query_and_update(self):
        data = [1, 5, 2, 8, 3]
        v0 = PersistentSegmentTree(data, max, 0)
        self.assertEqual(v0.query(0, 5), 8)
        self.assertEqual(v0.query(0, 3), 5)
        v1 = v0.update(0, 10)
        self.assertEqual(v1.query(0, 5), 10)
        self.assertEqual(v0.query(0, 5), 8)


class TestIdentityInvariant(unittest.TestCase):
    def test_identity_must_be_neutral(self):
        data = [3, 7, 2]
        # For max, identity must be -inf-like. Using 0 only works if all
        # values are non-negative. Here we use a sufficiently negative
        # value so the invariant combine(identity, x) == x holds.
        tree = PersistentSegmentTree(data, max, -10**9)
        self.assertEqual(tree.query(0, 3), 7)


class TestTypeErrors(unittest.TestCase):
    def test_non_list_data(self):
        with self.assertRaises(TypeError):
            PersistentSegmentTree("abc", lambda a, b: a + b, 0)

    def test_non_callable_combine(self):
        with self.assertRaises(TypeError):
            PersistentSegmentTree([1, 2], 123, 0)


if __name__ == "__main__":
    unittest.main()
