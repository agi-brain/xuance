Segment Tree 
========================================

In this module, we define two classes: SegmentTree and its subclasses SumSegmentTree, and MinSegmentTree. 
These classes are used for efficiently computing range queries (such as sum and minimum) over a dynamic array. 
They are often employed in algorithms that involve prioritized replay in DRL, where efficient computation of cumulative sums or minimum values is crucial.

.. py:class::
    xuance.common.segtree_tool.SegmentTree(capacity, operation, neutral_element)

    The SegmentTree class provides a basic implementation for range queries with support for user-defined associative operations. 
    It is used as a building block for more complex algorithms, 
    such as those involving priority queues or efficient range queries in various applications.

    :param capacity: The number of elements in the array represented by the segment tree. It must be a positive power of 2.
    :type capacity: float
    :param operation: A binary associative function that defines how values are combined.
    :param neutral_element: The neutral element for the specified operation.

.. py:function::
    xuance.common.segtree_tool.SegmentTree._reduce_helper(start, end, node, node_start, node_end)

    This function traverses the segment tree and combines values according to the specified binary associative operation within the given range.

    :param start: The starting index of the query range in the original array.
    :type start: int
    :param end: The ending index of the query range in the original array.
    :type end: int
    :param node: The index of the current node in the segment tree.
    :type node: int
    :param node_start: The starting index of the range represented by the current node in the original array.
    :type node_start: int
    :param node_end: The ending index of the range represented by the current node in the original array.
    :type node_end: int
    :return: the result of the range query within the specified range.

.. py:function::
    xuance.common.segtree_tool.SegmentTree.reduce(start=0, end=None)

    A public method that allows users to perform a range query on the segment tree. 
    It is essentially a wrapper around the _reduce_helper method, providing a more user-friendly interface for specifying the query range.

    :param start: The starting index of the query range in the original array. Defaults to 0 if not provided.
    :type start: int
    :param end: The ending index of the query range in the original array. If not provided, it defaults to the entire capacity of the segment tree.
    :type end: int
    :return: the result of a range query on the segment tree.

.. py:function::
    xuance.common.segtree_tool.SegmentTree.__setitem__(idx, val)

    Set the value at a specific index in the original array and update the segment tree accordingly.

    :param idx: The index at which the value is to be set in the original array.
    :type idx: int
    :param val: the specified value to be set.
    :type val: float

.. py:function::
    xuance.common.segtree_tool.SegmentTree.__getitem__(idx)

    Retrieve the value at a specific index in the original array represented by the segment tree.

    :param idx: The index for which the value is to be retrieved from the original array.
    :type idx: int
    :return: the value at the specified index in the original array.
    :rtype: float

.. py:class::
    xuance.common.segtree_tool.SumSegmentTree(capacity)

    This class is designed specifically for handling sum operations on a range of elements. 
    It inherits from the SegmentTree class and extends its functionality to support sum queries efficiently.

    :param capacity: the number of elements in the original array that the segment tree is designed to represent.
    :type capacity: int

.. py:function::
    xuance.common.segtree_tool.SumSegmentTree.sum(start=0, end=None)

    Be responsible for calculating the sum of elements in a specified range of the original array represented by the segment tree.

    :param start: The starting index of the range for the sum query (default is 0).
    :type start: int32
    :param end: The ending index of the range for the sum query (default is None, which means the last index).
    :type end: int
    :return: The result of the reduce operation is returned by the sum method, representing the sum of elements in the specified range [start, end].

.. py:function::
    xuance.common.segtree_tool.SumSegmentTree.find_prefixsum_idx(prefixsum)

    Find the index of the element in the original array such that the sum of all preceding elements is less than or equal to a given prefixsum.

    :param prefixsum: the cumulative sum of elements in an array up to a certain index.
    :type prefixsum: float
    :return: the final index by subtracting self._capacity from idx.
    :rtype: int

.. py:class::
    xuance.common.segtree_tool.MinSegmentTree(capacity)

    The MinSegmentTree class is designed to support range minimum queries over a sequence of values.
    It inherits from SegmentTree and provides a method min to find the minimum value within a specified range in the original array.

    :param capacity: The number of elements in the original array.
    :type capacity: int

.. py:function::
    xuance.common.segtree_tool.MinSegmentTree.min(start=0, end=None)

    Returns min(arr[start], ...,  arr[end]).

    :param start: default is 0.
    :type start: int
    :param end: default is None.
    :type end: int
    :return: min(arr[start], ...,  arr[end]).

.. raw:: html

    <br><hr>


Source Code
-----------------

.. code-block:: python

    import operator


    class SegmentTree(object):
        def __init__(self, capacity, operation, neutral_element):
            assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
            self._capacity = capacity
            self._value = [neutral_element for _ in range(2 * capacity)]
            self._operation = operation

        def _reduce_helper(self, start, end, node, node_start, node_end):
            if start == node_start and end == node_end:
                return self._value[node]
            mid = (node_start + node_end) // 2
            if end <= mid:
                return self._reduce_helper(start, end, 2 * node, node_start, mid)
            else:
                if mid + 1 <= start:
                    return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
                else:
                    return self._operation(
                        self._reduce_helper(start, mid, 2 * node, node_start, mid),
                        self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                    )

        def reduce(self, start=0, end=None):
            if end is None:
                end = self._capacity
            if end < 0:
                end += self._capacity
            end -= 1
            return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

        def __setitem__(self, idx, val):
            # index of the leaf
            idx += self._capacity
            self._value[idx] = val
            idx //= 2
            while idx >= 1:
                self._value[idx] = self._operation(
                    self._value[2 * idx],
                    self._value[2 * idx + 1]
                )
                idx //= 2

        def __getitem__(self, idx):
            assert 0 <= idx < self._capacity
            return self._value[self._capacity + idx]


    class SumSegmentTree(SegmentTree):
        def __init__(self, capacity):
            super(SumSegmentTree, self).__init__(
                capacity=capacity,
                operation=operator.add,
                neutral_element=0.0
            )

        def sum(self, start=0, end=None):
            """Returns arr[start] + ... + arr[end]"""
            return super(SumSegmentTree, self).reduce(start, end)

        def find_prefixsum_idx(self, prefixsum):
            assert 0 <= prefixsum <= self.sum() + 1e-5
            idx = 1
            while idx < self._capacity:  # while non-leaf
                if self._value[2 * idx] > prefixsum:
                    idx = 2 * idx
                else:
                    prefixsum -= self._value[2 * idx]
                    idx = 2 * idx + 1
            return idx - self._capacity


    class MinSegmentTree(SegmentTree):
        def __init__(self, capacity):
            super(MinSegmentTree, self).__init__(
                capacity=capacity,
                operation=min,
                neutral_element=float('inf')
            )

        def min(self, start=0, end=None):
            """Returns min(arr[start], ...,  arr[end])"""

            return super(MinSegmentTree, self).reduce(start, end)
