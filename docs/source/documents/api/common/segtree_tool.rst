Segment Tree 
========================================

xxxxxx.

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class::
  xuance.common.segtree_tool.SegmentTree(capacity, operation, neutral_element)

  :param capacity: xxxxxx.
  :type capacity: xxxxxx
  :param operation: xxxxxx.
  :type operation: xxxxxx
  :param neutral_element: xxxxxx.
  :type neutral_element: xxxxxx

.. py:function::
  xuance.common.segtree_tool.SegmentTree._reduce_helper(start, end, node, node_start, node_end)

  xxxxxx.

  :param start: xxxxxx.
  :type start: xxxxxx
  :param end: xxxxxx.
  :type end: xxxxxx
  :param node: xxxxxx.
  :type node: xxxxxx
  :param node_start: xxxxxx.
  :type node_start: xxxxxx
  :param node_end: xxxxxx.
  :type node_end: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.common.segtree_tool.SegmentTree.reduce(start, end)

  xxxxxx.

  :param start: xxxxxx.
  :type start: xxxxxx
  :param end: xxxxxx.
  :type end: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.common.segtree_tool.SegmentTree.__setitem__(idx, val)

  xxxxxx.

  :param idx: xxxxxx.
  :type idx: xxxxxx
  :param val: xxxxxx.
  :type val: xxxxxx

.. py:function::
  xuance.common.segtree_tool.SegmentTree.__getitem__(idx)

  xxxxxx.

  :param idx: xxxxxx.
  :type idx: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.common.segtree_tool.SumSegmentTree(capacity)

  :param idx: xxxxxx.
  :type idx: xxxxxx

.. py:function::
  xuance.common.segtree_tool.SumSegmentTree.sum(start, end)

  xxxxxx.

  :param start: xxxxxx.
  :type start: xxxxxx
  :param end: xxxxxx.
  :type end: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

  xuance.common.segtree_tool.SumSegmentTree.find_prefixsum_idx(prefixsum)

  xxxxxx.

  :param prefixsum: xxxxxx.
  :type prefixsum: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.common.segtree_tool.MinSegmentTree(capacity)

  :param capacity: xxxxxx.
  :type capacity: xxxxxx

.. py:function::
  xuance.common.segtree_tool.MinSegmentTree.min(start， end)

  xxxxxx.

  :param start: xxxxxx.
  :type start: xxxxxx
  :param end: xxxxxx.
  :type end: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. raw:: html

    <br><hr>

**TensorFlow:**

.. raw:: html

    <br><hr>

**MindSpore:**

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

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

  .. group-tab:: TensorFlow

    .. code-block:: python


  .. group-tab:: MindSpore

    .. code-block:: python




