import operator


class SegmentTree(object):
    """
    A data structure for efficient range queries and point updates using a binary tree representation.

    Attributes:
        _capacity (int): The number of elements in the tree, must be a power of 2.
        _value (list): Internal array to store the tree nodes.
        _operation (Callable): A binary operation (e.g., addition, min, max) for range queries.
        _neutral_element (Any): The neutral element for the operation (e.g., 0 for addition, infinity for min).

    Methods:
        __init__(capacity, operation, neutral_element):
            Initializes the segment tree with a specified capacity, operation, and neutral element.
        reduce(start=0, end=None):
            Computes the result of the operation over a range [start, end).
        __setitem__(idx, val):
            Updates the value at a specific index and propagates changes.
        __getitem__(idx):
            Retrieves the value at a specific index.
    """
    def __init__(self, capacity, operation, neutral_element):
        """
        Initialize a SegmentTree.

        Args:
            capacity (int): Number of elements in the tree, must be a power of 2.
            operation (Callable): Binary operation (e.g., lambda x, y: x + y) for combining elements.
            neutral_element (Any): Neutral element for the operation (e.g., 0 for addition, float('inf') for min).

        Raises:
            AssertionError: If capacity is not positive or not a power of 2.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        """
        Recursively computes the result of the operation over a range.

        Args:
            start (int): Start of the query range (inclusive).
            end (int): End of the query range (inclusive).
            node (int): Current node index in the tree.
            node_start (int): Start of the range represented by the current node.
            node_end (int): End of the range represented by the current node.

        Returns:
            Any: The result of the operation over the specified range.
        """
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
        """
        Computes the result of the operation over a range [start, end).

        Args:
            start (int, optional): Start of the range (default is 0).
            end (int, optional): End of the range (default is the tree's capacity).

        Returns:
            Any: The result of the operation over the specified range.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        """
        Updates the value at a specific index and propagates the changes.

        Args:
            idx (int): Index to update.
            val (Any): New value to set.
        """
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
        """
        Retrieves the value at a specific index.

        Args:
            idx (int): Index to query.

        Returns:
            Any: The value at the specified index.

        Raises:
            AssertionError: If the index is out of range.
        """
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    """
    A specialized implementation of a Segment Tree for summation queries and prefix-sum searches.

    Attributes:
        _capacity (int): The size of the underlying array, must be a power of 2.
        _value (list): The tree representation of the segment tree, storing intermediate sums.
        _operation (callable): The operation to be performed (addition in this case).
        _neutral_element (float): The neutral element for the operation (0.0 for addition).
    """

    def __init__(self, capacity):
        """
        Initialize a SumSegmentTree with the given capacity.

        Parameters:
            capacity (int): The capacity of the segment tree, must be a power of 2.
        """
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """
        Compute the sum of elements in the range [start, end).
        Returns arr[start] + ... + arr[end]

        Parameters:
            start (int): The starting index of the range (inclusive).
            end (int, optional): The ending index of the range (exclusive). Defaults to the full range.

        Returns:
            float: The sum of elements in the specified range.
        """
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """
        Find the index of the smallest prefix sum greater than or equal to the given value.

        Parameters:
            prefixsum (float): The target prefix sum.

        Returns:
            int: The index corresponding to the target prefix sum.

        Raises:
            AssertionError: If prefixsum is not within the valid range [0, total sum].
        """
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
    """
    A specialized implementation of a Segment Tree for range minimum queries.

    Attributes:
        _capacity (int): The size of the underlying array, must be a power of 2.
        _value (list): The tree representation of the segment tree, storing intermediate minimums.
        _operation (callable): The operation to be performed (minimum in this case).
        _neutral_element (float): The neutral element for the operation (infinity for minimum).
    """
    def __init__(self, capacity):
        """
        Initialize a MinSegmentTree with the given capacity.

        Parameters:
            capacity (int): The capacity of the segment tree, must be a power of 2.
        """
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """
        Compute the minimum value in the range [start, end).
        Returns min(arr[start], ...,  arr[end])

        Parameters:
            start (int): The starting index of the range (inclusive).
            end (int, optional): The ending index of the range (exclusive). Defaults to the full range.

        Returns:
            float: The minimum value in the specified range.
        """
        return super(MinSegmentTree, self).reduce(start, end)
