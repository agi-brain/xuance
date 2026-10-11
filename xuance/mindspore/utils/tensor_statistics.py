import mindspore as ms
from mindspore import ops, Tensor
import torch.distributed as dist
from xuance.common import Union, Sequence


class TensorRunningMeanStd:
    """
    Maintains a running mean and standard deviation.

    Attributes:
        shape (Union[Sequence[int], dict]): Shape of the input data.
        epsilon (float): Small value to prevent division by zero.
        device (torch.device): Device to use for computation.
        distributed (bool): Whether to use distributed computation.
    """
    def __init__(self,
                 shape: Union[Sequence[int], dict],
                 epsilon=1e-4,
                 distributed: bool = False):
        self.shape = shape
        self.epsilon = epsilon
        self.distributed = distributed

        if isinstance(shape, dict):
            self.mean = {k: ops.zeros(v, dtype=ms.float32) for k, v in shape.items()}
            self.var = {k: ops.ones(v, dtype=ms.float32) for k, v in shape.items()}
            self.count = {k: Tensor(epsilon, dtype=ms.float32) for k in shape.keys()}
        else:
            self.mean = ops.zeros(shape, dtype=ms.float32)
            self.var = ops.ones(shape, dtype=ms.float32)
            self.count = Tensor(epsilon, dtype=ms.float32)

    @property
    def std(self):
        """
        Compute the standard deviation.

        Returns:
            Union[dict, ndarray]: The standard deviation of the running statistics.
        """
        if isinstance(self.shape, dict):
            return {k: ops.sqrt(self.var[k]) for k in self.shape.keys()}
        else:
            return ops.sqrt(self.var)

    def _sync_distributed_moments(self, batch_mean, batch_var, batch_count):
        if not self.distributed:
            return batch_mean, batch_var, batch_count

        world_size = dist.get_world_size()
        dist.all_reduce(batch_mean, op=dist.ReduceOp.SUM)
        dist.all_reduce(batch_var, op=dist.ReduceOp.SUM)
        dist.all_reduce(batch_count, op=dist.ReduceOp.SUM)
        batch_mean = batch_mean / world_size
        batch_var = batch_var / world_size
        return batch_mean, batch_var, batch_count

    def update(self, x):
        """
        Update the running mean and standard deviation with new data.

        Parameters:
            x (Union[dict, ndarray]): New data to update the statistics.
        """
        if isinstance(x, dict):
            batch_means, batch_vars, batch_counts = {}, {}, {}
            for key in self.shape.keys():
                b_mean = ops.mean(x[key], axis=0)
                b_var = ops.var(x[key], axis=0, unbiased=False)
                b_count = Tensor(x[key].shape[0], dtype=ms.float32)

                b_mean, b_var, b_count = self._sync_distributed_moments(b_mean, b_var, b_count)
                batch_means[key], batch_vars[key], batch_counts[key] = b_mean, b_var, b_count

            self.update_from_moments(batch_means, batch_vars, batch_counts)
        else:
            batch_mean = ops.mean(x, axis=0)
            batch_var = ops.var(x, axis=0, unbiased=False)
            batch_count = Tensor(x.shape[0], dtype=ms.float32)

            batch_mean, batch_var, batch_count = self._sync_distributed_moments(batch_mean, batch_var, batch_count)
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """
        Update the running mean, variance, and count using new statistics.

        This method updates the current statistics by combining them with
        batch-level statistics, supporting both dictionary and array inputs.

        Parameters:
            batch_mean (Union[dict, ndarray]): Mean of the new batch.
            batch_var (Union[dict, ndarray]): Variance of the new batch.
            batch_count (Union[dict, int]): Number of samples in the new batch.

        Updates:
            self.mean (Union[dict, ndarray]): Updated running mean.
            self.var (Union[dict, ndarray]): Updated running variance.
            self.count (Union[dict, float]): Updated sample count.
        """
        if isinstance(batch_mean, dict):
            for key in self.shape:
                delta = batch_mean[key] - self.mean[key]
                tot_count = self.count[key] + batch_count[key]
                new_mean = self.mean[key] + delta * batch_count[key] / tot_count
                m_a = self.var[key] * (self.count[key])
                m_b = batch_var[key] * (batch_count[key])
                M2 = m_a + m_b + ops.square(delta) * self.count[key] * batch_count[key] / tot_count
                new_var = M2 / tot_count

                self.mean[key] = new_mean
                self.var[key] = new_var
                self.count[key] = tot_count
        else:
            delta = batch_mean - self.mean
            tot_count = self.count + batch_count
            new_mean = self.mean + delta * batch_count / tot_count
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + ops.square(delta) * self.count * batch_count / tot_count
            new_var = M2 / tot_count

            self.mean = new_mean
            self.var = new_var
            self.count = tot_count
