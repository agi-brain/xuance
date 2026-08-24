import tensorflow as tf
from xuance.common import Union, Sequence


class TensorRunningMeanStd:
    """
    Maintains a running mean and standard deviation.

    Args:
        shape (Union[Sequence[int], dict]): Shape of the input data.
        epsilon (float): Small value to prevent division by zero.
        device (str): Device to use for computation.
        distributed (bool): Whether to use distributed computation.
    """

    def __init__(self,
                 shape: Union[Sequence[int], dict],
                 epsilon=1e-4,
                 device: str = "/CPU:0",
                 distributed: bool = False):
        self.shape = shape
        self.epsilon = epsilon
        self.device = device
        self.distributed = distributed

        with tf.device(self.device):
            if isinstance(shape, dict):
                self.mean = {k: tf.Variable(tf.zeros(v, dtype=tf.float32), trainable=False) for k, v in shape.items()}
                self.var = {k: tf.Variable(tf.ones(v, dtype=tf.float32), trainable=False) for k, v in shape.items()}
                self.count = {k: tf.Variable(epsilon, dtype=tf.float32, trainable=False) for k in shape.keys()}
            else:
                self.mean = tf.Variable(tf.zeros(shape, dtype=tf.float32), trainable=False)
                self.var = tf.Variable(tf.ones(shape, dtype=tf.float32), trainable=False)
                self.count = tf.Variable(epsilon, dtype=tf.float32, trainable=False)

    @property
    def std(self):
        """
        Compute the standard deviation.

        Returns:
            Union[dict, ndarray]: The standard deviation of the running statistics.
        """
        if isinstance(self.shape, dict):
            return {k: tf.sqrt(self.var[k]) for k in self.shape.keys()}
        else:
            return tf.sqrt(self.var)

    def _sync_distributed_moments(self, batch_mean, batch_var, batch_count):
        """
        Synchronize batch moments across distributed replicas.

        Notes:
            This method is intended to be called inside the replica context
            created by ``tf.distribute.Strategy.run``. This preserves the
            behavior of the PyTorch implementation, which averages batch means
            and variances across workers and sums batch counts.
        """
        if not self.distributed:
            return batch_mean, batch_var, batch_count

        replica_context = tf.distribute.get_replica_context()
        if replica_context is None:
            raise RuntimeError(
                "distributed=True requires TensorRunningMeanStd.update() "
                "to be called inside tf.distribute.Strategy.run()."
            )

        batch_mean = replica_context.all_reduce(tf.distribute.ReduceOp.SUM, batch_mean)
        batch_var = replica_context.all_reduce(tf.distribute.ReduceOp.SUM, batch_var)
        batch_count = replica_context.all_reduce(tf.distribute.ReduceOp.SUM, batch_count)

        world_size = tf.cast(replica_context.num_replicas_in_sync, batch_mean.dtype)
        batch_mean = batch_mean / world_size
        batch_var = batch_var / world_size
        return batch_mean, batch_var, batch_count

    def update(self, x):
        """
        Update the running mean and standard deviation with new data.

        Args:
            x (Union[dict, tf.Tensor]): New data used to update the statistics.
        """
        if isinstance(x, dict):
            batch_means, batch_vars, batch_counts = {}, {}, {}

            for key in self.shape.keys():
                x_key = tf.convert_to_tensor(x[key], dtype=tf.float32)

                b_mean = tf.reduce_mean(x_key, axis=0)
                b_var = tf.math.reduce_variance(x_key, axis=0)
                b_count = tf.cast(tf.shape(x_key)[0], tf.float32)

                b_mean, b_var, b_count = self._sync_distributed_moments(b_mean, b_var, b_count)
                batch_means[key] = b_mean
                batch_vars[key] = b_var
                batch_counts[key] = b_count

            self.update_from_moments(batch_means, batch_vars, batch_counts)
        else:
            x = tf.convert_to_tensor(x, dtype=tf.float32)

            batch_mean = tf.reduce_mean(x, axis=0)
            batch_var = tf.math.reduce_variance(x, axis=0)
            batch_count = tf.cast(tf.shape(x)[0], tf.float32)

            batch_mean, batch_var, batch_count = self._sync_distributed_moments(batch_mean, batch_var, batch_count)
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """
        Update the running mean, variance, and count using new statistics.

        This method combines the existing running statistics with batch-level
        statistics and supports both dictionary and Tensor inputs.

        Args:
            batch_mean (Union[dict, tf.Tensor]): Mean of the new batch.
            batch_var (Union[dict, tf.Tensor]): Variance of the new batch.
            batch_count (Union[dict, tf.Tensor]): Number of samples in the batch.
        """
        if isinstance(batch_mean, dict):
            for key in self.shape:
                delta = batch_mean[key] - self.mean[key]
                total_count = self.count[key] + batch_count[key]

                new_mean = self.mean[key] + delta * batch_count[key] / total_count

                m_a = self.var[key] * self.count[key]
                m_b = batch_var[key] * batch_count[key]
                m2 = m_a + m_b + tf.square(delta) * self.count[key] * batch_count[key] / total_count
                new_var = m2 / total_count

                self.mean[key].assign(new_mean)
                self.var[key].assign(new_var)
                self.count[key].assign(total_count)
        else:
            delta = batch_mean - self.mean
            total_count = self.count + batch_count

            new_mean = self.mean + delta * batch_count / total_count

            m_a = self.var * self.count
            m_b = batch_var * batch_count
            m2 = m_a + m_b + tf.square(delta) * self.count * batch_count / total_count
            new_var = m2 / total_count

            self.mean.assign(new_mean)
            self.var.assign(new_var)
            self.count.assign(total_count)
