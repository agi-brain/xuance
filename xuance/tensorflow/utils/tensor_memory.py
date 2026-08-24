from gymnasium import Space
from abc import ABC, abstractmethod
from typing import Optional, Union
from xuance.environment.utils import space2shape

import tensorflow as tf
from tensorflow import Tensor


def create_tensor_memory(
        shape: Optional[Union[tuple, dict]],
        n_envs: int,
        n_size: int,
        dtype: Optional[tf.dtypes.DType] = tf.float32,
        device: str = "/CPU:0",
):
    """Creates TensorFlow variable memory for buffer data.

    Args:
        shape: Data shape.
        n_envs: Number of parallel environments.
        n_size: Length of the data sequence for each environment.
        dtype: TensorFlow data type.
        device: Device on which the memory is allocated.

    Returns:
        A TensorFlow variable or a dictionary of TensorFlow variables used to
        store buffer data.
    """
    if shape is None:
        return None

    with tf.device(device):
        if isinstance(shape, dict):
            memory = {}
            for key, value in shape.items():
                if value is None:  # save an object type
                    memory[key] = tf.Variable(tf.zeros([n_envs, n_size], dtype=dtype), trainable=False)
                else:
                    memory[key] = tf.Variable(tf.zeros([n_envs, n_size] + list(value), dtype=dtype), trainable=False)
            return memory

        if isinstance(shape, tuple):
            return tf.Variable(tf.zeros([n_envs, n_size] + list(shape), dtype=dtype), trainable=False)

    raise NotImplementedError


def store_tensor_element(
        data: Optional[Union[Tensor, dict]],
        memory: Union[Tensor, dict],
        ptr: int
):
    """
    Insert a step of data into current memory.

    Args:
        data: target data that to be stored.
        memory: the memory where data will be stored.
        ptr: pointer to the location for the data.
    """
    if data is None:
        return

    if isinstance(data, dict):
        for key, value in data.items():
            memory[key][:, ptr].assign(tf.cast(value, memory[key].dtype))
    else:
        memory[:, ptr].assign(tf.cast(data, memory.dtype))


def sample_batch(memory: Optional[Union[Tensor, dict]],
                 index: Optional[Union[Tensor, tuple]]):
    """
    Sample a batch of data from the selected memory.

    Args:
        memory: memory that contains experience data.
        index: pointer to the location for the selected data.

    Returns:
        A batch of data.
    """
    if memory is None:
        return None

    if isinstance(memory, dict):
        return {key: sample_batch(value, index) for key, value in memory.items()}

    if isinstance(index, tuple):
        env_ids, step_ids = index
        gather_indices = tf.stack([tf.cast(env_ids, tf.int32), tf.cast(step_ids, tf.int32)], axis=-1)
        return tf.gather_nd(memory, gather_indices)

    return tf.gather(memory, index)


def discount_cumsum_tensor(x: Tensor, discount: float) -> Tensor:
    """Computes discounted cumulative sums along a 1-D tensor."""
    x = tf.convert_to_tensor(x)
    discount = tf.cast(discount, x.dtype)

    reversed_x = tf.reverse(x, axis=[0])

    def scan_fn(acc, current):
        return current + discount * acc

    reversed_result = tf.scan(
        scan_fn,
        reversed_x,
        initializer=tf.zeros([], dtype=x.dtype),
    )
    return tf.reverse(reversed_result, axis=[0])


def _sample_std(x: Tensor) -> Tensor:
    """Returns standard deviation matching torch.std(..., correction=1)."""
    x = tf.convert_to_tensor(x)
    n = tf.cast(tf.size(x), x.dtype)
    mean = tf.reduce_mean(x)
    variance = tf.reduce_sum(tf.square(x - mean)) / (n - tf.cast(1.0, x.dtype))
    return tf.sqrt(variance)


class TensorBuffer(ABC):
    """
    Basic buffer single-agent DRL algorithms.

    Args:
        observation_space: the space for observation data.
        action_space: the space for action data.
        auxiliary_info_shape: the shape for auxiliary data if needed.
    """

    def __init__(
            self,
            observation_space: Space,
            action_space: Space,
            auxiliary_info_shape: Optional[dict],
            num_envs: int,
            buffer_size: int,
            device: str = "/CPU:0"
    ):
        assert buffer_size % num_envs == 0, "buffer_size must be divisible by the number of envs (parallels)"

        self.observation_space = observation_space
        self.action_space = action_space
        self.auxiliary_shape = auxiliary_info_shape

        # Pre-define the data that might be stored in replay buffer for training.
        self.observations: Optional[Tensor] = None
        self.next_observations: Optional[Tensor] = None
        self.actions: Optional[Tensor] = None
        self.auxiliary_infos: Optional[Tensor, dict] = None
        self.rewards: Optional[Tensor] = None
        self.terminals: Optional[Tensor] = None
        self.returns: Optional[Tensor] = None
        self.values: Optional[Tensor] = None
        self.advantages: Optional[Tensor] = None

        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.n_size = self.buffer_size // self.num_envs
        self.ptr = 0  # last data pointer
        self.size = 0  # current buffer size per environment.
        self.device = device

    @property
    def full(self):
        return self.size >= self.n_size

    @abstractmethod
    def store(self, *args):
        raise NotImplementedError

    @abstractmethod
    def clear(self, *args):
        raise NotImplementedError

    @abstractmethod
    def sample(self, *args):
        raise NotImplementedError

    def finish_path(self, *args):
        pass


class TensorOnPolicyBuffer(TensorBuffer):
    """
    Replay buffer for on-policy DRL algorithms.

    Args:
        observation_space: the observation space of the environment.
        action_space: the action space of the environment.
        auxiliary_shape: data shape of auxiliary information (if exists).
        n_envs: number of parallel environments.
        horizon_size: max length of steps to store for one environment.
        use_gae: if use GAE trick.
        use_advnorm: if use Advantage normalization trick.
        gamma: discount factor.
        gae_lam: gae lambda.
        device: the calculating device.
    """

    def __init__(
            self,
            observation_space: Space,
            action_space: Space,
            auxiliary_shape: Optional[dict],
            n_envs: int,
            horizon_size: int,
            use_gae: bool = True,
            use_advnorm: bool = True,
            gamma: float = 0.99,
            gae_lam: float = 0.95,
            device: str = "/CPU:0"
    ):
        self.buffer_size = horizon_size * n_envs
        super().__init__(observation_space, action_space, auxiliary_shape, n_envs, self.buffer_size, device)

        self.n_envs = n_envs
        self.horizon_size = horizon_size
        self.n_size = self.horizon_size
        self.use_gae = use_gae
        self.use_advnorm = use_advnorm
        self.gamma = gamma
        self.gae_lam = gae_lam
        self.start_ids = [0 for _ in range(self.n_envs)]
        self.clear()

    @property
    def full(self):
        return self.size >= self.n_size

    def clear(self):
        self.ptr, self.size = 0, 0
        self.observations = create_tensor_memory(space2shape(self.observation_space), self.n_envs, self.n_size,
                                                 device=self.device)
        self.actions = create_tensor_memory(space2shape(self.action_space), self.n_envs, self.n_size,
                                            device=self.device)
        self.rewards = create_tensor_memory((), self.n_envs, self.n_size, device=self.device)
        self.returns = create_tensor_memory((), self.n_envs, self.n_size, device=self.device)
        self.values = create_tensor_memory((), self.n_envs, self.n_size, device=self.device)
        self.terminals = create_tensor_memory((), self.n_envs, self.n_size, device=self.device)
        self.advantages = create_tensor_memory((), self.n_envs, self.n_size, device=self.device)
        self.auxiliary_infos = create_tensor_memory(self.auxiliary_shape, self.n_envs, self.n_size, device=self.device)

    def store(self, obs, acts, rews, value, terminals, aux_info=None):
        store_tensor_element(obs, self.observations, self.ptr)
        store_tensor_element(acts, self.actions, self.ptr)
        store_tensor_element(rews, self.rewards, self.ptr)
        store_tensor_element(value, self.values, self.ptr)
        store_tensor_element(terminals, self.terminals, self.ptr)
        store_tensor_element(aux_info, self.auxiliary_infos, self.ptr)
        self.ptr = (self.ptr + 1) % self.n_size
        self.size = min(self.size + 1, self.n_size)

    def finish_path(self, val, i):
        start = self.start_ids[i]
        end = self.n_size if self.full else self.ptr
        step_nums = end - start

        if step_nums <= 0:
            self.start_ids[i] = self.ptr
            return

        val = tf.reshape(tf.cast(tf.convert_to_tensor(val), self.values.dtype), [1])
        values = self.values[i, start:end]
        vs = tf.concat([values, val], axis=0)
        rewards = self.rewards[i, start:end]
        dones = self.terminals[i, start:end]

        if self.use_gae:  # use gae
            deltas = rewards + (1.0 - dones) * self.gamma * vs[1:] - vs[:-1]
            discounts = (1.0 - dones) * self.gamma * self.gae_lam

            # Reverse scan allows terminal masks to vary at every time step.
            rev_deltas = tf.reverse(deltas, axis=[0])
            rev_discounts = tf.reverse(discounts, axis=[0])
            scan_inputs = (rev_deltas, rev_discounts)

            def gae_scan(acc, elems):
                delta, discount = elems
                return delta + discount * acc

            reversed_advantages = tf.scan(gae_scan, scan_inputs, initializer=tf.zeros([], dtype=rewards.dtype))
            advantages = tf.reverse(reversed_advantages, axis=[0])
            returns = advantages + vs[:-1]

        else:
            rewards_with_bootstrap = tf.concat([rewards, val], axis=0)
            returns = discount_cumsum_tensor(rewards_with_bootstrap, self.gamma)[:-1]
            advantages = rewards + self.gamma * vs[1:] - vs[:-1]

        self.returns[i, start:end].assign(returns)
        self.advantages[i, start:end].assign(advantages)
        self.start_ids[i] = self.ptr

    def sample(self, indexes):
        assert self.full, "Not enough transitions for on-policy buffer to random sample"

        indexes = tf.convert_to_tensor(indexes, dtype=tf.int32)
        env_choices = indexes // self.n_size
        step_choices = indexes % self.n_size
        paired_indexes = (env_choices, step_choices)

        samples_dict = {
            "obs": sample_batch(self.observations, paired_indexes),
            "actions": sample_batch(self.actions, paired_indexes),
            "returns": sample_batch(self.returns, paired_indexes),
            "values": sample_batch(self.values, paired_indexes),
            "aux_batch": sample_batch(self.auxiliary_infos, paired_indexes),
            "batch_size": int(tf.size(indexes)),
        }

        adv_batch = sample_batch(self.advantages, paired_indexes)
        if self.use_advnorm:
            adv_batch = (adv_batch - tf.reduce_mean(adv_batch)) / (_sample_std(adv_batch) + 1e-8)

        samples_dict["advantages"] = adv_batch
        return samples_dict


class TensorOnPolicyBufferAtari(TensorOnPolicyBuffer):
    def __init__(self, *args, **kwargs):
        super(TensorOnPolicyBufferAtari, self).__init__(*args, **kwargs)

    def clear(self):
        self.ptr, self.size = 0, 0
        self.observations = create_tensor_memory(space2shape(self.observation_space), self.n_envs, self.n_size,
                                                 dtype=tf.uint8, device=self.device)
        self.actions = create_tensor_memory(space2shape(self.action_space), self.n_envs, self.n_size,
                                            device=self.device)
        self.rewards = create_tensor_memory((), self.n_envs, self.n_size, device=self.device)
        self.returns = create_tensor_memory((), self.n_envs, self.n_size, device=self.device)
        self.values = create_tensor_memory((), self.n_envs, self.n_size, device=self.device)
        self.terminals = create_tensor_memory((), self.n_envs, self.n_size, device=self.device)
        self.advantages = create_tensor_memory((), self.n_envs, self.n_size, device=self.device)
        self.auxiliary_infos = create_tensor_memory(self.auxiliary_shape, self.n_envs, self.n_size, device=self.device)


class TensorOffPolicyBuffer(TensorBuffer):
    """
    Replay buffer for off-policy DRL algorithms.

    Args:
        observation_space: the observation space of the environment.
        action_space: the action space of the environment.
        auxiliary_shape: data shape of auxiliary information (if exists).
        n_envs: number of parallel environments.
        buffer_size: the total size of the replay buffer.
        batch_size: size of transition data for a batch of sample.
        device: the calculating device.
    """

    def __init__(
            self,
            observation_space: Space,
            action_space: Space,
            auxiliary_shape: Optional[dict],
            n_envs: int,
            buffer_size: int,
            batch_size: int,
            device: str = "/CPU:0",
    ):
        super().__init__(observation_space, action_space, auxiliary_shape, n_envs, buffer_size, device)

        self.n_envs, self.batch_size = n_envs, batch_size
        assert buffer_size % self.n_envs == 0, "buffer_size must be divisible by the number of envs (parallels)"
        self.n_size = buffer_size // self.n_envs
        self.clear()

    def clear(self):
        self.ptr, self.size = 0, 0
        self.observations = create_tensor_memory(space2shape(self.observation_space), self.n_envs, self.n_size,
                                                 device=self.device)
        self.next_observations = create_tensor_memory(space2shape(self.observation_space), self.n_envs, self.n_size,
                                                      device=self.device)
        self.actions = create_tensor_memory(space2shape(self.action_space), self.n_envs, self.n_size,
                                            device=self.device)
        self.auxiliary_infos = create_tensor_memory(self.auxiliary_shape, self.n_envs, self.n_size, device=self.device)
        self.rewards = create_tensor_memory((), self.n_envs, self.n_size, device=self.device)
        self.terminals = create_tensor_memory((), self.n_envs, self.n_size, device=self.device)

    def store(self, obs, acts, rews, terminals, next_obs):
        store_tensor_element(obs, self.observations, self.ptr)
        store_tensor_element(acts, self.actions, self.ptr)
        store_tensor_element(rews, self.rewards, self.ptr)
        store_tensor_element(terminals, self.terminals, self.ptr)
        store_tensor_element(next_obs, self.next_observations, self.ptr)
        self.ptr = (self.ptr + 1) % self.n_size
        self.size = min(self.size + 1, self.n_size)

    def sample(self, batch_size=None):
        bs = batch_size or self.batch_size
        env_choices = tf.random.uniform(shape=(bs,), minval=0, maxval=self.n_envs, dtype=tf.int32)
        step_choices = tf.random.uniform(shape=(bs,), minval=0, maxval=self.size, dtype=tf.int32)
        paired_indexes = (env_choices, step_choices)

        samples_dict = {
            "obs": sample_batch(self.observations, paired_indexes),
            "actions": sample_batch(self.actions, paired_indexes),
            "obs_next": sample_batch(self.next_observations, paired_indexes),
            "rewards": sample_batch(self.rewards, paired_indexes),
            "terminals": sample_batch(self.terminals, paired_indexes),
            "batch_size": bs,
        }
        return samples_dict


class TensorOffPolicyBufferAtari(TensorOffPolicyBuffer):
    def __init__(self, *args, **kwargs):
        super(TensorOffPolicyBufferAtari, self).__init__(*args, **kwargs)

    def clear(self):
        self.observations = create_tensor_memory(space2shape(self.observation_space), self.n_envs, self.n_size,
                                                 dtype=tf.uint8, device=self.device)
        self.next_observations = create_tensor_memory(space2shape(self.observation_space), self.n_envs, self.n_size,
                                                      dtype=tf.uint8, device=self.device)
        self.actions = create_tensor_memory(space2shape(self.action_space), self.n_envs, self.n_size,
                                            device=self.device)
        self.auxiliary_infos = create_tensor_memory(self.auxiliary_shape, self.n_envs, self.n_size, device=self.device)
        self.rewards = create_tensor_memory((), self.n_envs, self.n_size, device=self.device)
        self.terminals = create_tensor_memory((), self.n_envs, self.n_size, device=self.device)
