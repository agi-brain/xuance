import torch
from gymnasium import Space
from abc import ABC, abstractmethod
from typing import Optional, Union
from torch.types import _dtype
from xuance.torch import Tensor
from xuance.common.common_tools import discount_cumsum
from xuance.environment.utils import space2shape


def create_tensor_memory(shape: Optional[Union[tuple, dict]],
                         n_envs: int,
                         n_size: int,
                         dtype: Optional[_dtype] = torch.float32,
                         device: torch.device = torch.device("cpu")):
    """
    Create a torch.Tensor for memory data.

    Args:
        shape: data shape.
        n_envs: number of parallel environments.
        n_size: length of data sequence for each environment.
        dtype: numpy data type.
        device: the calculating device.

    Returns:
        An empty memory space to store data. (initial: torch.zeros(..., device=device, dtype=dtype))
    """
    if shape is None:
        return None
    elif isinstance(shape, dict):
        memory = {}
        for key, value in shape.items():
            if value is None:  # save an object type
                memory[key] = torch.zeros(size=[n_envs, n_size], dtype=dtype, device=device)
            else:
                memory[key] = torch.zeros(size=[n_envs, n_size] + list(value), dtype=dtype, device=device)
        return memory
    elif isinstance(shape, tuple):
        return torch.zeros(size=[n_envs, n_size] + list(shape), dtype=dtype, device=device)
    else:
        raise NotImplementedError


def store_tensor_element(data: Optional[Union[Tensor, dict]],
                         memory: Union[Tensor, dict],
                         ptr: int):
    """
    Insert a step of data into current memory.

    Args:
        data: target data that to be stored.
        memory: the memory where data will be stored.
        ptr: pointer to the location for the data.
    """
    if data is None:
        return
    elif isinstance(data, dict):
        for key, value in data.items():
            memory[key][:, ptr] = data[key]
    else:
        memory[:, ptr] = data


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
    elif isinstance(memory, dict):
        batch = {}
        for key, value in memory.items():
            batch[key] = value[index]
        return batch
    else:
        return memory[index]


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
            device: torch.device = torch.device("cpu"),
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
        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.n_size = self.buffer_size // self.num_envs
        self.advantages: Optional[Tensor] = None
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

    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 auxiliary_shape: Optional[dict],
                 n_envs: int,
                 horizon_size: int,
                 use_gae: bool = True,
                 use_advnorm: bool = True,
                 gamma: float = 0.99,
                 gae_lam: float = 0.95,
                 device: torch.device = torch.device("cpu")):
        self.buffer_size = horizon_size * n_envs
        super(TensorOnPolicyBuffer, self).__init__(observation_space, action_space, auxiliary_shape, n_envs,
                                                   self.buffer_size, device)
        self.n_envs, self.horizon_size = n_envs, horizon_size
        self.n_size = self.horizon_size
        self.use_gae, self.use_advnorm = use_gae, use_advnorm
        self.gamma, self.gae_lam = gamma, gae_lam
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
        val = torch.as_tensor([val], device=self.device)
        if self.full:
            path_slice = range(self.start_ids[i], self.n_size)
        else:
            path_slice = range(self.start_ids[i], self.ptr)
        vs = torch.cat([self.values[i, path_slice], val], axis=0)
        if self.use_gae:  # use gae
            rewards = self.rewards[i, path_slice]
            advantages = torch.zeros_like(rewards)
            dones = self.terminals[i, path_slice]
            last_gae_lam = 0
            step_nums = len(path_slice)
            for t in reversed(range(step_nums)):
                delta = rewards[t] + (1 - dones[t]) * self.gamma * vs[t + 1] - vs[t]
                advantages[t] = last_gae_lam = delta + (1 - dones[t]) * self.gamma * self.gae_lam * last_gae_lam
            returns = advantages + vs[:-1]
        else:
            rewards = torch.cat(self.rewards[i, path_slice], val, axis=0)
            returns = discount_cumsum(rewards, self.gamma)[:-1]
            advantages = rewards[:-1] + self.gamma * vs[1:] - vs[:-1]

        self.returns[i, path_slice] = returns
        self.advantages[i, path_slice] = advantages
        self.start_ids[i] = self.ptr

    def sample(self, indexes):
        assert self.full, "Not enough transitions for on-policy buffer to random sample"

        env_choices, step_choices = divmod(indexes, self.n_size)

        samples_dict = {
            'obs': sample_batch(self.observations, tuple([env_choices, step_choices])),
            'actions': sample_batch(self.actions, tuple([env_choices, step_choices])),
            'returns': sample_batch(self.returns, tuple([env_choices, step_choices])),
            'values': sample_batch(self.values, tuple([env_choices, step_choices])),
            'aux_batch': sample_batch(self.auxiliary_infos, tuple([env_choices, step_choices])),
            'batch_size': len(indexes),
        }
        adv_batch = sample_batch(self.advantages, tuple([env_choices, step_choices]))
        if self.use_advnorm:
            adv_batch = (adv_batch - torch.mean(adv_batch)) / (torch.std(adv_batch) + 1e-8)
        samples_dict.update({
            'advantages': adv_batch
        })

        return samples_dict


class TensorOnPolicyBufferAtari(TensorOnPolicyBuffer):
    def __init__(self, *args, **kwargs):
        super(TensorOnPolicyBufferAtari, self).__init__(*args, **kwargs)

    def clear(self):
        self.ptr, self.size = 0, 0
        self.observations = create_tensor_memory(space2shape(self.observation_space), self.n_envs, self.n_size,
                                                 dtype=torch.int8, device=self.device)
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

    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 auxiliary_shape: Optional[dict],
                 n_envs: int,
                 buffer_size: int,
                 batch_size: int,
                 device: torch.device = torch.device("cpu")):
        super(TensorOffPolicyBuffer, self).__init__(observation_space, action_space, auxiliary_shape, n_envs,
                                                    buffer_size, device)
        self.n_envs, self.batch_size = n_envs, batch_size
        assert buffer_size % self.n_envs == 0, "buffer_size must be divisible by the number of envs (parallels)"
        self.n_size = buffer_size // self.n_envs
        self.clear()

    def clear(self):
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
        env_choices = torch.randint(0, self.n_envs, (bs,))
        step_choices = torch.randint(0, self.size, (bs,))

        samples_dict = {
            'obs': sample_batch(self.observations, tuple([env_choices, step_choices])),
            'actions': sample_batch(self.actions, tuple([env_choices, step_choices])),
            'obs_next': sample_batch(self.next_observations, tuple([env_choices, step_choices])),
            'rewards': sample_batch(self.rewards, tuple([env_choices, step_choices])),
            'terminals': sample_batch(self.terminals, tuple([env_choices, step_choices])),
            'batch_size': bs,
        }
        return samples_dict


class TensorOffPolicyBufferAtari(TensorOffPolicyBuffer):
    def __init__(self, *args, **kwargs):
        super(TensorOffPolicyBufferAtari, self).__init__(*args, **kwargs)

    def clear(self):
        self.observations = create_tensor_memory(space2shape(self.observation_space), self.n_envs, self.n_size,
                                                 dtype=torch.int8, device=self.device)
        self.next_observations = create_tensor_memory(space2shape(self.observation_space), self.n_envs, self.n_size,
                                                      dtype=torch.int8, device=self.device)
        self.actions = create_tensor_memory(space2shape(self.action_space), self.n_envs, self.n_size,
                                            device=self.device)
        self.auxiliary_infos = create_tensor_memory(self.auxiliary_shape, self.n_envs, self.n_size, device=self.device)
        self.rewards = create_tensor_memory((), self.n_envs, self.n_size, device=self.device)
        self.terminals = create_tensor_memory((), self.n_envs, self.n_size, device=self.device)
