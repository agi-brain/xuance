import random
import numpy as np
from xuanpolicy.common import discount_cumsum
from gym import Space
from abc import ABC, abstractmethod
from typing import Optional, Union
from xuanpolicy.common import space2shape
from xuanpolicy.common.segtree_tool import SumSegmentTree, MinSegmentTree
from collections import deque
from typing import Dict


def create_memory(shape: Optional[Union[tuple, dict]], nenvs: int, nsize: int, dtype=np.float32):
    if shape == None:
        return None
    elif isinstance(shape, dict):
        memory = {}
        for key, value in zip(shape.keys(), shape.values()):
            if value is None:  # save an object type
                memory[key] = np.zeros([nenvs, nsize], dtype=object)
            else:
                memory[key] = np.zeros([nenvs, nsize] + list(value), dtype=dtype)
        return memory
    elif isinstance(shape, tuple):
        return np.zeros([nenvs, nsize] + list(shape), dtype)
    else:
        raise NotImplementedError


def store_element(data: Optional[Union[np.ndarray, dict, float]], memory: Union[dict, np.ndarray], ptr: int):
    if data is None:
        return
    elif isinstance(data, dict):
        for key, value in zip(data.keys(), data.values()):
            memory[key][:, ptr] = data[key]
    else:
        memory[:, ptr] = data


def sample_batch(memory: Optional[Union[np.ndarray, dict]], index: Optional[Union[np.ndarray, tuple]]):
    if memory is None:
        return None
    elif isinstance(memory, dict):
        batch = {}
        for key, value in zip(memory.keys(), memory.values()):
            batch[key] = value[index]
        return batch
    else:
        return memory[index]


class Buffer(ABC):
    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 representation_info_shape: Optional[dict],
                 auxiliary_info_shape: Optional[dict]):
        self.observation_space = observation_space
        self.action_space = action_space
        self.representation_shape = representation_info_shape
        self.auxiliary_shape = auxiliary_info_shape
        self.size, self.ptr = 0, 0

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


class EpisodeBuffer:
    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.done = []

    def put(self, transition):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.done.append(transition[3])

    def sample(self, lookup_step=None, idx=None) -> Dict[str, np.ndarray]:
        obs = np.array(self.obs)
        action = np.array(self.action)
        reward = np.array(self.reward)
        done = np.array(self.done)

        obs = obs[idx:idx+lookup_step+1]
        action = action[idx:idx+lookup_step]
        reward = reward[idx:idx+lookup_step]
        done = done[idx:idx+lookup_step]

        return dict(obs=obs,
                    acts=action,
                    rews=reward,
                    done=done)

    def __len__(self) -> int:
        return len(self.action)


class DummyOnPolicyBuffer(Buffer):
    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 representation_shape: Optional[dict],
                 auxiliary_shape: Optional[dict],
                 nenvs: int,
                 nsize: int,
                 nminibatch: int,
                 gamma: float = 0.99,
                 lam: float = 0.95):
        super(DummyOnPolicyBuffer, self).__init__(observation_space,
                                                  action_space,
                                                  representation_shape,
                                                  auxiliary_shape)
        self.nenvs, self.nsize, self.nminibatch = nenvs, nsize, nminibatch
        self.gamma, self.lam = gamma, lam
        self.start_ids = np.zeros(self.nenvs, np.int64)
        self.observations = create_memory(space2shape(self.observation_space), self.nenvs, self.nsize)
        self.actions = create_memory(space2shape(self.action_space), self.nenvs, self.nsize)
        self.rewards = create_memory((), self.nenvs, self.nsize)
        self.returns = create_memory((), self.nenvs, self.nsize)
        self.terminals = create_memory((), self.nenvs, self.nsize)
        self.advantages = create_memory((), self.nenvs, self.nsize)
        self.auxiliary_infos = create_memory(self.auxiliary_shape, self.nenvs, self.nsize)

    @property
    def full(self):
        return self.size >= self.nsize

    def clear(self):
        self.ptr, self.size = 0, 0
        self.observations = create_memory(space2shape(self.observation_space), self.nenvs, self.nsize)
        self.actions = create_memory(space2shape(self.action_space), self.nenvs, self.nsize)
        self.rewards = create_memory((), self.nenvs, self.nsize)
        self.returns = create_memory((), self.nenvs, self.nsize)
        self.advantages = create_memory((), self.nenvs, self.nsize)
        self.auxiliary_infos = create_memory(self.auxiliary_shape, self.nenvs, self.nsize)

    def store(self, obs, acts, rews, rets, terminals, aux_info=None):
        store_element(obs, self.observations, self.ptr)
        store_element(acts, self.actions, self.ptr)
        store_element(rews, self.rewards, self.ptr)
        store_element(rets, self.returns, self.ptr)
        store_element(terminals, self.terminals, self.ptr)
        store_element(aux_info, self.auxiliary_infos, self.ptr)
        self.ptr = (self.ptr + 1) % self.nsize
        self.size = min(self.size + 1, self.nsize)

    def finish_path(self, val, i):
        if self.full:
            path_slice = np.arange(self.start_ids[i], self.nsize).astype(np.int32)
        else:
            path_slice = np.arange(self.start_ids[i], self.ptr).astype(np.int32)
        rewards = np.append(np.array(self.rewards[i, path_slice]), [val], axis=0)
        critics = np.append(np.array(self.returns[i, path_slice]), [val], axis=0)
        returns = discount_cumsum(rewards, self.gamma)[:-1]
        deltas = rewards[:-1] + self.gamma * critics[1:] - critics[:-1]
        advantages = discount_cumsum(deltas, self.gamma * self.lam)
        self.returns[i, path_slice] = returns
        self.advantages[i, path_slice] = advantages
        self.start_ids[i] = self.ptr

    def sample(self):
        assert self.full, "Not enough transitions for on-policy buffer to random sample"

        env_choices = np.random.choice(self.nenvs, self.nenvs * self.nsize // self.nminibatch)
        step_choices = np.random.choice(self.nsize, self.nenvs * self.nsize // self.nminibatch)

        obs_batch = sample_batch(self.observations, tuple([env_choices, step_choices]))
        act_batch = sample_batch(self.actions, tuple([env_choices, step_choices]))
        ret_batch = sample_batch(self.returns, tuple([env_choices, step_choices]))
        adv_batch = sample_batch(self.advantages, tuple([env_choices, step_choices]))
        adv_batch = (adv_batch - np.mean(self.advantages)) / (np.std(self.advantages) + 1e-8)
        aux_batch = sample_batch(self.auxiliary_infos, tuple([env_choices, step_choices]))

        return obs_batch, act_batch, ret_batch, adv_batch, aux_batch


class DummyOffPolicyBuffer(Buffer):
    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 representation_shape: Optional[dict],
                 auxiliary_shape: Optional[dict],
                 nenvs: int,
                 nsize: int,
                 batchsize: int):
        super(DummyOffPolicyBuffer, self).__init__(observation_space,
                                                   action_space,
                                                   representation_shape,
                                                   auxiliary_shape)
        self.nenvs, self.nsize, self.batchsize = nenvs, nsize, batchsize
        self.observations = create_memory(space2shape(self.observation_space), self.nenvs, self.nsize)
        self.next_observations = create_memory(space2shape(self.observation_space), self.nenvs, self.nsize)
        self.actions = create_memory(space2shape(self.action_space), self.nenvs, self.nsize)
        self.rewards = create_memory((), self.nenvs, self.nsize)
        self.terminals = create_memory((), self.nenvs, self.nsize)

    def clear(self):
        self.observations = create_memory(space2shape(self.observation_space), self.nenvs, self.nsize)
        self.next_observations = create_memory(space2shape(self.observation_space), self.nenvs, self.nsize)
        self.actions = create_memory(space2shape(self.action_space), self.nenvs, self.nsize)
        self.rewards = create_memory((), self.nenvs, self.nsize)
        self.terminals = create_memory((), self.nenvs, self.nsize)

    def store(self, obs, acts, rews, terminals, next_obs):
        store_element(obs, self.observations, self.ptr)
        store_element(acts, self.actions, self.ptr)
        store_element(rews, self.rewards, self.ptr)
        store_element(terminals, self.terminals, self.ptr)
        store_element(next_obs, self.next_observations, self.ptr)
        self.ptr = (self.ptr + 1) % self.nsize
        self.size = min(self.size + 1, self.nsize)

    def sample(self):
        env_choices = np.random.choice(self.nenvs, self.batchsize)
        step_choices = np.random.choice(self.size, self.batchsize)
        obs_batch = sample_batch(self.observations, tuple([env_choices, step_choices]))
        act_batch = sample_batch(self.actions, tuple([env_choices, step_choices]))
        rew_batch = sample_batch(self.rewards, tuple([env_choices, step_choices]))
        terminal_batch = sample_batch(self.terminals, tuple([env_choices, step_choices]))
        next_batch = sample_batch(self.next_observations, tuple([env_choices, step_choices]))
        return obs_batch, act_batch, rew_batch, terminal_batch, next_batch


class RecurrentOffPolicyBuffer(Buffer):
    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 representation_shape: Optional[dict],
                 auxiliary_shape: Optional[dict],
                 nenvs: int,
                 nsize: int,
                 batchsize: int,
                 episode_length: int,
                 lookup_length: int):
        super(RecurrentOffPolicyBuffer, self).__init__(observation_space,
                                                       action_space,
                                                       representation_shape,
                                                       auxiliary_shape)
        self.nenvs, self.nsize, self.episode_length, self.batchsize = nenvs, nsize, episode_length, batchsize
        self.lookup_length = lookup_length
        self.memory = deque(maxlen=self.nsize)

    @property
    def full(self):
        return self.size >= self.nsize

    def clear(self, *args):
        self.memory = deque(maxlen=self.nsize)

    def store(self, episode):
        self.memory.append(episode)
        self.ptr = (self.ptr + 1) % self.nsize
        self.size = min(self.size + 1, self.nsize)

    def sample(self):
        obs_batch, act_batch, rew_batch, terminal_batch = [], [], [], []
        episode_choices = np.random.choice(self.memory, self.batchsize)
        length_min = self.episode_length
        for episode in episode_choices:
            length_min = min(length_min, len(episode))

        if length_min > self.lookup_length:
            for episode in episode_choices:
                start_idx = np.random.randint(0, len(episode)-self.lookup_length+1)
                sampled_data = episode.sample(lookup_step=self.lookup_length, idx=start_idx)
                obs_batch.append(sampled_data["obs"])
                act_batch.append(sampled_data["acts"])
                rew_batch.append(sampled_data["rews"])
                terminal_batch.append(sampled_data["done"])
        else:
            for episode in episode_choices:
                start_idx = np.random.randint(0, len(episode) - length_min + 1)
                sampled_data = episode.sample(lookup_step=length_min, idx=start_idx)
                obs_batch.append(sampled_data["obs"])
                act_batch.append(sampled_data["acts"])
                rew_batch.append(sampled_data["rews"])
                terminal_batch.append(sampled_data["done"])

        return np.array(obs_batch), np.array(act_batch), np.array(rew_batch), np.array(terminal_batch)


class PerOffPolicyBuffer(Buffer):
    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 representation_shape: Optional[dict],
                 auxiliary_shape: Optional[dict],
                 nenvs: int,
                 nsize: int,
                 batchsize: int,
                 alpha=0.6):
        super(PerOffPolicyBuffer, self).__init__(observation_space,
                                                 action_space,
                                                 representation_shape,
                                                 auxiliary_shape)
        self.nenvs, self.nsize, self.batchsize = nenvs, nsize, batchsize
        self.observations = create_memory(space2shape(self.observation_space), self.nenvs, self.nsize)
        self.next_observations = create_memory(space2shape(self.observation_space), self.nenvs, self.nsize)
        self.actions = create_memory(space2shape(self.action_space), self.nenvs, self.nsize)
        self.rewards = create_memory((), self.nenvs, self.nsize)
        self.terminals = create_memory((), self.nenvs, self.nsize)

        self._alpha = alpha

        # set segment tree size
        it_capacity = 1
        while it_capacity < self.nsize:
            it_capacity *= 2

        # init segment tree
        self._it_sum = []
        self._it_min = []
        for _ in range(nenvs):
            self._it_sum.append(SumSegmentTree(it_capacity))
            self._it_min.append(MinSegmentTree(it_capacity))
        self._max_priority = np.ones((nenvs))

    def _sample_proportional(self, env_idx, batch_size):
        res = []
        p_total = self._it_sum[env_idx].sum(0, self.size - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum[env_idx].find_prefixsum_idx(mass)
            res.append(int(idx))
        return res

    def clear(self):
        self.observations = create_memory(space2shape(self.observation_space), self.nenvs, self.nsize)
        self.next_observations = create_memory(space2shape(self.observation_space), self.nenvs, self.nsize)
        self.actions = create_memory(space2shape(self.action_space), self.nenvs, self.nsize)
        self.rewards = create_memory((), self.nenvs, self.nsize)
        self.terminals = create_memory((), self.nenvs, self.nsize)
        self._it_sum = []
        self._it_min = []

    def store(self, obs, acts, rews, terminals, next_obs):
        store_element(obs, self.observations, self.ptr)
        store_element(acts, self.actions, self.ptr)
        store_element(rews, self.rewards, self.ptr)
        store_element(terminals, self.terminals, self.ptr)
        store_element(next_obs, self.next_observations, self.ptr)

        # prioritized process
        for i in range(self.nenvs):
            self._it_sum[i][self.ptr] = self._max_priority[i] ** self._alpha
            self._it_min[i][self.ptr] = self._max_priority[i] ** self._alpha

        self.ptr = (self.ptr + 1) % self.nsize
        self.size = min(self.size + 1, self.nsize)

    def sample(self, beta):
        env_choices = np.array(range(self.nenvs)).repeat(int(self.batchsize / self.nenvs))
        step_choices = np.zeros((self.nenvs, int(self.batchsize / self.nenvs)))
        weights = np.zeros((self.nenvs, int(self.batchsize / self.nenvs)))

        assert beta > 0

        for i in range(self.nenvs):
            idxes = self._sample_proportional(i, int(self.batchsize / self.nenvs))

            weights_ = []
            p_min = self._it_min[i].min() / self._it_sum[i].sum()
            max_weight = p_min * self.size ** (-beta)

            for idx in idxes:
                p_sample = self._it_sum[i][idx] / self._it_sum[i].sum()
                weight = p_sample * self.size ** (-beta)
                weights_.append(weight / max_weight)
            step_choices[i] = idxes
            weights[i] = np.array(weights_)
        step_choices = step_choices.astype(np.uint8)

        obs_batch = sample_batch(self.observations, tuple([env_choices, step_choices.flatten()]))
        act_batch = sample_batch(self.actions, tuple([env_choices, step_choices.flatten()]))
        rew_batch = sample_batch(self.rewards, tuple([env_choices, step_choices.flatten()]))
        terminal_batch = sample_batch(self.terminals, tuple([env_choices, step_choices.flatten()]))
        next_batch = sample_batch(self.next_observations, tuple([env_choices, step_choices.flatten()]))

        # return tuple(list(encoded_sample) + [weights, idxes])
        return (obs_batch,
                act_batch,
                rew_batch,
                terminal_batch,
                next_batch,
                weights,
                step_choices)

    def update_priorities(self, idxes, priorities):
        priorities = priorities.reshape((self.nenvs, int(self.batchsize / self.nenvs)))
        for i in range(self.nenvs):
            for idx, priority in zip(idxes[i], priorities[i]):
                if priority == 0:
                    priority += 1e-8
                assert 0 <= idx < self.size
                self._it_sum[i][idx] = priority ** self._alpha
                self._it_min[i][idx] = priority ** self._alpha

                self._max_priority[i] = max(self._max_priority[i], priority)


class DummyOffPolicyBuffer_Atari(DummyOffPolicyBuffer):
    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 representation_shape: Optional[dict],
                 auxiliary_shape: Optional[dict],
                 nenvs: int,
                 nsize: int,
                 batchsize: int):
        super(DummyOffPolicyBuffer_Atari, self).__init__(observation_space, action_space,
                                                         representation_shape,
                                                         auxiliary_shape,
                                                         nenvs,
                                                         nsize,
                                                         batchsize)
        self.observations = create_memory(space2shape(self.observation_space), self.nenvs, self.nsize, np.uint8)
        self.next_observations = create_memory(space2shape(self.observation_space), self.nenvs, self.nsize, np.uint8)

    def clear(self):
        self.observations = create_memory(space2shape(self.observation_space), self.nenvs, self.nsize, np.uint8)
        self.next_observations = create_memory(space2shape(self.observation_space), self.nenvs, self.nsize, np.uint8)
        self.actions = create_memory(space2shape(self.action_space), self.nenvs, self.nsize)
        self.representation_infos = create_memory(self.representation_shape, self.nenvs, self.nsize)
        self.auxiliary_infos = create_memory(self.auxiliary_shape, self.nenvs, self.nsize)
        self.rewards = create_memory((), self.nenvs, self.nsize)
        self.terminals = create_memory((), self.nenvs, self.nsize)


class DummyOnPolicyBuffer_Atari(DummyOnPolicyBuffer):
    def __init__(self, observation_space: Space,
                 action_space: Space,
                 representation_shape: Optional[dict],
                 auxiliary_shape: Optional[dict],
                 nenvs: int,
                 nsize: int,
                 nminibatch: int,
                 gamma: float = 0.99,
                 lam: float = 0.95):
        super(DummyOnPolicyBuffer_Atari, self).__init__(observation_space, action_space,
                                                        representation_shape, auxiliary_shape, nenvs, nsize, nminibatch,
                                                        gamma, lam)
        self.observations = create_memory(space2shape(self.observation_space), self.nenvs, self.nsize, np.uint8)

    def clear(self):
        self.ptr, self.size = 0, 0
        self.observations = create_memory(space2shape(self.observation_space), self.nenvs, self.nsize, np.uint8)
        self.actions = create_memory(space2shape(self.action_space), self.nenvs, self.nsize)
        self.representation_infos = create_memory(self.representation_shape, self.nenvs, self.nsize)
        self.auxiliary_infos = create_memory(self.auxiliary_shape, self.nenvs, self.nsize)
        self.rewards = create_memory((), self.nenvs, self.nsize)
        self.returns = create_memory((), self.nenvs, self.nsize)
        self.advantages = create_memory((), self.nenvs, self.nsize)
