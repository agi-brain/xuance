Memory
==============================================

This module defines different type of classes used to implement the experience replay buffer for DRL with single agent.

.. raw:: html

    <br><hr>

Basic Memory Tools
-----------------------------------

.. py:function::
  xuance.common.memory_tools.create_memory(shape, n_envs, n_size, dtype)

  Create a numpy array for memory data.

  :param shape: data shape.
  :type shape: tuple, dict
  :param n_envs: number of parallel environments.
  :type n_envs: int
  :param n_size: length of data sequence for each environment.
  :type n_size: int
  :param dtype: numpy data type.
  :type dtype: np.dtype
  :return: An empty memory space to store data. (initial: numpy.zeros()).
  :rtype: np.ndarray

.. py:function::
  xuance.common.memory_tools.store_element(data, memory, ptr)

  Insert a step of data into current memory.

  :param data: target data that to be stored.
  :type data: np.ndarray, dict, float
  :param memory: the memory where data will be stored.
  :type memory: dict, np.ndarray
  :param ptr: pointer to the location for the data.
  :type ptr: int

.. py:function::
  xuance.common.memory_tools.sample_batch(memory, index)

  Sample a batch of data from the selected memory.

  :param memory: memory that contains experience data.
  :type memory: np.ndarray, dict
  :param index: pointer to the location for the selected data.
  :type index: np.ndarray, tuple
  :return: A batch of data.
  :rtype: np.ndarray, dict

.. py:class::
  xuance.common.memory_tools.Buffer(observation_space, action_space, auxiliary_info_shape)

  Basic buffer single-agent DRL algorithms

  :param observation_space: the space for observation data.
  :type observation_space: Space
  :param action_space: the space for action data.
  :type action_space: Space
  :param auxiliary_info_shape: the shape for auxiliary data if needed.
  :type auxiliary_info_shape: dict

.. py:function::
  xuance.common.memory_tools.Buffer.full()

  Determine whether the current experience replay buffer is full.

  :return: A bool value, True means the buffer is full, False means the buffer is not full yet.
  :rtype: bool

.. py:function::
  xuance.common.memory_tools.Buffer.store(*args)
  
  Store new experience data to the buffer.

.. py:function::
  xuance.common.memory_tools.Buffer.clear(*args)

  Clear the whole buffer.

.. py:function::
  xuance.common.memory_tools.Buffer.sample(*args)

  Sample a batch of experience data from the buffer.

.. py:function::
  xuance.common.memory_tools.Buffer.finish_path(*args)

  When an episode is finished, calculate the returns, advantages, and others.

.. py:class::
  xuance.common.memory_tools.EpisodeBuffer(obs, action, reward, done)

  Episode buffer for DRQN agent.

  :param obs: The observation variables.
  :type obs: np.ndarray
  :param action: The action variables.
  :type action: np.ndarray
  :param reward: The reward variables.
  :type reward: np.ndarray
  :param done: The terminal variables.
  :type done: np.ndarray

.. py:function::
  xuance.common.memory_tools.EpisodeBuffer.put(transition)

  Put a transition data to the buffer.

  :param transition: One step transition data.
  :type transition: list

.. py:function::
  xuance.common.memory_tools.EpisodeBuffer.sample(lookup_step, idx)

  Sample a bach of sequences.

  :param lookup_step: The length of the history steps.
  :type lookup_step: int
  :param idx: The start index of the data.
  :type idx: int
  :return: A batch of sampled experience data.
  :rtype: Dict[str, np.ndarray]

.. py:function::
  xuance.common.memory_tools.EpisodeBuffer.__len__(lookup_step, idx)

  Get the current length of the replay buffer.

  :return: The current length of the replay buffer.
  :rtype: int

.. raw:: html

    <br><hr>

On-Policy Buffer for DRL
---------------------------------------------------------

.. py:class::
  xuance.common.memory_tools.DummyOnPolicyBuffer(observation_space, action_space, auxiliary_shape, n_envs, n_size, use_gae, use_advnorm, gamma, gae_lam)
  
  Replay buffer for on-policy DRL algorithms.

  :param observation_space: the observation space of the environment.
  :type observation_space: Space
  :param action_space: the action space of the environment.
  :type action_space: Space
  :param auxiliary_shape: data shape of auxiliary information (if exists).
  :type auxiliary_shape: dict
  :param n_envs: number of parallel environments.
  :type n_envs: int
  :param n_size: max length of steps to store for one environment.
  :type n_size: int
  :param use_gae: if use GAE trick.
  :type use_gae: bool
  :param use_advnorm: if use Advantage normalization trick.
  :type use_advnorm: bool
  :param gamma: discount factor.
  :type gamma: float
  :param gae_lam: gae lambda.
  :type gae_lam: float

.. py:function::
  xuance.common.memory_tools.DummyOnPolicyBuffer.full()

  Determine whether the current experience replay buffer is full.

  :return: A bool value, True means the buffer is full, False means the buffer is not full yet.
  :rtype: bool

.. py:function::
  xuance.common.memory_tools.DummyOnPolicyBuffer.clear()

  Clear the whole buffer.

.. py:function::
  xuance.common.memory_tools.DummyOnPolicyBuffer.store(obs, acts, rews, value, terminals, aux_info)

  Store one-step transition data, including observations, actions, rewars, values, terminal variables, and auxiliary informations, into the buffer.

  :param obs: The one-step observation variables.
  :type obs: np.ndarray
  :param acts: The one-step actions variables.
  :type acts: np.ndarray
  :param rews: The rewards that is achieved after executing the acts according to obs.
  :type rews: np.ndarray
  :param value: The values that is calculated by the current model.
  :type value: np.ndarray
  :param terminals: The bool variables, True means the episode if terminated, False means the episode is not terminated.
  :type terminals: bool
  :param aux_info: Some auxiliary information for some specific algorithms.
  :type aux_info: dict

.. py:function::
  xuance.common.memory_tools.DummyOnPolicyBuffer.finish_path(val, i)

  When an episode is finished, calculate the returns, advantages, and others.

  :param val: The values for the final state.
  :type val: np.ndarray
  :param i: The index of the environment that is terminated.
  :type i: int

.. py:function::
  xuance.common.memory_tools.DummyOnPolicyBuffer.sample(indexes)

  Sample a batch of experience data from the buffer.

  :param indexes: The indexes of the data in the buffer.
  :type indexes: np.ndarray
  :return: A tuple that contains a batch of observations, actions, returns, values, advantages, and auxiliary data.
  :rtype: tuple

.. py:class::
  xuance.common.memory_tools.DummyOnPolicyBuffer_Atari(observation_space, action_space, auxiliary_shape, n_envs, n_size, use_gae, use_advnorm, gamma, gae_lam)
  
  Replay buffer for on-policy DRL algorithms and Atari tasks

  :param observation_space: the observation space of the environment.
  :type observation_space: Space
  :param action_space: the action space of the environment.
  :type action_space: Space
  :param auxiliary_shape: data shape of auxiliary information (if exists).
  :type auxiliary_shape: dict
  :param n_envs: number of parallel environments.
  :type n_envs: int
  :param n_size: max length of steps to store for one environment.
  :type n_size: int
  :param use_gae: determin whether to use GAE trick.
  :type use_gae: bool
  :param use_advnorm: determin whether to use Advantage normalization trick.
  :type use_advnorm: bool
  :param gamma: discount factor.
  :type gamma: float
  :param gae_lam: gae lambda.
  :type gae_lam: float

.. py:function::
  xuance.common.memory_tools.DummyOnPolicyBuffer_Atari.clear()

  Clear the whole buffer.


.. raw:: html

    <br><hr>

Off-Policy Buffer for DRL
---------------------------------------------------------

.. py:class::
  xuance.common.memory_tools.DummyOffPolicyBuffer(observation_space, action_space, auxiliary_shape, n_envs, n_size, batch_size)

  Replay buffer for off-policy DRL algorithms

  :param observation_space: the observation space of the environment.
  :type observation_space: Space
  :param action_space: the action space of the environment.
  :type action_space: Space
  :param auxiliary_shape: data shape of auxiliary information (if exists).
  :type auxiliary_shape: dict
  :param n_envs: number of parallel environments.
  :type n_envs: int
  :param n_size: max length of steps to store for one environment.
  :type n_size: int
  :param batch_size: batch size of transition data for a sample.
  :type batch_size: int

.. py:function::
  xuance.common.memory_tools.DummyOffPolicyBuffer.clear()

  Clear the whole buffer.

.. py:function::
  xuance.common.memory_tools.DummyOffPolicyBuffer.store(obs, acts, rews, terminals, next_obs)

  Store one-step transition data, including observations, actions, rewars, terminal variables, and next step observations, into the buffer.

  :param obs: The observation variables.
  :type obs: np.ndarray
  :param acts: The one-step actions variables.
  :type acts: np.ndarray
  :param rews: The rewards that is achieved after executing the acts according to obs.
  :type rews: np.ndarray
  :param terminals: The bool variables, True means the episode if terminated, False means the episode is not terminated.
  :type terminals: np.ndarray
  :param next_obs: The next step observations.
  :type next_obs: np.ndarray

.. py:function::
  xuance.common.memory_tools.DummyOffPolicyBuffer.sample(indexes)

  Sample a batch of experience data from the buffer.

  :return: The indexes of the data in the buffer.
  :rtype: np.ndarray

.. py:class::
  xuance.common.memory_tools.RecurrentOffPolicyBuffer(observation_space, action_space, auxiliary_shape, n_envs, n_size, batch_size, episode_length, lookup_length)

  Replay buffer for DRQN-based algorithms

  :param observation_space: the observation space of the environment.
  :type observation_space: Space
  :param action_space: the action space of the environment.
  :type action_space: Space
  :param auxiliary_shape: data shape of auxiliary information (if exists).
  :type auxiliary_shape: dict
  :param n_envs: number of parallel environments.
  :type n_envs: int
  :param n_size: max length of steps to store for one environment.
  :type n_size: int
  :param batch_size: batch size of transition data for a sample.
  :type batch_size: int
  :param episode_length: data length for an episode.
  :type episode_length: int
  :param lookup_length: the length of history data.
  :type lookup_length: int

.. py:function::
  xuance.common.memory_tools.RecurrentOffPolicyBuffer.full()

  Determine whether the current experience replay buffer is full.

  :return: A bool value, True means the buffer is full, False means the buffer is not full yet.
  :type return: bool

.. py:function::
  xuance.common.memory_tools.RecurrentOffPolicyBuffer.clear()

  Clear the whole buffer.

.. py:function::
  xuance.common.memory_tools.RecurrentOffPolicyBuffer.store(episode)

  Store a sequence of data for an episode.

  :param episode: The sequence data of an episode to be stored.
  :type episode: dict

.. py:function::
  xuance.common.memory_tools.RecurrentOffPolicyBuffer.sample()

  Sample a batch of experience data for different episodes from the buffer.

  :return: A batch of episodes data.
  :rtype: dict

.. py:class::
  xuance.common.memory_tools.PerOffPolicyBuffer(observation_space, action_space, auxiliary_shape, n_envs, n_size, batch_size, alpha)

  A Prioritized Replay Buffer for reinforcement learning. 
  This buffer is designed for storing and sampling experiences with priorities based on their TD errors.

  :param observation_space: the observation space of the environment.
  :type observation_space: Space
  :param action_space: the action space of the environment.
  :type action_space: Space
  :param auxiliary_shape: data shape of auxiliary information (if exists).
  :type auxiliary_shape: dict
  :param n_envs: number of parallel environments.
  :type n_envs: int
  :param n_size: max length of steps to store for one environment.
  :type n_size: int
  :param batch_size: batch size of transition data for a sample.
  :type batch_size: int
  :param alpha: prioritized factor, default is 0.6.
  :type alpha: float

.. py:function::
  xuance.common.memory_tools.PerOffPolicyBuffer._sample_proportional(env_idx, batch_size)

  This method performs proportional sampling based on priorities for a given environment and batch size.

  :param env_idx: The index of the environment.
  :type env_idx: int
  :param batch_size: The sample size of the batch data.
  :type batch_size: int
  :return: A batch of sampled experience data from the buffer.
  :rtype: np.ndarray

.. py:function::
  xuance.common.memory_tools.PerOffPolicyBuffer.clear()

  Resets all memory buffers and segment trees.

.. py:function::
  xuance.common.memory_tools.PerOffPolicyBuffer.store(obs, acts, rews, terminals, next_obs)

  Stores a transition (observation, action, reward, terminal, next observation) in the memory buffers.
  Updates priorities based on TD errors.

  :param obs: The one-step observation variables.
  :type obs: np.ndarray
  :param acts: The one-step action variables.
  :type acts: np.ndarray
  :param rews: The reward variables.
  :type rews: np.ndarray
  :param terminals: The bool variables, True means the episode if terminated, False means the episode is not terminated.
  :type terminals: np.ndarray
  :param next_obs: The next-step observation variables.
  :type next_obs: np.ndarray

.. py:function::
  xuance.common.memory_tools.PerOffPolicyBuffer.sample(beta)

  Samples a batch of transitions from the buffer based on priorities.
  Uses proportional sampling and calculates importance weights for prioritized replay..

  :param beta: The hyperparameter that influences the balance between prioritized sampling and uniform random sampling.
  :type beta: float
  :return: A tuple of batch samples, including a batch of observations, actions, rewards, terminals, next observations, weights, and step choices.
  :rtype: tuple

.. py:function::
  xuance.common.memory_tools.PerOffPolicyBuffer.update_priorities(idxes, priorities)

  Updates priorities in the buffer based on given indices and priorities.

  :param idxes: the indices of experiences in the replay buffer for which priorities are being updated.
  :type idxes: np.ndarray, list
  :param priorities: the new priority values associated with specific experiences or transitions in the replay buffer.
  :type priorities: np.ndarray

.. py:class::
  xuance.common.memory_tools.DummyOffPolicyBuffer_Atari(observation_space, action_space, auxiliary_shape, n_envs, n_size, batch_size)

  Replay buffer for off-policy DRL algorithms and Atari tasks

  :param observation_space: the observation space of the environment.
  :type observation_space: Space
  :param action_space: the action space of the environment.
  :type action_space: Space
  :param auxiliary_shape: data shape of auxiliary information (if exists).
  :type auxiliary_shape: dict
  :param n_envs: number of parallel environments.
  :type n_envs: int
  :param n_size: max length of steps to store for one environment.
  :type n_size: int
  :param batch_size: batch size of transition data for a sample.
  :type batch_size: int

.. py:function::
  xuance.common.memory_tools.DummyOffPolicyBuffer_Atari.clear()

  Clear the whole buffer.


.. raw:: html

    <br><hr>

Source Code
-----------------

.. code-block:: python

  import random
  import numpy as np
  from gym import Space
  from abc import ABC, abstractmethod
  from typing import Optional, Union
  from xuance.common import space2shape, discount_cumsum
  from xuance.common.segtree_tool import SumSegmentTree, MinSegmentTree
  from collections import deque
  from typing import Dict


  def create_memory(shape: Optional[Union[tuple, dict]],
                    n_envs: int,
                    n_size: int,
                    dtype: type = np.float32):
      """
      Create a numpy array for memory data.

      Args:
          shape: data shape.
          n_envs: number of parallel environments.
          n_size: length of data sequence for each environment.
          dtype: numpy data type.

      Returns:
          An empty memory space to store data. (initial: numpy.zeros())
      """
      if shape is None:
          return None
      elif isinstance(shape, dict):
          memory = {}
          for key, value in zip(shape.keys(), shape.values()):
              if value is None:  # save an object type
                  memory[key] = np.zeros([n_envs, n_size], dtype=object)
              else:
                  memory[key] = np.zeros([n_envs, n_size] + list(value), dtype=dtype)
          return memory
      elif isinstance(shape, tuple):
          return np.zeros([n_envs, n_size] + list(shape), dtype)
      else:
          raise NotImplementedError


  def store_element(data: Optional[Union[np.ndarray, dict, float]],
                    memory: Union[dict, np.ndarray],
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
          for key, value in zip(data.keys(), data.values()):
              memory[key][:, ptr] = data[key]
      else:
          memory[:, ptr] = data


  def sample_batch(memory: Optional[Union[np.ndarray, dict]],
                  index: Optional[Union[np.ndarray, tuple]]):
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
          for key, value in zip(memory.keys(), memory.values()):
              batch[key] = value[index]
          return batch
      else:
          return memory[index]


  class Buffer(ABC):
      """
      Basic buffer single-agent DRL algorithms.

      Args:
          observation_space: the space for observation data.
          action_space: the space for action data.
          auxiliary_info_shape: the shape for auxiliary data if needed.
      """
      def __init__(self,
                  observation_space: Space,
                  action_space: Space,
                  auxiliary_info_shape: Optional[dict]):
          self.observation_space = observation_space
          self.action_space = action_space
          self.auxiliary_shape = auxiliary_info_shape
          self.size, self.ptr = 0, 0

      def full(self):
          pass

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
      """
      Episode buffer for DRQN agent.
      """
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

          obs = obs[idx:idx + lookup_step + 1]
          action = action[idx:idx + lookup_step]
          reward = reward[idx:idx + lookup_step]
          done = done[idx:idx + lookup_step]

          return dict(obs=obs,
                      acts=action,
                      rews=reward,
                      done=done)

      def __len__(self) -> int:
          return len(self.action)


  class DummyOnPolicyBuffer(Buffer):
      """
      Replay buffer for on-policy DRL algorithms.

      Args:
          observation_space: the observation space of the environment.
          action_space: the action space of the environment.
          auxiliary_shape: data shape of auxiliary information (if exists).
          n_envs: number of parallel environments.
          n_size: max length of steps to store for one environment.
          use_gae: if use GAE trick.
          use_advnorm: if use Advantage normalization trick.
          gamma: discount factor.
          gae_lam: gae lambda.
      """
      def __init__(self,
                  observation_space: Space,
                  action_space: Space,
                  auxiliary_shape: Optional[dict],
                  n_envs: int,
                  n_size: int,
                  use_gae: bool = True,
                  use_advnorm: bool = True,
                  gamma: float = 0.99,
                  gae_lam: float = 0.95):
          super(DummyOnPolicyBuffer, self).__init__(observation_space, action_space, auxiliary_shape)
          self.n_envs, self.n_size = n_envs, n_size
          self.buffer_size = self.n_size * self.n_envs
          self.use_gae, self.use_advnorm = use_gae, use_advnorm
          self.gamma, self.gae_lam = gamma, gae_lam
          self.start_ids = np.zeros(self.n_envs, np.int64)
          self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
          self.actions = create_memory(space2shape(self.action_space), self.n_envs, self.n_size)
          self.rewards = create_memory((), self.n_envs, self.n_size)
          self.returns = create_memory((), self.n_envs, self.n_size)
          self.values = create_memory((), self.n_envs, self.n_size)
          self.terminals = create_memory((), self.n_envs, self.n_size)
          self.advantages = create_memory((), self.n_envs, self.n_size)
          self.auxiliary_infos = create_memory(self.auxiliary_shape, self.n_envs, self.n_size)

      @property
      def full(self):
          return self.size >= self.n_size

      def clear(self):
          self.ptr, self.size = 0, 0
          self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
          self.actions = create_memory(space2shape(self.action_space), self.n_envs, self.n_size)
          self.rewards = create_memory((), self.n_envs, self.n_size)
          self.returns = create_memory((), self.n_envs, self.n_size)
          self.values = create_memory((), self.n_envs, self.n_size)
          self.terminals = create_memory((), self.n_envs, self.n_size)
          self.advantages = create_memory((), self.n_envs, self.n_size)
          self.auxiliary_infos = create_memory(self.auxiliary_shape, self.n_envs, self.n_size)

      def store(self, obs, acts, rews, value, terminals, aux_info=None):
          store_element(obs, self.observations, self.ptr)
          store_element(acts, self.actions, self.ptr)
          store_element(rews, self.rewards, self.ptr)
          store_element(value, self.values, self.ptr)
          store_element(terminals, self.terminals, self.ptr)
          store_element(aux_info, self.auxiliary_infos, self.ptr)
          self.ptr = (self.ptr + 1) % self.n_size
          self.size = min(self.size + 1, self.n_size)

      def finish_path(self, val, i):
          if self.full:
              path_slice = np.arange(self.start_ids[i], self.n_size).astype(np.int32)
          else:
              path_slice = np.arange(self.start_ids[i], self.ptr).astype(np.int32)
          vs = np.append(np.array(self.values[i, path_slice]), [val], axis=0)
          if self.use_gae:  # use gae
              rewards = np.array(self.rewards[i, path_slice])
              advantages = np.zeros_like(rewards)
              dones = np.array(self.terminals[i, path_slice])
              last_gae_lam = 0
              step_nums = len(path_slice)
              for t in reversed(range(step_nums)):
                  delta = rewards[t] + (1 - dones[t]) * self.gamma * vs[t + 1] - vs[t]
                  advantages[t] = last_gae_lam = delta + (1 - dones[t]) * self.gamma * self.gae_lam * last_gae_lam
              returns = advantages + vs[:-1]
          else:
              rewards = np.append(np.array(self.rewards[i, path_slice]), [val], axis=0)
              returns = discount_cumsum(rewards, self.gamma)[:-1]
              advantages = rewards[:-1] + self.gamma * vs[1:] - vs[:-1]

          self.returns[i, path_slice] = returns
          self.advantages[i, path_slice] = advantages
          self.start_ids[i] = self.ptr

      def sample(self, indexes):
          assert self.full, "Not enough transitions for on-policy buffer to random sample"

          env_choices, step_choices = divmod(indexes, self.n_size)

          obs_batch = sample_batch(self.observations, tuple([env_choices, step_choices]))
          act_batch = sample_batch(self.actions, tuple([env_choices, step_choices]))
          ret_batch = sample_batch(self.returns, tuple([env_choices, step_choices]))
          val_batch = sample_batch(self.values, tuple([env_choices, step_choices]))
          adv_batch = sample_batch(self.advantages, tuple([env_choices, step_choices]))
          if self.use_advnorm:
              adv_batch = (adv_batch - np.mean(adv_batch)) / (np.std(adv_batch) + 1e-8)
          aux_batch = sample_batch(self.auxiliary_infos, tuple([env_choices, step_choices]))

          return obs_batch, act_batch, ret_batch, val_batch, adv_batch, aux_batch


  class DummyOnPolicyBuffer_Atari(DummyOnPolicyBuffer):
      """
      Replay buffer for on-policy DRL algorithms and Atari tasks.

      Args:
          observation_space: the observation space of the environment.
          action_space: the action space of the environment.
          auxiliary_shape: data shape of auxiliary information (if exists).
          n_envs: number of parallel environments.
          n_size: max length of steps to store for one environment.
          use_gae: if use GAE trick.
          use_advnorm: if use Advantage normalization trick.
          gamma: discount factor.
          gae_lam: gae lambda.
      """
      def __init__(self,
                  observation_space: Space,
                  action_space: Space,
                  auxiliary_shape: Optional[dict],
                  n_envs: int,
                  n_size: int,
                  use_gae: bool = True,
                  use_advnorm: bool = True,
                  gamma: float = 0.99,
                  gae_lam: float = 0.95):
          super(DummyOnPolicyBuffer_Atari, self).__init__(observation_space, action_space, auxiliary_shape,
                                                          n_envs, n_size, use_gae, use_advnorm, gamma, gae_lam)
          self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size, np.uint8)

      def clear(self):
          self.ptr, self.size = 0, 0
          self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size, np.uint8)
          self.actions = create_memory(space2shape(self.action_space), self.n_envs, self.n_size)
          self.auxiliary_infos = create_memory(self.auxiliary_shape, self.n_envs, self.n_size)
          self.rewards = create_memory((), self.n_envs, self.n_size)
          self.returns = create_memory((), self.n_envs, self.n_size)
          self.advantages = create_memory((), self.n_envs, self.n_size)


  class DummyOffPolicyBuffer(Buffer):
      """
      Replay buffer for off-policy DRL algorithms.

      Args:
          observation_space: the observation space of the environment.
          action_space: the action space of the environment.
          auxiliary_shape: data shape of auxiliary information (if exists).
          n_envs: number of parallel environments.
          n_size: max length of steps to store for one environment.
          batch_size: batch size of transition data for a sample.
      """
      def __init__(self,
                  observation_space: Space,
                  action_space: Space,
                  auxiliary_shape: Optional[dict],
                  n_envs: int,
                  n_size: int,
                  batch_size: int):
          super(DummyOffPolicyBuffer, self).__init__(observation_space, action_space, auxiliary_shape)
          self.n_envs, self.n_size, self.batch_size = n_envs, n_size, batch_size
          self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
          self.next_observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
          self.actions = create_memory(space2shape(self.action_space), self.n_envs, self.n_size)
          self.auxiliary_infos = create_memory(self.auxiliary_shape, self.n_envs, self.n_size)
          self.rewards = create_memory((), self.n_envs, self.n_size)
          self.terminals = create_memory((), self.n_envs, self.n_size)

      def clear(self):
          self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
          self.next_observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
          self.actions = create_memory(space2shape(self.action_space), self.n_envs, self.n_size)
          self.rewards = create_memory((), self.n_envs, self.n_size)
          self.terminals = create_memory((), self.n_envs, self.n_size)

      def store(self, obs, acts, rews, terminals, next_obs):
          store_element(obs, self.observations, self.ptr)
          store_element(acts, self.actions, self.ptr)
          store_element(rews, self.rewards, self.ptr)
          store_element(terminals, self.terminals, self.ptr)
          store_element(next_obs, self.next_observations, self.ptr)
          self.ptr = (self.ptr + 1) % self.n_size
          self.size = min(self.size + 1, self.n_size)

      def sample(self):
          env_choices = np.random.choice(self.n_envs, self.batch_size)
          step_choices = np.random.choice(self.size, self.batch_size)
          obs_batch = sample_batch(self.observations, tuple([env_choices, step_choices]))
          act_batch = sample_batch(self.actions, tuple([env_choices, step_choices]))
          rew_batch = sample_batch(self.rewards, tuple([env_choices, step_choices]))
          terminal_batch = sample_batch(self.terminals, tuple([env_choices, step_choices]))
          next_batch = sample_batch(self.next_observations, tuple([env_choices, step_choices]))
          return obs_batch, act_batch, rew_batch, terminal_batch, next_batch


  class RecurrentOffPolicyBuffer(Buffer):
      """
      Replay buffer for DRQN-based algorithms.

      Args:
          observation_space: the observation space of the environment.
          action_space: the action space of the environment.
          auxiliary_shape: data shape of auxiliary information (if exists).
          n_envs: number of parallel environments.
          n_size: max length of steps to store for one environment.
          batch_size: batch size of transition data for a sample.
          episode_length: data length for an episode.
          lookup_length: the length of history data.
      """
      def __init__(self,
                  observation_space: Space,
                  action_space: Space,
                  auxiliary_shape: Optional[dict],
                  n_envs: int,
                  n_size: int,
                  batch_size: int,
                  episode_length: int,
                  lookup_length: int):
          super(RecurrentOffPolicyBuffer, self).__init__(observation_space, action_space, auxiliary_shape)
          self.n_envs, self.n_size, self.episode_length, self.batch_size = n_envs, n_size, episode_length, batch_size
          self.lookup_length = lookup_length
          self.memory = deque(maxlen=self.n_size)

      @property
      def full(self):
          return self.size >= self.n_size

      def can_sample(self):
          return self.size >= self.batch_size

      def clear(self, *args):
          self.memory = deque(maxlen=self.n_size)

      def store(self, episode):
          self.memory.append(episode)
          self.ptr = (self.ptr + 1) % self.n_size
          self.size = min(self.size + 1, self.n_size)

      def sample(self):
          obs_batch, act_batch, rew_batch, terminal_batch = [], [], [], []
          episode_choices = np.random.choice(self.memory, self.batch_size)
          length_min = self.episode_length
          for episode in episode_choices:
              length_min = min(length_min, len(episode))

          if length_min > self.lookup_length:
              for episode in episode_choices:
                  start_idx = np.random.randint(0, len(episode) - self.lookup_length + 1)
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
      """
      Prioritized Replay Buffer.

      Args:
          observation_space: the observation space of the environment.
          action_space: the action space of the environment.
          auxiliary_shape: data shape of auxiliary information (if exists).
          n_envs: number of parallel environments.
          n_size: max length of steps to store for one environment.
          batch_size: batch size of transition data for a sample.
          alpha: prioritized factor.
      """
      def __init__(self,
                  observation_space: Space,
                  action_space: Space,
                  auxiliary_shape: Optional[dict],
                  n_envs: int,
                  n_size: int,
                  batch_size: int,
                  alpha: float = 0.6):
          super(PerOffPolicyBuffer, self).__init__(observation_space, action_space, auxiliary_shape)
          self.n_envs, self.n_size, self.batch_size = n_envs, n_size, batch_size
          self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
          self.next_observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
          self.actions = create_memory(space2shape(self.action_space), self.n_envs, self.n_size)
          self.rewards = create_memory((), self.n_envs, self.n_size)
          self.terminals = create_memory((), self.n_envs, self.n_size)

          self._alpha = alpha

          # set segment tree size
          it_capacity = 1
          while it_capacity < self.n_size:
              it_capacity *= 2

          # init segment tree
          self._it_sum = []
          self._it_min = []
          for _ in range(n_envs):
              self._it_sum.append(SumSegmentTree(it_capacity))
              self._it_min.append(MinSegmentTree(it_capacity))
          self._max_priority = np.ones((n_envs))

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
          self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
          self.next_observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
          self.actions = create_memory(space2shape(self.action_space), self.n_envs, self.n_size)
          self.rewards = create_memory((), self.n_envs, self.n_size)
          self.terminals = create_memory((), self.n_envs, self.n_size)
          self._it_sum = []
          self._it_min = []

      def store(self, obs, acts, rews, terminals, next_obs):
          store_element(obs, self.observations, self.ptr)
          store_element(acts, self.actions, self.ptr)
          store_element(rews, self.rewards, self.ptr)
          store_element(terminals, self.terminals, self.ptr)
          store_element(next_obs, self.next_observations, self.ptr)

          # prioritized process
          for i in range(self.n_envs):
              self._it_sum[i][self.ptr] = self._max_priority[i] ** self._alpha
              self._it_min[i][self.ptr] = self._max_priority[i] ** self._alpha

          self.ptr = (self.ptr + 1) % self.n_size
          self.size = min(self.size + 1, self.n_size)

      def sample(self, beta):
          env_choices = np.array(range(self.n_envs)).repeat(int(self.batch_size / self.n_envs))
          step_choices = np.zeros((self.n_envs, int(self.batch_size / self.n_envs)))
          weights = np.zeros((self.n_envs, int(self.batch_size / self.n_envs)))

          assert beta > 0

          for i in range(self.n_envs):
              idxes = self._sample_proportional(i, int(self.batch_size / self.n_envs))

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
          priorities = priorities.reshape((self.n_envs, int(self.batch_size / self.n_envs)))
          for i in range(self.n_envs):
              for idx, priority in zip(idxes[i], priorities[i]):
                  if priority == 0:
                      priority += 1e-8
                  assert 0 <= idx < self.size
                  self._it_sum[i][idx] = priority ** self._alpha
                  self._it_min[i][idx] = priority ** self._alpha

                  self._max_priority[i] = max(self._max_priority[i], priority)


  class DummyOffPolicyBuffer_Atari(DummyOffPolicyBuffer):
      """
      Replay buffer for off-policy DRL algorithms and Atari tasks.

      Args:
          observation_space: the observation space of the environment.
          action_space: the action space of the environment.
          auxiliary_shape: data shape of auxiliary information (if exists).
          n_envs: number of parallel environments.
          n_size: max length of steps to store for one environment.
          batch_size: batch size of transition data for a sample.
      """
      def __init__(self,
                  observation_space: Space,
                  action_space: Space,
                  auxiliary_shape: Optional[dict],
                  n_envs: int,
                  n_size: int,
                  batch_size: int):
          super(DummyOffPolicyBuffer_Atari, self).__init__(observation_space, action_space, auxiliary_shape,
                                                          n_envs, n_size, batch_size)
          self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size, np.uint8)
          self.next_observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size, np.uint8)

      def clear(self):
          self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size, np.uint8)
          self.next_observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size, np.uint8)
          self.actions = create_memory(space2shape(self.action_space), self.n_envs, self.n_size)
          self.auxiliary_infos = create_memory(self.auxiliary_shape, self.n_envs, self.n_size)
          self.rewards = create_memory((), self.n_envs, self.n_size)
          self.terminals = create_memory((), self.n_envs, self.n_size)


