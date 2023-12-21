Memory for MARL
=========================================

This module defines different type of classes used to implement the experience replay buffer for MARL algorithms.

.. raw:: html

    <br><hr>

Basic Memory Tools for MARL
---------------------------------------------------------

.. py:class::
    xuance.common.memory_tools_marl.BaseBuffer(*args)

    The basic class of the memory for multiple agents.

    :param args: The input arguments.
    :type args: tuple

.. py:function::
    xuance.common.memory_tools_marl.BaseBuffer.full()

    Determine whether the current experience replay buffer is full.

    :return: A bool value, True means the buffer is full, False means the buffer is not full yet.
    :rtype: bool

.. py:function::
    xuance.common.memory_tools_marl.BaseBuffer.store(*args, **kwargs)

    Store new experience data to the buffer.

    :param args: The input arguments.
    :type args: tuple
    :param kwargs: The additional input arguments.
    :type kwargs: dict

.. py:function::
    xuance.common.memory_tools_marl.BaseBuffer.clear(*args)

    Clear the whole buffer.

    :param args: The input arguments.
    :type args: tuple

.. py:function::
    xuance.common.memory_tools_marl.BaseBuffer.sample(*args)

    Sample a batch of experience data from the buffer.

    :param args: The input arguments.
    :type args: tuple

.. py:function::
    xuance.common.memory_tools_marl.BaseBuffer.finish_path(*args)

    When an episode is finished, calculate the returns, advantages, and others.

    :param args: The input arguments.
    :type args: tuple

.. raw:: html

    <br><hr>


On-Policy Buffer for MARL
--------------------------------------------------

.. py:class::
    xuance.common.memory_tools_marl.MARL_OnPolicyBuffer(n_agents, state_space, obs_space, act_space, rew_space, done_space, n_envs, buffer_size,
                    use_gae, use_advnorm, gamma, gae_lam, **kwargs)

    Replay buffer for on-policy MARL algorithms.

    :param n_agents: The number of agents.
    :type n_agents: int
    :param state_space: global state space.
    :type state_space: tuple, list
    :param obs_space: observation space for one agent (suppose same obs space for group agents).
    :type obs_space: tuple, list
    :param act_space: action space for one agent (suppose same actions space for group agents).
    :type act_space: tuple, list
    :param rew_space: reward space.
    :type rew_space: tuple, list
    :param done_space: terminal variable space.
    :type done_space: tuple, list
    :param n_envs: number of parallel environments.
    :type n_envs: int
    :param buffer_size: buffer size of transition data for one environment.
    :type buffer_size: int
    :param use_gae: whether to use GAE trick.
    :type use_gae: bool
    :param use_advnorm: whether to use Advantage normalization trick.
    :type use_advnorm: bool
    :param gamma: discount factor.
    :type gamma: float
    :param gae_lam: gae lambda.
    :type gae_lam: float
    :param kwargs: Other arguments.
    :type kwargs: dict

.. py:function::
    xuance.common.memory_tools_marl.MARL_OnPolicyBuffer.clear()

    Clear the whole buffer.

.. py:function::
    xuance.common.memory_tools_marl.MARL_OnPolicyBuffer.store(step_data)

    Store one-step transition data, 
    including observations, actions, rewars, values, terminal variables, and auxiliary informations, 
    into the buffer.

    :param step_data: One-step data to be stored.
    :type step_data: dict

.. py:function::
    xuance.common.memory_tools_marl.MARL_OnPolicyBuffer.finish_path(value, i_env, value_normalizer=None)

    When an episode is finished, calculate the returns, advantages, and others.

    :param value: The values for the final state.
    :type value: np.ndarray
    :param i_env: The index of the environment that is terminated.
    :type i_env: int
    :param value_normalizer: The function handle that normalizes the values.

.. py:function::
    xuance.common.memory_tools_marl.MARL_OnPolicyBuffer.sample(indexes)

    Sample a batch of experience data from the buffer.

    :param indexes: The indexes of the data in the buffer.
    :type indexes: np.ndarray
    :return: a dict variable that contains a batch of sampled data.
    :rtype: dict


.. py:class::
    xuance.common.memory_tools_marl.MARL_OnPolicyBuffer_RNN(n_agents, state_space, obs_space, act_space, rew_space, done_space, n_envs, buffer_size, use_gae, use_advnorm, gamma, gae_lam, **kwargs)
    
    Replay buffer for on-policy MARL algorithms with DRQN trick

    :param n_agents: The number of agents.
    :type n_agents: int
    :param state_space: global state space.
    :type state_space: tuple, list
    :param obs_space: observation space for one agent (suppose same obs space for group agents).
    :type obs_space: tuple, list
    :param act_space: action space for one agent (suppose same actions space for group agents).
    :type act_space: tuple, list
    :param rew_space: reward space.
    :type rew_space: tuple, list
    :param done_space: terminal variable space.
    :type done_space: tuple, list
    :param n_envs: number of parallel environments.
    :type n_envs: int
    :param buffer_size: buffer size of transition data for one environment.
    :type buffer_size: int
    :param use_gae: whether to use GAE trick.
    :type use_gae: bool
    :param use_advnorm: whether to use Advantage normalization trick.
    :type use_advnorm: bool
    :param gamma: discount factor.
    :type gamma: float
    :param gae_lam: gae lambda.
    :type gae_lam: float
    :param kwargs: Other arguments.
    :type kwargs: dict

.. py:function::
    xuance.common.memory_tools_marl.MARL_OnPolicyBuffer_RNN.full()

    Determine whether the buffer is full.

    :return: a bool value, True means the buffer is full, False means the buffer is not full yet.
    :rtype: bool

.. py:function::
    xuance.common.memory_tools_marl.MARL_OnPolicyBuffer_RNN.clear()

    Clear the whole buffer.

.. py:function::
    xuance.common.memory_tools_marl.MARL_OnPolicyBuffer_RNN.clear_episodes()

    Clear the current episode data for current n environments.

.. py:function::
    xuance.common.memory_tools_marl.MARL_OnPolicyBuffer_RNN.store_transitions(t_envs, *transition_data)

    Store one-step transition data to the episode buffer.

    :param t_envs: The time step to store the data.
    :type t_envs: list
    :param transition_data: The one-step transition data.
    :type transition_data: dict

.. py:function::
    xuance.common.memory_tools_marl.MARL_OnPolicyBuffer_RNN.store_episodes()

    Store episode data that is terminated for n environments.

.. py:function::
    xuance.common.memory_tools_marl.MARL_OnPolicyBuffer_RNN.finish_path(i_env, next_t, *terminal_data, value_next=None, value_normalizer=None)

    When an episode is finished, calculate the returns, advantages, and others.

    :param i_env: The index of the environment that is terminated.
    :type i_env: int
    :param next_t: The next time step for terminated state.
    :type next_t: int
    :param terminal_data: The terminal data, includes terminal observations, actions, rewards, etc.
    :type terminal_data: dict
    :param value_next: The values of the terminal observations or states.
    :type value_next: np.ndarray
    :param value_normalizer: The function handle that normalizes the values.

.. py:function::
    xuance.common.memory_tools_marl.MARL_OnPolicyBuffer_RNN.sample(indexes)

    Sample a batch of experience data from the buffer.

    :param indexes: The indexes of the episodes in the buffer.
    :type indexes: list
    :return: A dict of data that includes the episodes data for training.
    :rtype: dict


.. py:class::
    xuance.common.memory_tools_marl.MeanField_OnPolicyBuffer(n_agents, state_space, obs_space, act_space, rew_space, done_space, n_envs,
                 n_size, use_gae, use_advnorm, gamma, gae_lam, **kwargs)
                
    Replay buffer for on-policy Mean-Field MARL algorithms (Mean-Field Actor-Critic).

    :param n_agents: The number of agents.
    :type n_agents: int
    :param state_space: The data shape of the global state.
    :type state_space: tuple, list
    :param obs_space: The data shape of the local observations.
    :type obs_space: tuple, list
    :param act_space: The data shape of the actions.
    :type act_space: tuple, list
    :param rew_space: The data shape of the rewards.
    :type rew_space: tuple, list
    :param done_space: The data shape of the terminal variable.
    :type done_space: tuple, list
    :param n_envs: The number of agents.
    :type n_envs: int
    :param n_size: The size of the episodes that will be stored in this buffer.
    :type n_size: int
    :param use_gae: whether to use GAE trick.
    :type use_gae: bool
    :param use_advnorm: whether to use Advantage normalization trick.
    :type use_advnorm: bool
    :param gamma: discount factor.
    :type gamma: float
    :param gae_lam: gae lambda.
    :type gae_lam: float
    :param kwags: Other arguments.
    :type n_actions: dict

.. py:function::
    xuance.common.memory_tools_marl.MeanField_OnPolicyBuffer.clear()

    Clear the whole buffer.

.. py:function::
    xuance.common.memory_tools_marl.MeanField_OnPolicyBuffer.finish_ac_path(value, i_env)

    When an episode is finished, calculate the returns, advantages, and others.

    :param value: The values for the final state.
    :type value: np.ndarray
    :param i_env: The index of the environment that is terminated.
    :type i_env: int


.. py:class::
    xuance.common.memory_tools_marl.COMA_Buffer(n_agents, state_space, obs_space, act_space, rew_space, done_space, n_envs,
                    buffer_size, use_gae, use_advnorm, gamma, gae_lam, **kwargs)

    Replay buffer for Counterfactual Multi-Agent Policy Gradient (COMA). 

    :param n_agents: The number of agents.
    :type n_agents: int
    :param state_space: global state space.
    :type state_space: tuple, list
    :param obs_space: observation space for one agent (suppose same obs space for group agents).
    :type obs_space: tuple, list
    :param act_space: action space for one agent (suppose same actions space for group agents).
    :type act_space: tuple, list
    :param rew_space: reward space.
    :type rew_space: tuple, list
    :param done_space: terminal variable space.
    :type done_space: tuple, list
    :param n_envs: number of parallel environments.
    :type n_envs: int
    :param buffer_size: buffer size of transition data for one environment.
    :type buffer_size: int
    :param use_gae: whether to use GAE trick.
    :type use_gae: bool
    :param use_advnorm: whether to use Advantage normalization trick.
    :type use_advnorm: bool
    :param gamma: discount factor.
    :type gamma: float
    :param gae_lam: gae lambda.
    :type gae_lam: float
    :param kwargs: Other arguments.
    :type kwargs: dict

.. py:function::
    xuance.common.memory_tools_marl.COMA_Buffer.clear()

    Clear the whole buffer.

.. py:function::
    xuance.common.memory_tools_marl.COMA_Buffer.finish_path(value, i_env, value_normalizer)

    When an episode is finished, calculate the returns, advantages, and others.

    :param value: The values for the final state.
    :type value: np.ndarray
    :param i_env: The index of the environment that is terminated.
    :type i_env: int
    :param value_normalizer: The function handle that normalizes the values.

.. py:class::
    xuance.common.memory_tools_marl.COMA_Buffer_RNN(n_agents, state_space, obs_space, act_space, rew_space, done_space, n_envs,
                    buffer_size, use_gae, use_advnorm, gamma, gae_lam, **kwargs)
    
    Replay buffer for COMA algorithm with DRQN trick

    :param n_agents: The number of agents.
    :type n_agents: int
    :param state_space: global state space.
    :type state_space: tuple, list
    :param obs_space: observation space for one agent (suppose same obs space for group agents).
    :type obs_space: tuple, list
    :param act_space: action space for one agent (suppose same actions space for group agents).
    :type act_space: tuple, list
    :param rew_space: reward space.
    :type rew_space: tuple, list
    :param done_space: terminal variable space.
    :type done_space: tuple, list
    :param n_envs: number of parallel environments.
    :type n_envs: int
    :param buffer_size: buffer size of transition data for one environment.
    :type buffer_size: int
    :param use_gae: whether to use GAE trick.
    :type use_gae: bool
    :param use_advnorm: whether to use Advantage normalization trick.
    :type use_advnorm: bool
    :param gamma: discount factor.
    :type gamma: float
    :param gae_lam: gae lambda.
    :type gae_lam: float
    :param kwargs: Other arguments.
    :type kwargs: dict

.. py:function::
    xuance.common.memory_tools_marl.COMA_Buffer_RNN.clear()

    Clear the whole buffer.

.. py:function::
    xuance.common.memory_tools_marl.COMA_Buffer_RNN.clear_episodes()

    Clear the current episode data for current n environments.

.. py:function::
    xuance.common.memory_tools_marl.COMA_Buffer_RNN.store_transitions(t_envs, *transition_data)

    Store one-step transition data to the episode buffer.

    :param t_envs: The time step to store the data.
    :type t_envs: list
    :param transition_data: The one-step transition data.
    :type transition_data: dict

.. py:function::
    xuance.common.memory_tools_marl.COMA_Buffer_RNN.finish_path(i_env, next_t, *terminal_data, value_next=None, value_normalizer=None)

    When an episode is finished, calculate the returns, advantages, and others.

    :param i_env: The index of the environment that is terminated.
    :type i_env: int
    :param next_t: The next time step for terminated state.
    :type next_t: int
    :param terminal_data: The terminal data, includes terminal observations, actions, rewards, etc.
    :type terminal_data: dict
    :param value_next: The values of the terminal observations or states.
    :type value_next: np.ndarray
    :param value_normalizer: The function handle that normalizes the values.


.. raw:: html

    <br><hr>

Off-Policy Buffer for MARL
--------------------------------------------------

.. py:class::
    xuance.common.memory_tools_marl.MARL_OffPolicyBuffer(n_agents, state_space, obs_space, act_space, rew_space, done_space,
                 n_envs, buffer_size, batch_size, **kwargs)

    Replay buffer for off-policy MARL algorithms

    :param n_agents: number of agents.
    :type n_agents: int
    :param state_space: global state shape.
    :type state_space: tuple, list
    :param obs_space: observation shape for one agent (suppose same obs space for group agents).
    :type obs_space: tuple, list
    :param act_space: action shape for one agent (suppose same actions space for group agents).
    :type act_space: tuple, list
    :param rew_space: reward shape.
    :type rew_space: tuple, list
    :param done_space: terminal variable shape.
    :type done_space: tuple, list
    :param n_envs: number of parallel environments.
    :type n_envs: int
    :param buffer_size: buffer size for one environment.
    :type buffer_size: int
    :param batch_size: batch size of transition data for a sample.
    :type batch_size: int
    :param kwargs: other arguments.
    :type kwargs: dict

.. py:function::
    xuance.common.memory_tools_marl.MARL_OffPolicyBuffer.clear()

    Clear the whole buffer.

.. py:function::
    xuance.common.memory_tools_marl.MARL_OffPolicyBuffer.store(step_data)

    Store one-step transition data, including observations, actions, rewars, values, terminal variables, and auxiliary informations, into the buffer.

    :param step_data: One-step data to be stored.
    :type step_data: dict

.. py:function::
    xuance.common.memory_tools_marl.MARL_OffPolicyBuffer.sample()

    Sample a batch of experience data from the buffer.

    :return: a dict variable that contains a batch of sampled data.
    :rtype: dict

.. py:class::
    xuance.common.memory_tools_marl.MARL_OffPolicyBuffer_RNN(n_agents, state_space, obs_space, act_space, rew_space, done_space,
                    n_envs, buffer_size, batch_size, **kwargs)
    
    Replay buffer for off-policy MARL algorithms with DRQN trick.

    :param n_agents: number of agents.
    :type n_agents: int
    :param state_space: global state shape.
    :type state_space: tuple, list
    :param obs_space: observation shape for one agent (suppose same obs space for group agents).
    :type obs_space: tuple, list
    :param act_space: action shape for one agent (suppose same actions space for group agents).
    :type act_space: tuple, list
    :param rew_space: reward shape.
    :type rew_space: tuple, list
    :param done_space: terminal variable shape.
    :type done_space: tuple, list
    :param n_envs: number of parallel environments.
    :type n_envs: int
    :param buffer_size: buffer size for one environment.
    :type buffer_size: int
    :param batch_size: batch size of transition data for a sample.
    :type batch_size: int
    :param kwargs: other arguments.
    :type kwargs: dict

.. py:function::
    xuance.common.memory_tools_marl.MARL_OffPolicyBuffer_RNN.clear()

    Clear the whole buffer.

.. py:function::
    xuance.common.memory_tools_marl.MARL_OffPolicyBuffer_RNN.clear_episodes()

    Clear the current episode data for current n environments.

.. py:function::
    xuance.common.memory_tools_marl.MARL_OffPolicyBuffer_RNN.store_transitions(t_envs, *transition_data)

    Store one-step transition data to the episode buffer.

    :param list: The time step to store the data.
    :type t_envs: dict
    :param transition_data: The one-step transition data.
    :type transition_data: dict

.. py:function::
    xuance.common.memory_tools_marl.MARL_OffPolicyBuffer_RNN.store_episodes()

    Store episode data that is terminated for n environments.

.. py:function::
    xuance.common.memory_tools_marl.MARL_OffPolicyBuffer_RNN.finish_path(i_env, next_t, *terminal_data)

    When an episode is finished, calculate the returns, advantages, and others.

    :param i_env: The index of the environment that is terminated.
    :type i_env: int
    :param next_t: The next time step for terminated state.
    :type next_t: int
    :param terminal_data: The terminal data, includes terminal observations, actions, rewards, etc.
    :type terminal_data: dict

.. py:function::
    xuance.common.memory_tools_marl.MARL_OffPolicyBuffer_RNN.sample()

    Sample a batch of experience data from the buffer.

    :return: A dict of data that includes the episodes data for training.
    :rtype: dict

.. py:class::
    xuance.common.memory_tools_marl.MeanField_OffPolicyBuffer(n_agents, state_space, obs_space, act_space, prob_shape, rew_space, done_space,
                    n_envs, buffer_size, batch_size)
                
    Replay buffer for Mean-Field Q algorithms.

    :param n_agents: The number of agents.
    :type n_agents: int
    :param state_space: global state shape.
    :type state_space: tuple, list
    :param obs_space: observation shape for one agent (suppose same obs space for group agents).
    :type obs_space: tuple, list
    :param act_space: action shape for one agent (suppose same actions space for group agents).
    :type act_space: tuple, list
    :param prob_shape: the data shape of the action probabilities.
    :type prob_shape: tuple, list
    :param rew_space: reward shape.
    :type rew_space: tuple, list
    :param done_space: terminal variable shape.
    :type done_space: tuple, list
    :param n_envs: number of parallel environments.
    :type n_envs: int
    :param buffer_size: buffer size for one environment.
    :type buffer_size: int
    :param batch_size: batch size of transition data for a sample.
    :type batch_size: int

.. py:function::
    xuance.common.memory_tools_marl.MeanField_OffPolicyBuffer.clear()

    Clear the whole buffer.

.. py:function::
    xuance.common.memory_tools_marl.MeanField_OffPolicyBuffer.sample()

    Sample a batch of experience data from the buffer.

    :return: A dict of data that includes the episodes data for training.
    :rtype: dict


.. raw:: html

    <br><hr>


Source Code
-----------------

.. code-block:: python

    import numpy as np
    from abc import ABC, abstractmethod


    class BaseBuffer(ABC):
        """
        Basic buffer for MARL algorithms.
        """

        def __init__(self, *args):
            self.n_agents, self.state_space, self.obs_space, self.act_space, self.rew_space, self.done_space, self.n_envs, self.buffer_size = args
            self.ptr = 0  # last data pointer
            self.size = 0  # current buffer size

        @property
        def full(self):
            return self.size >= self.n_size

        @abstractmethod
        def store(self, *args, **kwargs):
            raise NotImplementedError

        @abstractmethod
        def clear(self, *args):
            raise NotImplementedError

        @abstractmethod
        def sample(self, *args):
            raise NotImplementedError

        def finish_path(self, *args):
            return


    class MARL_OnPolicyBuffer(BaseBuffer):
        """
        Replay buffer for on-policy MARL algorithms.

        Args:
            n_agents: number of agents.
            state_space: global state space, type: Discrete, Box.
            obs_space: observation space for one agent (suppose same obs space for group agents).
            act_space: action space for one agent (suppose same actions space for group agents).
            rew_space: reward space.
            done_space: terminal variable space.
            n_envs: number of parallel environments.
            buffer_size: buffer size of transition data for one environment.
            use_gae: whether to use GAE trick.
            use_advnorm: whether to use Advantage normalization trick.
            gamma: discount factor.
            gae_lam: gae lambda.
        """

        def __init__(self, n_agents, state_space, obs_space, act_space, rew_space, done_space, n_envs, buffer_size,
                    use_gae, use_advnorm, gamma, gae_lam, **kwargs):
            super(MARL_OnPolicyBuffer, self).__init__(n_agents, state_space, obs_space, act_space, rew_space, done_space,
                                                    n_envs, buffer_size)
            self.n_size = buffer_size // self.n_envs
            self.use_gae = use_gae
            self.use_advantage_norm = use_advnorm
            self.gamma, self.gae_lambda = gamma, gae_lam
            self.data, self.start_ids = {}, None
            self.clear()
            self.keys = self.data.keys()
            self.data_shapes = {k: self.data[k].shape for k in self.keys}

        def clear(self):
            self.data = {
                'obs': np.zeros((self.n_envs, self.n_size, self.n_agents) + self.obs_space).astype(np.float32),
                'actions': np.zeros((self.n_envs, self.n_size, self.n_agents) + self.act_space).astype(np.float32),
                'rewards': np.zeros((self.n_envs, self.n_size,) + self.rew_space).astype(np.float32),
                'returns': np.zeros((self.n_envs, self.n_size,) + self.rew_space).astype(np.float32),
                'values': np.zeros((self.n_envs, self.n_size, self.n_agents, 1)).astype(np.float32),
                'log_pi_old': np.zeros((self.n_envs, self.n_size, self.n_agents,)).astype(np.float32),
                'advantages': np.zeros((self.n_envs, self.n_size,) + self.rew_space).astype(np.float32),
                'terminals': np.zeros((self.n_envs, self.n_size,) + self.done_space).astype(np.bool),
                'agent_mask': np.ones((self.n_envs, self.n_size, self.n_agents)).astype(np.bool),
            }
            if self.state_space is not None:
                self.data.update({'state': np.zeros((self.n_envs, self.n_size,) + self.state_space).astype(np.float32)})
            self.ptr, self.size = 0, 0
            self.start_ids = np.zeros(self.n_envs, np.int64)  # the start index of the last episode for each env.

        def store(self, step_data):
            step_data_keys = step_data.keys()
            for k in self.keys:
                if k == "advantages":
                    continue
                if k in step_data_keys:
                    self.data[k][:, self.ptr] = step_data[k]
            self.ptr = (self.ptr + 1) % self.n_size
            self.size = min(self.size + 1, self.n_size)

        def finish_path(self, value, i_env, value_normalizer=None):  # when an episode is finished
            if self.size == 0:
                return
            if self.full:
                path_slice = np.arange(self.start_ids[i_env], self.n_size).astype(np.int32)
            else:
                path_slice = np.arange(self.start_ids[i_env], self.ptr).astype(np.int32)

            # calculate advantages and returns
            rewards = np.array(self.data['rewards'][i_env, path_slice])
            vs = np.append(np.array(self.data['values'][i_env, path_slice]), [value], axis=0)
            dones = np.array(self.data['terminals'][i_env, path_slice])[:, :, None]
            returns = np.zeros_like(rewards)
            last_gae_lam = 0
            step_nums = len(path_slice)

            if self.use_gae:
                for t in reversed(range(step_nums)):
                    delta = rewards[t] + (1 - dones[t]) * self.gamma * vs[t + 1] - vs[t]
                    last_gae_lam = delta + (1 - dones[t]) * self.gamma * self.gae_lambda * last_gae_lam
                    returns[t] = last_gae_lam + vs[t]
            else:
                returns = np.append(returns, [value], axis=0)
                for t in reversed(range(step_nums)):
                    returns[t] = rewards[t] + (1 - dones[t]) * self.gamma * returns[t + 1]

            advantages = returns - vs[:-1]
            self.data['returns'][i_env, path_slice] = returns
            self.data['advantages'][i_env, path_slice] = advantages
            self.start_ids[i_env] = self.ptr

        def sample(self, indexes):
            assert self.full, "Not enough transitions for on-policy buffer to random sample"

            samples = {}
            env_choices, step_choices = divmod(indexes, self.n_size)
            for k in self.keys:
                if k == "advantages":
                    adv_batch = self.data[k][env_choices, step_choices]
                    if self.use_advantage_norm:
                        adv_batch = (adv_batch - np.mean(adv_batch)) / (np.std(adv_batch) + 1e-8)
                    samples[k] = adv_batch
                else:
                    samples[k] = self.data[k][env_choices, step_choices]
            return samples


    class MARL_OnPolicyBuffer_RNN(MARL_OnPolicyBuffer):
        """
        Replay buffer for on-policy MARL algorithms with DRQN trick.

        Args:
            n_agents: number of agents.
            state_space: global state space, type: Discrete, Box.
            obs_space: observation space for one agent (suppose same obs space for group agents).
            act_space: action space for one agent (suppose same actions space for group agents).
            rew_space: reward space.
            done_space: terminal variable space.
            n_envs: number of parallel environments.
            buffer_size: buffer size of trajectory data for one environment.
            use_gae: whether to use GAE trick.
            use_advnorm: whether to use Advantage normalization trick.
            gamma: discount factor.
            gae_lam: gae lambda.
            max_episode_length: maximum length of data for one episode trajectory.
        """

        def __init__(self, n_agents, state_space, obs_space, act_space, rew_space, done_space, n_envs, buffer_size,
                    use_gae, use_advnorm, gamma, gae_lam, **kwargs):
            self.max_eps_len = kwargs['max_episode_length']
            self.dim_act = kwargs['dim_act']
            super(MARL_OnPolicyBuffer_RNN, self).__init__(n_agents, state_space, obs_space, act_space, rew_space,
                                                        done_space, n_envs, buffer_size,
                                                        use_gae, use_advnorm, gamma, gae_lam,
                                                        **kwargs)
            self.episode_data = {}
            self.clear_episodes()

        @property
        def full(self):
            return self.size >= self.buffer_size

        def clear(self):
            self.data = {
                'obs': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len + 1) + self.obs_space, np.float32),
                'actions': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len) + self.act_space, np.float32),
                'rewards': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len) + self.rew_space, np.float32),
                'returns': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len) + self.rew_space, np.float32),
                'values': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len) + self.rew_space, np.float32),
                'advantages': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len) + self.rew_space, np.float32),
                'log_pi_old': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len,), np.float32),
                'terminals': np.zeros((self.buffer_size, self.max_eps_len) + self.done_space, np.bool),
                'avail_actions': np.ones((self.buffer_size, self.n_agents, self.max_eps_len + 1, self.dim_act), np.bool),
                'filled': np.zeros((self.buffer_size, self.max_eps_len, 1), np.bool)
            }
            if self.state_space is not None:
                self.data.update({
                    'state': np.zeros((self.buffer_size, self.max_eps_len + 1) + self.state_space, np.float32)
                })
            self.ptr, self.size = 0, 0

        def clear_episodes(self):
            self.episode_data = {
                'obs': np.zeros((self.n_envs, self.n_agents, self.max_eps_len + 1) + self.obs_space, dtype=np.float32),
                'actions': np.zeros((self.n_envs, self.n_agents, self.max_eps_len) + self.act_space, dtype=np.float32),
                'rewards': np.zeros((self.n_envs, self.n_agents, self.max_eps_len) + self.rew_space, dtype=np.float32),
                'returns': np.zeros((self.n_envs, self.n_agents, self.max_eps_len) + self.rew_space, np.float32),
                'values': np.zeros((self.n_envs, self.n_agents, self.max_eps_len) + self.rew_space, np.float32),
                'advantages': np.zeros((self.n_envs, self.n_agents, self.max_eps_len) + self.rew_space, np.float32),
                'log_pi_old': np.zeros((self.n_envs, self.n_agents, self.max_eps_len,), np.float32),
                'terminals': np.zeros((self.n_envs, self.max_eps_len) + self.done_space, dtype=np.bool),
                'avail_actions': np.ones((self.n_envs, self.n_agents, self.max_eps_len + 1, self.dim_act), dtype=np.bool),
                'filled': np.zeros((self.n_envs, self.max_eps_len, 1), dtype=np.bool),
            }
            if self.state_space is not None:
                self.episode_data.update({
                    'state': np.zeros((self.n_envs, self.max_eps_len + 1) + self.state_space, dtype=np.float32),
                })

        def store_transitions(self, t_envs, *transition_data):
            obs_n, actions_dict, state, rewards, terminated, avail_actions = transition_data
            self.episode_data['obs'][:, :, t_envs] = obs_n
            self.episode_data['actions'][:, :, t_envs] = actions_dict['actions_n']
            self.episode_data['rewards'][:, :, t_envs] = rewards
            self.episode_data['values'][:, :, t_envs] = actions_dict['values']
            self.episode_data['log_pi_old'][:, :, t_envs] = actions_dict['log_pi']
            self.episode_data['terminals'][:, t_envs] = terminated
            self.episode_data['avail_actions'][:, :, t_envs] = avail_actions
            if self.state_space is not None:
                self.episode_data['state'][:, t_envs] = state

        def store_episodes(self):
            episode_data_keys = self.episode_data.keys()
            for i_env in range(self.n_envs):
                for k in self.keys:
                    if k in episode_data_keys:
                        self.data[k][self.ptr] = self.episode_data[k][i_env].copy()
                self.ptr = (self.ptr + 1) % self.buffer_size
                self.size = min(self.size + 1, self.buffer_size)
            self.clear_episodes()

        def finish_path(self, i_env, next_t, *terminal_data, value_next=None, value_normalizer=None):
            obs_next, state_next, available_actions, filled = terminal_data
            self.episode_data['obs'][i_env, :, next_t] = obs_next[i_env]
            self.episode_data['state'][i_env, next_t] = state_next[i_env]
            self.episode_data['avail_actions'][i_env, :, next_t] = available_actions[i_env]
            self.episode_data['filled'][i_env] = filled[i_env]

            """ when an episode is finished. """
            if next_t > self.max_eps_len:
                path_slice = np.arange(0, self.max_eps_len).astype(np.int32)
            else:
                path_slice = np.arange(0, next_t).astype(np.int32)

            # calculate advantages and returns
            rewards = np.array(self.episode_data['rewards'][i_env, :, path_slice])
            vs = np.append(np.array(self.episode_data['values'][i_env, :, path_slice]),
                        [value_next.reshape(self.n_agents, 1)],
                        axis=0)
            dones = np.array(self.episode_data['terminals'][i_env, path_slice])[:, :, None]
            returns = np.zeros_like(rewards)
            last_gae_lam = 0
            step_nums = len(path_slice)
            use_value_norm = False if (value_normalizer is None) else True

            if self.use_gae:
                for t in reversed(range(step_nums)):
                    if use_value_norm:
                        vs_t, vs_next = value_normalizer.denormalize(vs[t]), value_normalizer.denormalize(vs[t + 1])
                    else:
                        vs_t, vs_next = vs[t], vs[t + 1]
                    delta = rewards[t] + (1 - dones[t]) * self.gamma * vs_next - vs_t
                    last_gae_lam = delta + (1 - dones[t]) * self.gamma * self.gae_lambda * last_gae_lam
                    returns[t] = last_gae_lam + vs_t
                advantages = returns - value_normalizer.denormalize(vs[:-1]) if use_value_norm else returns - vs[:-1]
            else:
                returns = np.append(returns, [value_next.reshape(self.n_agents, 1)], axis=0)
                for t in reversed(range(step_nums)):
                    returns[t] = rewards[t] + (1 - dones[t]) * self.gamma * returns[t + 1]
                advantages = returns - value_normalizer.denormalize(vs) if use_value_norm else returns - vs
                advantages = advantages[:-1]

            self.episode_data['returns'][i_env, :, path_slice] = returns
            self.episode_data['advantages'][i_env, :, path_slice] = advantages

        def sample(self, indexes):
            assert self.full, "Not enough transitions for on-policy buffer to random sample"
            samples = {}
            filled_batch = self.data['filled'][indexes]
            samples['filled'] = filled_batch
            for k in self.keys:
                if k == "filled":
                    continue
                if k == "advantages":
                    adv_batch = self.data[k][indexes]
                    if self.use_advantage_norm:
                        adv_batch_copy = adv_batch.copy()
                        filled_batch_n = filled_batch[:, None, :, :].repeat(self.n_agents, axis=1)
                        adv_batch_copy[filled_batch_n == 0] = np.nan
                        adv_batch = (adv_batch - np.nanmean(adv_batch_copy)) / (np.nanstd(adv_batch_copy) + 1e-8)
                    samples[k] = adv_batch
                else:
                    samples[k] = self.data[k][indexes]
            return samples


    class MeanField_OnPolicyBuffer(MARL_OnPolicyBuffer):
        """
        Replay buffer for on-policy Mean-Field MARL algorithms (Mean-Field Actor-Critic).

        Args:
            n_agents: number of agents.
            state_space: global state space, type: Discrete, Box.
            obs_space: observation space for one agent (suppose same obs space for group agents).
            act_space: action space for one agent (suppose same actions space for group agents).
            rew_space: reward space.
            done_space: terminal variable space.
            n_envs: number of parallel environments.
            n_size: buffer size of trajectory data for one environment.
            use_gae: whether to use GAE trick.
            use_advnorm: whether to use Advantage normalization trick.
            gamma: discount factor.
            gae_lam: gae lambda.
            kwargs: the other arguments.
        """

        def __init__(self, n_agents, state_space, obs_space, act_space, rew_space, done_space, n_envs,
                    n_size, use_gae, use_advnorm, gamma, gae_lam, **kwargs):
            self.prob_space = kwargs['prob_space']
            super(MeanField_OnPolicyBuffer, self).__init__(n_agents, state_space, obs_space, act_space, rew_space,
                                                        done_space, n_envs, n_size, use_gae, use_advnorm, gamma, gae_lam,
                                                        **kwargs)

        def clear(self):
            self.data.update({
                'obs': np.zeros((self.n_envs, self.n_size, self.n_agents) + self.obs_space).astype(np.float32),
                'actions': np.zeros((self.n_envs, self.n_size, self.n_agents) + self.act_space).astype(np.float32),
                'act_mean': np.zeros((self.n_envs, self.n_size,) + self.prob_space).astype(np.float32),
                'rewards': np.zeros((self.n_envs, self.n_size,) + self.rew_space).astype(np.float32),
                'returns': np.zeros((self.n_envs, self.n_size,) + self.rew_space).astype(np.float32),
                'values': np.zeros((self.n_envs, self.n_size, self.n_agents, 1)).astype(np.float32),
                'advantages': np.zeros((self.n_envs, self.n_size,) + self.rew_space).astype(np.float32),
                'terminals': np.zeros((self.n_envs, self.n_size,) + self.done_space).astype(np.bool),
                'agent_mask': np.ones((self.n_envs, self.n_size, self.n_agents)).astype(np.bool),
            })
            if self.state_space is not None:
                self.data.update({'state': np.zeros((self.n_envs, self.n_size,) + self.state_space).astype(np.float32)})
            self.ptr = 0  # current pointer
            self.size = 0  # current buffer size
            self.start_ids = np.zeros(self.n_envs)

        def finish_ac_path(self, value, i_env):  # when an episode is finished
            if self.size == 0:
                return
            self.start_ids[i_env] = self.ptr


    class COMA_Buffer(MARL_OnPolicyBuffer):
        def __init__(self, n_agents, state_space, obs_space, act_space, rew_space, done_space, n_envs, buffer_size,
                    use_gae, use_advnorm, gamma, gae_lam, **kwargs):
            self.dim_act = kwargs['dim_act']
            self.td_lambda = kwargs['td_lambda']
            super(COMA_Buffer, self).__init__(n_agents, state_space, obs_space, act_space, rew_space, done_space, n_envs,
                                            buffer_size, use_gae, use_advnorm, gamma, gae_lam, **kwargs)

        def clear(self):
            self.data = {
                'obs': np.zeros((self.n_envs, self.n_size, self.n_agents) + self.obs_space).astype(np.float32),
                'actions': np.zeros((self.n_envs, self.n_size, self.n_agents) + self.act_space).astype(np.float32),
                'actions_onehot': np.zeros((self.n_envs, self.n_size, self.n_agents, self.dim_act)).astype(np.float32),
                'rewards': np.zeros((self.n_envs, self.n_size,) + self.rew_space).astype(np.float32),
                'returns': np.zeros((self.n_envs, self.n_size,) + self.rew_space).astype(np.float32),
                'values': np.zeros((self.n_envs, self.n_size, self.n_agents, 1)).astype(np.float32),
                'log_pi_old': np.zeros((self.n_envs, self.n_size, self.n_agents,)).astype(np.float32),
                'terminals': np.zeros((self.n_envs, self.n_size,) + self.done_space).astype(np.bool),
                'agent_mask': np.ones((self.n_envs, self.n_size, self.n_agents)).astype(np.bool),
            }
            if self.state_space is not None:
                self.data.update({'state': np.zeros((self.n_envs, self.n_size,) + self.state_space).astype(np.float32)})
            self.ptr, self.size = 0, 0
            self.start_ids = np.zeros(self.n_envs, np.int64)  # the start index of the last episode for each env.

        def finish_path(self, value, i_env, value_normalizer=None):  # when an episode is finished
            """
            Build td-lambda targets.
            """
            if self.size == 0:
                return
            if self.full:
                path_slice = np.arange(self.start_ids[i_env], self.n_size).astype(np.int32)
            else:
                path_slice = np.arange(self.start_ids[i_env], self.ptr).astype(np.int32)

            # calculate advantages and returns
            rewards = np.array(self.data['rewards'][i_env, path_slice])
            vs = np.append(np.array(self.data['values'][i_env, path_slice]), [value], axis=0)
            dones = np.array(self.data['terminals'][i_env, path_slice])[:, :, None]
            returns = np.zeros_like(vs)
            step_nums = len(path_slice)
            for t in reversed(range(step_nums)):
                returns[t] = self.td_lambda * self.gamma * returns[t + 1] + \
                            rewards[t] + (1 - self.td_lambda) * self.gamma * vs[t + 1] * (1 - dones[t])
            self.data['returns'][i_env, path_slice] = returns[:-1]
            self.start_ids[i_env] = self.ptr


    class COMA_Buffer_RNN(MARL_OnPolicyBuffer_RNN):
        """
        Replay buffer for on-policy MARL algorithms.

        Args:
            n_agents: number of agents.
            state_space: global state space, type: Discrete, Box.
            obs_space: observation space for one agent (suppose same obs space for group agents).
            act_space: action space for one agent (suppose same actions space for group agents).
            rew_space: reward space.
            done_space: terminal variable space.
            n_envs: number of parallel environments.
            buffer_size: buffer size of transition data for one environment.
            use_gae: whether to use GAE trick.
            use_advnorm: whether to use Advantage normalization trick.
            gamma: discount factor.
            gae_lam: gae lambda.
            **kwargs: other args.
        """
        def __init__(self, n_agents, state_space, obs_space, act_space, rew_space, done_space, n_envs, buffer_size,
                    use_gae, use_advnorm, gamma, gae_lam, **kwargs):
            self.td_lambda = kwargs['td_lambda']
            super(COMA_Buffer_RNN, self).__init__(n_agents, state_space, obs_space, act_space, rew_space, done_space,
                                                n_envs, buffer_size, use_gae, use_advnorm, gamma, gae_lam, **kwargs)

        def clear(self):
            self.data = {
                'obs': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len + 1) + self.obs_space, np.float32),
                'actions': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len) + self.act_space, np.float32),
                'actions_onehot': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len, self.dim_act)).astype(
                    np.float32),
                'rewards': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len) + self.rew_space, np.float32),
                'returns': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len) + self.rew_space, np.float32),
                'values': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len) + self.rew_space, np.float32),
                'advantages': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len) + self.rew_space, np.float32),
                'log_pi_old': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len,), np.float32),
                'terminals': np.zeros((self.buffer_size, self.max_eps_len) + self.done_space, np.bool),
                'avail_actions': np.ones((self.buffer_size, self.n_agents, self.max_eps_len + 1, self.dim_act), np.bool),
                'filled': np.zeros((self.buffer_size, self.max_eps_len, 1), np.bool)
            }
            if self.state_space is not None:
                self.data.update({'state': np.zeros(
                    (self.buffer_size, self.max_eps_len + 1) + self.state_space, np.float32)})
            self.ptr, self.size = 0, 0

        def clear_episodes(self):
            self.episode_data = {
                'obs': np.zeros((self.n_envs, self.n_agents, self.max_eps_len + 1) + self.obs_space, dtype=np.float32),
                'actions': np.zeros((self.n_envs, self.n_agents, self.max_eps_len) + self.act_space, dtype=np.float32),
                'actions_onehot': np.zeros((self.n_envs, self.n_agents, self.max_eps_len, self.dim_act), dtype=np.float32),
                'rewards': np.zeros((self.n_envs, self.n_agents, self.max_eps_len) + self.rew_space, dtype=np.float32),
                'returns': np.zeros((self.n_envs, self.n_agents, self.max_eps_len) + self.rew_space, np.float32),
                'values': np.zeros((self.n_envs, self.n_agents, self.max_eps_len) + self.rew_space, np.float32),
                'advantages': np.zeros((self.n_envs, self.n_agents, self.max_eps_len) + self.rew_space, np.float32),
                'log_pi_old': np.zeros((self.n_envs, self.n_agents, self.max_eps_len,), np.float32),
                'terminals': np.zeros((self.n_envs, self.max_eps_len) + self.done_space, dtype=np.bool),
                'avail_actions': np.ones((self.n_envs, self.n_agents, self.max_eps_len + 1, self.dim_act), dtype=np.bool),
                'filled': np.zeros((self.n_envs, self.max_eps_len, 1), dtype=np.bool),
            }
            if self.state_space is not None:
                self.episode_data.update({
                    'state': np.zeros((self.n_envs, self.max_eps_len + 1) + self.state_space, dtype=np.float32)
                })

        def store_transitions(self, t_envs, *transition_data):
            obs_n, actions_dict, state, rewards, terminated, avail_actions = transition_data
            self.episode_data['obs'][:, :, t_envs] = obs_n
            self.episode_data['actions'][:, :, t_envs] = actions_dict['actions_n']
            self.episode_data['actions_onehot'][:, :, t_envs] = actions_dict['act_n_onehot']
            self.episode_data['rewards'][:, :, t_envs] = rewards
            self.episode_data['values'][:, :, t_envs] = actions_dict['values']
            self.episode_data['log_pi_old'][:, :, t_envs] = actions_dict['log_pi']
            self.episode_data['terminals'][:, t_envs] = terminated
            self.episode_data['avail_actions'][:, :, t_envs] = avail_actions
            if self.state_space is not None:
                self.episode_data['state'][:, t_envs] = state

        def finish_path(self, i_env, next_t, *terminal_data, value_next=None, value_normalizer=None):
            obs_next, state_next, available_actions, filled = terminal_data
            self.episode_data['obs'][i_env, :, next_t] = obs_next[i_env]
            self.episode_data['state'][i_env, next_t] = state_next[i_env]
            self.episode_data['avail_actions'][i_env, :, next_t] = available_actions[i_env]
            self.episode_data['filled'][i_env] = filled[i_env]

            """
            when an episode is finished, build td-lambda targets.
            """
            if next_t > self.max_eps_len:
                path_slice = np.arange(0, self.max_eps_len).astype(np.int32)
            else:
                path_slice = np.arange(0, next_t).astype(np.int32)
            # calculate advantages and returns
            rewards = np.array(self.episode_data['rewards'][i_env, :, path_slice])
            vs = np.append(np.array(self.episode_data['values'][i_env, :, path_slice]),
                        [value_next.reshape(self.n_agents, 1)], axis=0)
            dones = np.array(self.episode_data['terminals'][i_env, path_slice])[:, :, None]
            returns = np.zeros_like(vs)
            step_nums = len(path_slice)

            for t in reversed(range(step_nums)):
                returns[t] = self.td_lambda * self.gamma * returns[t + 1] + \
                            rewards[t] + (1 - self.td_lambda) * self.gamma * vs[t + 1] * (1 - dones[t])

            self.episode_data['returns'][i_env, :, path_slice] = returns[:-1]


    class MARL_OffPolicyBuffer(BaseBuffer):
        """
        Replay buffer for off-policy MARL algorithms.

        Args:
            n_agents: number of agents.
            state_space: global state space, type: Discrete, Box.
            obs_space: observation space for one agent (suppose same obs space for group agents).
            act_space: action space for one agent (suppose same actions space for group agents).
            rew_space: reward space.
            done_space: terminal variable space.
            n_envs: number of parallel environments.
            buffer_size: buffer size for one environment.
            batch_size: batch size of transition data for a sample.
            **kwargs: other arguments.
        """

        def __init__(self, n_agents, state_space, obs_space, act_space, rew_space, done_space,
                    n_envs, buffer_size, batch_size, **kwargs):
            super(MARL_OffPolicyBuffer, self).__init__(n_agents, state_space, obs_space, act_space, rew_space, done_space,
                                                    n_envs, buffer_size)
            self.n_size = buffer_size // n_envs
            self.batch_size = batch_size
            if self.state_space is not None:
                self.store_global_state = True
            else:
                self.store_global_state = False
            self.data = {}
            self.clear()
            self.keys = self.data.keys()

        def clear(self):
            self.data = {
                'obs': np.zeros((self.n_envs, self.n_size, self.n_agents) + self.obs_space).astype(np.float32),
                'actions': np.zeros((self.n_envs, self.n_size, self.n_agents) + self.act_space).astype(np.float32),
                'obs_next': np.zeros((self.n_envs, self.n_size, self.n_agents) + self.obs_space).astype(np.float32),
                'rewards': np.zeros((self.n_envs, self.n_size) + self.rew_space).astype(np.float32),
                'terminals': np.zeros((self.n_envs, self.n_size) + self.done_space).astype(np.bool),
                'agent_mask': np.ones((self.n_envs, self.n_size, self.n_agents)).astype(np.bool)
            }
            if self.state_space is not None:
                self.data.update({'state': np.zeros((self.n_envs, self.n_size) + self.state_space).astype(np.float32),
                                'state_next': np.zeros((self.n_envs, self.n_size) + self.state_space).astype(np.float32)})
            self.ptr, self.size = 0, 0

        def store(self, step_data):
            for k in self.keys:
                self.data[k][:, self.ptr] = step_data[k]
            self.ptr = (self.ptr + 1) % self.n_size
            self.size = np.min([self.size + 1, self.n_size])

        def sample(self):
            env_choices = np.random.choice(self.n_envs, self.batch_size)
            step_choices = np.random.choice(self.size, self.batch_size)
            samples = {k: self.data[k][env_choices, step_choices] for k in self.keys}
            return samples


    class MARL_OffPolicyBuffer_RNN(MARL_OffPolicyBuffer):
        """
        Replay buffer for off-policy MARL algorithms with DRQN trick.

        Args:
            n_agents: number of agents.
            state_space: global state space, type: Discrete, Box.
            obs_space: observation space for one agent (suppose same obs space for group agents).
            act_space: action space for one agent (suppose same actions space for group agents).
            rew_space: reward space.
            done_space: terminal variable space.
            n_envs: number of parallel environments.
            buffer_size: buffer size for one environment.
            batch_size: batch size of episodes for a sample.
            kwargs: other arguments.
        """

        def __init__(self, n_agents, state_space, obs_space, act_space, rew_space, done_space,
                    n_envs, buffer_size, batch_size, **kwargs):
            self.max_eps_len = kwargs['max_episode_length']
            self.dim_act = kwargs['dim_act']
            super(MARL_OffPolicyBuffer_RNN, self).__init__(n_agents, state_space, obs_space, act_space, rew_space,
                                                        done_space, n_envs, buffer_size, batch_size)

            self.episode_data = {}
            self.clear_episodes()

        def clear(self):
            self.data = {
                'obs': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len + 1) + self.obs_space, np.float),
                'actions': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len) + self.act_space, np.float),
                'rewards': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len) + self.rew_space, np.float),
                'terminals': np.zeros((self.buffer_size, self.max_eps_len) + self.done_space, np.bool),
                'avail_actions': np.ones((self.buffer_size, self.n_agents, self.max_eps_len + 1, self.dim_act), np.bool),
                'filled': np.zeros((self.buffer_size, self.max_eps_len, 1)).astype(np.bool)
            }
            if self.state_space is not None:
                self.data.update({'state': np.zeros(
                    (self.buffer_size, self.max_eps_len + 1) + self.state_space).astype(np.float32)})
            self.ptr, self.size = 0, 0

        def clear_episodes(self):
            self.episode_data = {
                'obs': np.zeros((self.n_envs, self.n_agents, self.max_eps_len + 1) + self.obs_space, dtype=np.float32),
                'actions': np.zeros((self.n_envs, self.n_agents, self.max_eps_len) + self.act_space, dtype=np.float32),
                'rewards': np.zeros((self.n_envs, self.n_agents, self.max_eps_len) + self.rew_space, dtype=np.float32),
                'terminals': np.zeros((self.n_envs, self.max_eps_len) + self.done_space, dtype=np.bool),
                'avail_actions': np.ones((self.n_envs, self.n_agents, self.max_eps_len + 1, self.dim_act), dtype=np.bool),
                'filled': np.zeros((self.n_envs, self.max_eps_len, 1), dtype=np.bool),
            }
            if self.state_space is not None:
                self.episode_data.update({
                    'state': np.zeros((self.n_envs, self.max_eps_len + 1) + self.state_space, dtype=np.float32),
                })

        def store_transitions(self, t_envs, *transition_data):
            obs_n, actions_dict, state, rewards, terminated, avail_actions = transition_data
            self.episode_data['obs'][:, :, t_envs] = obs_n
            self.episode_data['actions'][:, :, t_envs] = actions_dict['actions_n']
            self.episode_data['rewards'][:, :, t_envs] = rewards
            self.episode_data['terminals'][:, t_envs] = terminated
            self.episode_data['avail_actions'][:, :, t_envs] = avail_actions
            if self.state_space is not None:
                self.episode_data['state'][:, t_envs] = state

        def store_episodes(self):
            for i_env in range(self.n_envs):
                for k in self.keys:
                    self.data[k][self.ptr] = self.episode_data[k][i_env].copy()
                self.ptr = (self.ptr + 1) % self.buffer_size
                self.size = np.min([self.size + 1, self.buffer_size])
            self.clear_episodes()

        def finish_path(self, i_env, next_t, *terminal_data):
            obs_next, state_next, available_actions, filled = terminal_data
            self.episode_data['obs'][i_env, :, next_t] = obs_next[i_env]
            self.episode_data['state'][i_env, next_t] = state_next[i_env]
            self.episode_data['avail_actions'][i_env, :, next_t] = available_actions[i_env]
            self.episode_data['filled'][i_env] = filled[i_env]

        def sample(self):
            sample_choices = np.random.choice(self.size, self.batch_size)
            samples = {k: self.data[k][sample_choices] for k in self.keys}
            return samples


    class MeanField_OffPolicyBuffer(MARL_OffPolicyBuffer):
        """
        Replay buffer for off-policy Mean-Field MARL algorithms (Mean-Field Q-Learning).

        Args:
            n_agents: number of agents.
            state_space: global state space, type: Discrete, Box.
            obs_space: observation space for one agent (suppose same obs space for group agents).
            act_space: action space for one agent (suppose same actions space for group agents).
            prob_shape: the data shape of the action probabilities.
            rew_space: reward space.
            done_space: terminal variable space.
            n_envs: number of parallel environments.
            buffer_size: buffer size for one environment.
            batch_size: batch size of transition data for a sample.
        """

        def __init__(self, n_agents, state_space, obs_space, act_space, prob_shape, rew_space, done_space,
                    n_envs, buffer_size, batch_size):
            self.prob_shape = prob_shape
            super(MeanField_OffPolicyBuffer, self).__init__(n_agents, state_space, obs_space, act_space, rew_space,
                                                            done_space, n_envs, buffer_size, batch_size)

        def clear(self):
            super(MeanField_OffPolicyBuffer, self).clear()
            self.data.update({"act_mean": np.zeros((self.n_envs, self.n_size,) + self.prob_shape).astype(np.float32)})

        def sample(self):
            env_choices = np.random.choice(self.n_envs, self.batch_size)
            step_choices = np.random.choice(self.size, self.batch_size)
            samples = {k: self.data[k][env_choices, step_choices] for k in self.keys}
            next_index = (step_choices + 1) % self.n_size
            samples.update({'act_mean_next': self.data['act_mean'][env_choices, next_index]})
            return samples




