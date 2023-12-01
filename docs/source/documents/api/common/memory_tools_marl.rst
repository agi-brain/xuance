Memory for MARL
=========================================

xxxxxx.

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class::
  xuance.common.memory_tools_marl.BaseBuffer(*args)

  :param *args: xxxxxx.
  :type *args: xxxxxx

.. py:function::
  xuance.common.memory_tools_marl.BaseBuffer.full()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.common.memory_tools_marl.BaseBuffer.store(*args, **kwargs)

  xxxxxx.

  :param *args: xxxxxx.
  :type *args: xxxxxx
  :param **kwargs: xxxxxx.
  :type **kwargs: xxxxxx

.. py:function::
  xuance.common.memory_tools_marl.BaseBuffer.clear(*args)

  xxxxxx.

  :param *args: xxxxxx.
  :type *args: xxxxxx

.. py:function::
  xuance.common.memory_tools_marl.BaseBuffer.sample(*args)

  xxxxxx.

  :param *args: xxxxxx.
  :type *args: xxxxxx

.. py:function::
  xuance.common.memory_tools_marl.BaseBuffer.finish_path(*args)

  xxxxxx.

  :param *args: xxxxxx.
  :type *args: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.common.memory_tools_marl.MARL_OffPolicyBuffer(n_agents, state_space, obs_space, act_space, rew_space, done_space,
                 n_envs, buffer_size, batch_size, **kwargs)

  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
  :param state_space: xxxxxx.
  :type state_space: xxxxxx
  :param obs_space: xxxxxx.
  :type obs_space: xxxxxx
  :param act_space: xxxxxx.
  :type act_space: xxxxxx
  :param rew_space: xxxxxx.
  :type rew_space: xxxxxx
  :param done_space: xxxxxx.
  :type done_space: xxxxxx
  :param n_envs: xxxxxx.
  :type n_envs: xxxxxx
  :param buffer_size: xxxxxx.
  :type buffer_size: xxxxxx
  :param batch_size: xxxxxx.
  :type batch_size: xxxxxx
  :param **kwargs: xxxxxx.
  :type **kwargs: xxxxxx

.. py:function::
  xuance.common.memory_tools_marl.MARL_OffPolicyBuffer.clear()

  xxxxxx.

.. py:function::
  xuance.common.memory_tools_marl.MARL_OffPolicyBuffer.store(step_data)

  xxxxxx.

  :param step_data: xxxxxx.
  :type step_data: xxxxxx

.. py:function::
  xuance.common.memory_tools_marl.MARL_OffPolicyBuffer.sample()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.common.memory_tools_marl.MARL_OffPolicyBuffer_RNN(n_agents, state_space, obs_space, act_space, rew_space, done_space,
                 n_envs, buffer_size, batch_size, **kwargs)

  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
  :param state_space: xxxxxx.
  :type state_space: xxxxxx
  :param obs_space: xxxxxx.
  :type obs_space: xxxxxx
  :param act_space: xxxxxx.
  :type act_space: xxxxxx
  :param rew_space: xxxxxx.
  :type rew_space: xxxxxx
  :param done_space: xxxxxx.
  :type done_space: xxxxxx
  :param n_envs: xxxxxx.
  :type n_envs: xxxxxx
  :param buffer_size: xxxxxx.
  :type buffer_size: xxxxxx
  :param batch_size: xxxxxx.
  :type batch_size: xxxxxx
  :param **kwargs: xxxxxx.
  :type **kwargs: xxxxxx

.. py:function::
  xuance.common.memory_tools_marl.MARL_OffPolicyBuffer_RNN.clear()

  xxxxxx.

.. py:function::
  xuance.common.memory_tools_marl.MARL_OffPolicyBuffer_RNN.clear_episodes()

  xxxxxx.

.. py:function::
  xuance.common.memory_tools_marl.MARL_OffPolicyBuffer_RNN.store_transitions()

  xxxxxx.

.. py:function::
  xuance.common.memory_tools_marl.MARL_OffPolicyBuffer_RNN.store_episodes()

  xxxxxx.

.. py:function::
  xuance.common.memory_tools_marl.MARL_OffPolicyBuffer_RNN.finish_path()

  xxxxxx.

.. py:function::
  xuance.common.memory_tools_marl.MARL_OffPolicyBuffer_RNN.sample()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.common.memory_tools_marl.MeanField_OffPolicyBuffer(n_agents, state_space, obs_space, act_space, prob_shape, rew_space, done_space,
                 n_envs, buffer_size, batch_size)

  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
  :param state_space: xxxxxx.
  :type state_space: xxxxxx
  :param obs_space: xxxxxx.
  :type obs_space: xxxxxx
  :param act_space: xxxxxx.
  :type act_space: xxxxxx
  :param prob_shape: xxxxxx.
  :type prob_shape: xxxxxx
  :param rew_space: xxxxxx.
  :type rew_space: xxxxxx
  :param done_space: xxxxxx.
  :type done_space: xxxxxx
  :param n_envs: xxxxxx.
  :type n_envs: xxxxxx
  :param buffer_size: xxxxxx.
  :type buffer_size: xxxxxx
  :param batch_size: xxxxxx.
  :type batch_size: xxxxxx

.. py:function::
  xuance.common.memory_tools_marl.MeanField_OffPolicyBuffer.clear()

  xxxxxx.

.. py:function::
  xuance.common.memory_tools_marl.MeanField_OffPolicyBuffer.sample()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.common.memory_tools_marl.MeanField_OffPolicyBuffer(n_agents, state_space, obs_space, act_space, rew_space, done_space, n_envs, buffer_size,
                 use_gae, use_advnorm, gamma, gae_lam, **kwargs)

  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
  :param state_space: xxxxxx.
  :type state_space: xxxxxx
  :param obs_space: xxxxxx.
  :type obs_space: xxxxxx
  :param act_space: xxxxxx.
  :type act_space: xxxxxx
  :param rew_space: xxxxxx.
  :type rew_space: xxxxxx
  :param done_space: xxxxxx.
  :type done_space: xxxxxx
  :param n_envs: xxxxxx.
  :type n_envs: xxxxxx
  :param buffer_size: xxxxxx.
  :type buffer_size: xxxxxx
  :param use_gae: xxxxxx.
  :type use_gae: xxxxxx
  :param use_advnorm: xxxxxx.
  :type use_advnorm: xxxxxx
  :param gamma: xxxxxx.
  :type gamma: xxxxxx
  :param gae_lam: xxxxxx.
  :type gae_lam: xxxxxx
  :param **kwargs: xxxxxx.
  :type **kwargs: xxxxxx

.. py:function::
  xuance.common.memory_tools_marl.MeanField_OffPolicyBuffer.clear()

  xxxxxx.

.. py:function::
  xuance.common.memory_tools_marl.MeanField_OffPolicyBuffer.store(step_data)

  xxxxxx.

  :param step_data: xxxxxx.
  :type step_data: xxxxxx

.. py:function::
  xuance.common.memory_tools_marl.MeanField_OffPolicyBuffer.finish_path(value, i_env, value_normalizer)

  xxxxxx.

  :param value: xxxxxx.
  :type value: xxxxxx
  :param i_env: xxxxxx.
  :type i_env: xxxxxx
  :param value_normalizer: xxxxxx.
  :type value_normalizer: xxxxxx

.. py:function::
  xuance.common.memory_tools_marl.MeanField_OffPolicyBuffer.sample(indexes)

  xxxxxx.

  :param indexes: xxxxxx.
  :type indexes: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.common.memory_tools_marl.MARL_OnPolicyBuffer_RNN(n_agents, state_space, obs_space, act_space, rew_space, done_space, n_envs, buffer_size,
                 use_gae, use_advnorm, gamma, gae_lam, **kwargs)

  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
  :param state_space: xxxxxx.
  :type state_space: xxxxxx
  :param obs_space: xxxxxx.
  :type obs_space: xxxxxx
  :param act_space: xxxxxx.
  :type act_space: xxxxxx
  :param rew_space: xxxxxx.
  :type rew_space: xxxxxx
  :param done_space: xxxxxx.
  :type done_space: xxxxxx
  :param n_envs: xxxxxx.
  :type n_envs: xxxxxx
  :param buffer_size: xxxxxx.
  :type buffer_size: xxxxxx
  :param use_gae: xxxxxx.
  :type use_gae: xxxxxx
  :param use_advnorm: xxxxxx.
  :type use_advnorm: xxxxxx
  :param gamma: xxxxxx.
  :type gamma: xxxxxx
  :param gae_lam: xxxxxx.
  :type gae_lam: xxxxxx
  :param **kwargs: xxxxxx.
  :type **kwargs: xxxxxx

.. py:function::
  xuance.common.memory_tools_marl.MARL_OnPolicyBuffer_RNN.full()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.common.memory_tools_marl.MARL_OnPolicyBuffer_RNN.clear()

  xxxxxx.

.. py:function::
  xuance.common.memory_tools_marl.MARL_OnPolicyBuffer_RNN.clear_episodes()

  xxxxxx.

.. py:function::
  xuance.common.memory_tools_marl.MARL_OnPolicyBuffer_RNN.store_transitions()

  xxxxxx.

.. py:function::
  xuance.common.memory_tools_marl.MARL_OnPolicyBuffer_RNN.store_episodes()

  xxxxxx.

.. py:function::
  xuance.common.memory_tools_marl.MARL_OnPolicyBuffer_RNN.finish_path(i_env, next_t, *terminal_data, value_next, value_normalizer)

  xxxxxx.

  :param i_env: xxxxxx.
  :type i_env: xxxxxx
  :param next_t: xxxxxx.
  :type next_t: xxxxxx
  :param *terminal_data: xxxxxx.
  :type *terminal_data: xxxxxx
  :param value_next: xxxxxx.
  :type value_next: xxxxxx
  :param value_normalizer: xxxxxx.
  :type value_normalizer: xxxxxx

.. py:function::
  xuance.common.memory_tools_marl.MARL_OnPolicyBuffer_RNN.sample(indexes)

  xxxxxx.

  :param indexes: xxxxxx.
  :type indexes: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.common.memory_tools_marl.MARL_OnPolicyBuffer_MindSpore(n_agents, state_space, obs_space, act_space, rew_space, done_space, n_envs,
                 n_size, use_gae, use_advnorm, gamma, gae_lam, n_actions)

  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
  :param state_space: xxxxxx.
  :type state_space: xxxxxx
  :param obs_space: xxxxxx.
  :type obs_space: xxxxxx
  :param act_space: xxxxxx.
  :type act_space: xxxxxx
  :param rew_space: xxxxxx.
  :type rew_space: xxxxxx
  :param done_space: xxxxxx.
  :type done_space: xxxxxx
  :param n_envs: xxxxxx.
  :type n_envs: xxxxxx
  :param n_size: xxxxxx.
  :type n_size: xxxxxx
  :param use_gae: xxxxxx.
  :type use_gae: xxxxxx
  :param use_advnorm: xxxxxx.
  :type use_advnorm: xxxxxx
  :param gamma: xxxxxx.
  :type gamma: xxxxxx
  :param gae_lam: xxxxxx.
  :type gae_lam: xxxxxx
  :param n_actions: xxxxxx.
  :type n_actions: xxxxxx

.. py:function::
  xuance.common.memory_tools_marl.MARL_OnPolicyBuffer_MindSpore.clear()

  xxxxxx.

.. py:function::
  xuance.common.memory_tools_marl.MARL_OnPolicyBuffer_MindSpore.store()

  xxxxxx.

.. py:class::
  xuance.common.memory_tools_marl.MeanField_OnPolicyBuffer(n_agents, state_space, obs_space, act_space, rew_space, done_space, n_envs,
                 n_size, use_gae, use_advnorm, gamma, gae_lam, **kwargs)

  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
  :param state_space: xxxxxx.
  :type state_space: xxxxxx
  :param obs_space: xxxxxx.
  :type obs_space: xxxxxx
  :param act_space: xxxxxx.
  :type act_space: xxxxxx
  :param rew_space: xxxxxx.
  :type rew_space: xxxxxx
  :param done_space: xxxxxx.
  :type done_space: xxxxxx
  :param n_envs: xxxxxx.
  :type n_envs: xxxxxx
  :param n_size: xxxxxx.
  :type n_size: xxxxxx
  :param use_gae: xxxxxx.
  :type use_gae: xxxxxx
  :param use_advnorm: xxxxxx.
  :type use_advnorm: xxxxxx
  :param gamma: xxxxxx.
  :type gamma: xxxxxx
  :param gae_lam: xxxxxx.
  :type gae_lam: xxxxxx
  :param **kwargs: xxxxxx.
  :type **kwargs: xxxxxx

.. py:function::
  xuance.common.memory_tools_marl.MeanField_OnPolicyBuffer.clear()

  xxxxxx.

.. py:function::
  xuance.common.memory_tools_marl.MeanField_OnPolicyBuffer.finish_ac_path(value, i_env)

  xxxxxx.

  :param value: xxxxxx.
  :type value: xxxxxx
  :param i_env: xxxxxx.
  :type i_env: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.common.memory_tools_marl.COMA_Buffer(n_agents, state_space, obs_space, act_space, rew_space, done_space, n_envs,
                 buffer_size, use_gae, use_advnorm, gamma, gae_lam, **kwargs)

  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
  :param state_space: xxxxxx.
  :type state_space: xxxxxx
  :param obs_space: xxxxxx.
  :type obs_space: xxxxxx
  :param act_space: xxxxxx.
  :type act_space: xxxxxx
  :param rew_space: xxxxxx.
  :type rew_space: xxxxxx
  :param done_space: xxxxxx.
  :type done_space: xxxxxx
  :param n_envs: xxxxxx.
  :type n_envs: xxxxxx
  :param buffer_size: xxxxxx.
  :type buffer_size: xxxxxx
  :param use_gae: xxxxxx.
  :type use_gae: xxxxxx
  :param use_advnorm: xxxxxx.
  :type use_advnorm: xxxxxx
  :param gamma: xxxxxx.
  :type gamma: xxxxxx
  :param gae_lam: xxxxxx.
  :type gae_lam: xxxxxx
  :param **kwargs: xxxxxx.
  :type **kwargs: xxxxxx

.. py:function::
  xuance.common.memory_tools_marl.COMA_Buffer.clear()

  xxxxxx.

.. py:function::
  xuance.common.memory_tools_marl.COMA_Buffer.finish_path(value, i_env, value_normalizer)

  xxxxxx.

  :param value: xxxxxx.
  :type value: xxxxxx
  :param i_env: xxxxxx.
  :type i_env: xxxxxx
  :param value_normalizer: xxxxxx.
  :type value_normalizer: xxxxxx

.. py:class::
  xuance.common.memory_tools_marl.COMA_Buffer_RNN(n_agents, state_space, obs_space, act_space, rew_space, done_space, n_envs,
                 buffer_size, use_gae, use_advnorm, gamma, gae_lam, **kwargs)

  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
  :param state_space: xxxxxx.
  :type state_space: xxxxxx
  :param obs_space: xxxxxx.
  :type obs_space: xxxxxx
  :param act_space: xxxxxx.
  :type act_space: xxxxxx
  :param rew_space: xxxxxx.
  :type rew_space: xxxxxx
  :param done_space: xxxxxx.
  :type done_space: xxxxxx
  :param n_envs: xxxxxx.
  :type n_envs: xxxxxx
  :param buffer_size: xxxxxx.
  :type buffer_size: xxxxxx
  :param use_gae: xxxxxx.
  :type use_gae: xxxxxx
  :param use_advnorm: xxxxxx.
  :type use_advnorm: xxxxxx
  :param gamma: xxxxxx.
  :type gamma: xxxxxx
  :param gae_lam: xxxxxx.
  :type gae_lam: xxxxxx
  :param **kwargs: xxxxxx.
  :type **kwargs: xxxxxx

.. py:function::
  xuance.common.memory_tools_marl.COMA_Buffer_RNN.clear()

  xxxxxx.

.. py:function::
  xuance.common.memory_tools_marl.COMA_Buffer_RNN.clear_episodes()

  xxxxxx.

.. py:function::
  xuance.common.memory_tools_marl.COMA_Buffer_RNN.store_transitions()

  xxxxxx.

.. py:function::
  xuance.common.memory_tools_marl.COMA_Buffer_RNN.finish_path(i_env, next_t, *terminal_data, value_next, value_normalizer)

  xxxxxx.

  :param i_env: xxxxxx.
  :type i_env: xxxxxx
  :param next_t: xxxxxx.
  :type next_t: xxxxxx
  :param *terminal_data: xxxxxx.
  :type *terminal_data: xxxxxx
  :param value_next: xxxxxx.
  :type value_next: xxxxxx
  :param value_normalizer: xxxxxx.
  :type value_normalizer: xxxxxx

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


        class MARL_OffPolicyBuffer(BaseBuffer):
            """
            Replay buffer for off-policy MARL algorithms.
                n_agents: number of agents.
                state_space: global state space, type: Discrete, Box.
                obs_space: observation space for one agent (suppose same obs space for group agents).
                act_space: action space for one agent (suppose same actions space for group agents).
                rew_space: reward space.
                done_space: terminal variable space.
                n_envs: number of parallel environments.
                n_size: buffer size for one environment.
                batch_size: batch size of transition data for a sample.
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
                n_agents: number of agents.
                state_space: global state space, type: Discrete, Box.
                obs_space: observation space for one agent (suppose same obs space for group agents).
                act_space: action space for one agent (suppose same actions space for group agents).
                rew_space: reward space.
                done_space: terminal variable space.
                n_envs: number of parallel environments.
                n_size: buffer size for one environment.
                batch_size: batch size of episodes for a sample.
                max_episode_length: maximum length of data for one episode trajectory.
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
                n_agents: number of agents.
                state_space: global state space, type: Discrete, Box.
                obs_space: observation space for one agent (suppose same obs space for group agents).
                act_space: action space for one agent (suppose same actions space for group agents).
                rew_space: reward space.
                done_space: terminal variable space.
                n_envs: number of parallel environments.
                n_size: buffer size for one environment.
                batch_size: batch size of transition data for a sample.
            """

            def __init__(self, n_agents, state_space, obs_space, act_space, prob_shape, rew_space, done_space,
                         n_envs, buffer_size, batch_size):
                super(MeanField_OffPolicyBuffer, self).__init__(n_agents, state_space, obs_space, act_space, rew_space,
                                                                done_space, n_envs, buffer_size, batch_size)
                self.prob_shape = prob_shape

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


        class MARL_OnPolicyBuffer(BaseBuffer):
            """
            Replay buffer for on-policy MARL algorithms.
                n_agents: number of agents.
                state_space: global state space, type: Discrete, Box.
                obs_space: observation space for one agent (suppose same obs space for group agents).
                act_space: action space for one agent (suppose same actions space for group agents).
                rew_space: reward space.
                done_space: terminal variable space.
                n_envs: number of parallel environments.
                n_size: buffer size of transition data for one environment.
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


        class MARL_OnPolicyBuffer_MindSpore(MARL_OnPolicyBuffer):
            """
            Replay buffer for on-policy MARL algorithms implemented by MindSpore.
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
                n_actions: number of discrete actions.
            """

            def __init__(self, n_agents, state_space, obs_space, act_space, rew_space, done_space, n_envs,
                         n_size, use_gae, use_advnorm, gamma, gae_lam, n_actions=None):
                self.n_actions = n_actions
                super(MARL_OnPolicyBuffer_MindSpore, self).__init__(n_agents, state_space, obs_space, act_space, rew_space,
                                                                    done_space, n_envs, n_size,
                                                                    use_gae, use_advnorm, gamma, gae_lam)
                self.keys = self.data.keys()
                self.data_shapes = {k: self.data[k].shape for k in self.keys}

            def clear(self):
                self.data.update({
                    'obs': np.zeros((self.n_envs, self.n_size,) + self.obs_space).astype(np.float32),
                    'state': np.zeros((self.n_envs, self.n_size,) + self.state_space).astype(np.float32),
                    'actions': np.zeros((self.n_envs, self.n_size,) + self.act_space).astype(np.float32),
                    'rewards': np.zeros((self.n_envs, self.n_size,) + self.rew_space).astype(np.float32),
                    'values': np.zeros((self.n_envs, self.n_size,) + self.rew_space).astype(np.float32),
                    'log_pi_old': np.zeros((self.n_envs, self.n_size, self.n_agents,)).astype(np.float32),
                    'act_prob_old': np.zeros((self.n_envs, self.n_size, self.n_agents, self.n_actions)).astype(np.float32),
                    'advantages': np.zeros((self.n_envs, self.n_size,) + self.rew_space).astype(np.float32),
                    'terminals': np.zeros((self.n_envs, self.n_size,) + self.done_space).astype(np.bool),
                    'agent_mask': np.ones((self.n_envs, self.n_size, self.n_agents)).astype(np.bool),
                })

                self.ptr = 0  # current pointer
                self.size = 0  # current buffer size
                self.start_ids = np.zeros(self.n_envs)

            def store(self, step_data):
                for k in self.keys:
                    if k == "advantages": continue
                    if k in step_data.keys():
                        self.data[k][:, self.ptr] = step_data[k]
                self.ptr = (self.ptr + 1) % self.n_size
                self.size = np.min([self.size + 1, self.n_size])


        class MeanField_OnPolicyBuffer(MARL_OnPolicyBuffer):
            """
            Replay buffer for on-policy Mean-Field MARL algorithms (Mean-Field Actor-Critic).
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
                prob_space: action probabilistic space.
            """

            def __init__(self, n_agents, state_space, obs_space, act_space, rew_space, done_space, n_envs,
                         n_size, use_gae, use_advnorm, gamma, gae_lam, **kwargs):
                self.prob_space = kwargs['prob_space']
                super(MeanField_OnPolicyBuffer, self).__init__(n_agents, state_space, obs_space, act_space, rew_space,
                                                               done_space, n_envs, n_size, use_gae, use_advnorm, gamma, gae_lam,
                                                               **kwargs)

            def clear(self):
                self.data.update({
                    'obs': np.zeros((self.n_envs, self.n_size,) + self.obs_space).astype(np.float32),
                    'obs_next': np.zeros((self.n_envs, self.n_size,) + self.obs_space).astype(np.float32),
                    'state': np.zeros((self.n_envs, self.n_size,) + self.state_space).astype(np.float32),
                    'state_next': np.zeros((self.n_envs, self.n_size,) + self.state_space).astype(np.float32),
                    'actions': np.zeros((self.n_envs, self.n_size,) + self.act_space).astype(np.float32),
                    'act_mean': np.zeros((self.n_envs, self.n_size,) + self.prob_space).astype(np.float32),
                    'rewards': np.zeros((self.n_envs, self.n_size,) + self.rew_space).astype(np.float32),
                    'terminals': np.zeros((self.n_envs, self.n_size,) + self.done_space).astype(np.bool),
                    'agent_mask': np.ones((self.n_envs, self.n_size, self.n_agents)).astype(np.bool),
                })
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

  .. group-tab:: TensorFlow

    .. code-block:: python


  .. group-tab:: MindSpore

    .. code-block:: python




