import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict
from gym.spaces import Space
from xuance.common import space2shape


def create_memory(shape: Optional[Union[tuple, dict]],
                  n_envs: int,
                  n_size: int,
                  n_agent: Optional[int] = None,
                  dtype: type = np.float32):
    """
    Create a numpy array for memory data.

    Args:
        shape: data shape.
        n_envs: number of parallel environments.
        n_size: length of data sequence for each environment.
        n_agent: number of agents.
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
                if n_agent is None:
                    memory[key] = np.zeros([n_envs, n_size], dtype=object)
                else:
                    memory[key] = np.zeros([n_envs, n_size, n_agent], dtype=object)
            else:
                if n_agent is None:
                    memory[key] = np.zeros([n_envs, n_size] + list(value), dtype=dtype)
                else:
                    memory[key] = np.zeros([n_envs, n_size, n_agent] + list(value), dtype=dtype)
        return memory
    elif isinstance(shape, tuple):
        if n_agent is None:
            return np.zeros([n_envs, n_size] + list(shape), dtype)
        else:
            return np.zeros([n_envs, n_size, n_agent] + list(shape), dtype)
    else:
        raise NotImplementedError


class BaseBuffer(ABC):
    """
    Basic buffer for MARL algorithms.
    """

    def __init__(self, *args):
        self.n_agents, self.state_space, self.obs_space, self.act_space, self.n_envs, self.buffer_size = args
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

    def store_transitions(self, *args, **kwargs):
        return

    def store_episodes(self, *args, **kwargs):
        return

    def finish_path(self, *args, **kwargs):
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
        buffer_size: buffer size of total experience data.
        use_gae: whether to use GAE trick.
        use_advnorm: whether to use Advantage normalization trick.
        gamma: discount factor.
        gae_lam: gae lambda.
    """

    def __init__(self, n_agents, state_space, obs_space, act_space, rew_space, done_space, n_envs, buffer_size,
                 use_gae, use_advnorm, gamma, gae_lam, **kwargs):
        super(MARL_OnPolicyBuffer, self).__init__(n_agents, state_space, obs_space, act_space, rew_space, done_space,
                                                  n_envs, buffer_size)
        assert buffer_size % self.n_envs == 0, "buffer_size must be divisible by the number of envs (parallels)"
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
            'terminals': np.zeros((self.n_envs, self.n_size,) + self.done_space).astype(np.bool_),
            'agent_mask': np.ones((self.n_envs, self.n_size, self.n_agents)).astype(np.bool_),
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
            returns = np.append(returns, [value], axis=0)
            for t in reversed(range(step_nums)):
                returns[t] = rewards[t] + (1 - dones[t]) * self.gamma * returns[t + 1]
            advantages = returns - value_normalizer.denormalize(vs) if use_value_norm else returns - vs
            advantages = advantages[:-1]

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
        buffer_size: buffer size of total experience data.
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
            'terminals': np.zeros((self.buffer_size, self.max_eps_len) + self.done_space, np.bool_),
            'avail_actions': np.ones((self.buffer_size, self.n_agents, self.max_eps_len + 1, self.dim_act), np.bool_),
            'filled': np.zeros((self.buffer_size, self.max_eps_len, 1), np.bool_)
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
            'terminals': np.zeros((self.n_envs, self.max_eps_len) + self.done_space, dtype=np.bool_),
            'avail_actions': np.ones((self.n_envs, self.n_agents, self.max_eps_len + 1, self.dim_act), dtype=np.bool_),
            'filled': np.zeros((self.n_envs, self.max_eps_len, 1), dtype=np.bool_),
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
            'terminals': np.zeros((self.n_envs, self.n_size,) + self.done_space).astype(np.bool_),
            'agent_mask': np.ones((self.n_envs, self.n_size, self.n_agents)).astype(np.bool_),
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
            'terminals': np.zeros((self.n_envs, self.n_size,) + self.done_space).astype(np.bool_),
            'agent_mask': np.ones((self.n_envs, self.n_size, self.n_agents)).astype(np.bool_),
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
            'terminals': np.zeros((self.buffer_size, self.max_eps_len) + self.done_space, np.bool_),
            'avail_actions': np.ones((self.buffer_size, self.n_agents, self.max_eps_len + 1, self.dim_act), np.bool_),
            'filled': np.zeros((self.buffer_size, self.max_eps_len, 1), np.bool_)
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
            'terminals': np.zeros((self.n_envs, self.max_eps_len) + self.done_space, dtype=np.bool_),
            'avail_actions': np.ones((self.n_envs, self.n_agents, self.max_eps_len + 1, self.dim_act), dtype=np.bool_),
            'filled': np.zeros((self.n_envs, self.max_eps_len, 1), dtype=np.bool_),
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
    Replay buffer for off-policy MARL algorithms with parameter sharing.

    Parameters:
        n_agents (int): number of agents.
        state_space (Dict[str, Space]): global state space, type: Discrete, Box.
        obs_space (Dict[str, Dict[str, Space]]): observation space for one agent (suppose same obs space for group agents).
        act_space (Dict[str, Dict[str, Space]]): action space for one agent (suppose same actions space for group agents).
        n_envs (int): number of parallel environments.
        buffer_size (int): buffer size of total experience data.
        batch_size (int): batch size of transition data for a sample.
        **kwargs: other arguments.

    Example:
        $ obs_space={'agent_0': Box(-inf, inf, (18,), float32),
                     'agent_1': Box(-inf, inf, (18,), float32),
                     'agent_2': Box(-inf, inf, (18,), float32)},
        $ act_space={'agent_0': Box(0.0, 1.0, (5,), float32),
                     'agent_1': Box(0.0, 1.0, (5,), float32),
                     'agent_2': Box(0.0, 1.0, (5,), float32)},
        $ n_envs=50,
        $ buffer_size=10000,
        $ batch_size=256,
        $ model_keys=['agent_0', 'agent_1', 'agent_2'],
        $ use_parameter_sharing=False)
        $ memory = MARL_OffPolicyBuffer(n_agents=3, obs_space=obs_space, act_space=act_space, n_envs=n_envs,
                                        buffer_size=buffer_size, batch_size=batch_size, model_keys=model_keys,
                                        use_parameter_sharing=use_parameter_sharing)
    """

    def __init__(self,
                 n_agents: int,
                 state_space: Dict[str, Space] = None,
                 obs_space: Dict[str, Dict[str, Space]] = None,
                 act_space: Dict[str, Dict[str, Space]] = None,
                 n_envs: int = 1,
                 buffer_size: int = 1,
                 batch_size: int = 1,
                 **kwargs):
        super(MARL_OffPolicyBuffer, self).__init__(n_agents, state_space, obs_space, act_space,
                                                   n_envs, buffer_size)
        assert buffer_size % self.n_envs == 0, "buffer_size must be divisible by the number of envs (i.e., parallels)"
        self.n_size = buffer_size // n_envs
        self.batch_size = batch_size
        self.model_keys = kwargs['model_keys']
        self.store_global_state = False if self.state_space is None else True
        self.use_parameter_sharing = kwargs['use_parameter_sharing'] if 'use_parameter_sharing' in kwargs else False
        self.store_avail_actions = kwargs['store_avail_actions'] if 'store_avail_actions' in kwargs else False
        self.n_actions = kwargs['n_actions'] if 'n_actions' in kwargs else None
        self.data = {}
        self.clear()
        self.data_keys = self.data.keys()

    def clear(self):
        """
        Clear the memory data in the replay buffer.

        Example:
        An example shows the data shape: (n_env=50, n_agent=3, buffer_size=10000).
        When use_parameter_sharing is True, then
            self.data: {'obs': {'agent_0': shape=[50, 200, 3, 18]},  # dim_obs: 18
                        'actions': {'agent_0': shape=[50, 200, 3, 5]},  # dim_act: 5
                         ...}
        When use_parameter_sharing is False, then
            self.data: {'obs': {'agent_0': shape=[50, 200, 18],
                                'agent_1': shape=[50, 200, 18],
                                'agent_2': shape=[50, 200, 18]},  # dim_obs: 18
                        'actions': {'agent_0': shape=[50, 200, 3, 5],
                                    'agent_1': shape=[50, 200, 3, 5],
                                    'agent_2': shape=[50, 200, 3, 5]},  # dim_act: 5
                         ...}
        """
        num_agents = self.n_agents if self.use_parameter_sharing else None
        obs_space = {key: self.obs_space[key] for key in self.model_keys}
        act_space = {key: self.act_space[key] for key in self.model_keys}
        reward_space = {key: () for key in self.model_keys}
        terminal_space = {key: () for key in self.model_keys}
        agent_mask_space = {key: () for key in self.model_keys}
        avail_actions_space = {key: (self.n_actions,) for key in self.model_keys}

        self.data = {
            'obs': create_memory(space2shape(obs_space), self.n_envs, self.n_size, num_agents),
            'actions': create_memory(space2shape(act_space), self.n_envs, self.n_size, num_agents),
            'obs_next': create_memory(space2shape(obs_space), self.n_envs, self.n_size, num_agents),
            'rewards': create_memory(reward_space, self.n_envs, self.n_size, num_agents),
            'terminals': create_memory(terminal_space, self.n_envs, self.n_size, num_agents, np.bool_),
            'agent_mask': create_memory(agent_mask_space, self.n_envs, self.n_size, num_agents, np.bool_)
        }
        if self.store_global_state:
            state_shape = space2shape(self.state_space)
            self.data.update({
                'state': create_memory(state_shape, self.n_envs, self.n_size, None),
                'state_next': create_memory(state_shape, self.n_envs, self.n_size, None)
            })
        if self.store_avail_actions:
            self.data.update({
                "avail_actions": create_memory(avail_actions_space, self.n_envs, self.n_size, num_agents, np.bool_)
            })
        self.ptr, self.size = 0, 0

    def store(self, step_data):
        """ Store a step of data into the replay buffer. """
        for data_key in self.data_keys:
            if data_key in ['state', 'state_next']:
                self.data[data_key][:, self.ptr] = step_data[data_key]
                continue
            for agt_key in self.model_keys:
                self.data[data_key][agt_key][:, self.ptr] = step_data[data_key][agt_key]
        self.ptr = (self.ptr + 1) % self.n_size
        self.size = np.min([self.size + 1, self.n_size])

    def sample(self, batch_size=None):
        """
        Sample a batch of data from the replay buffer.

        Parameters:
            batch_size (int): The size of the batch data to be sampled.

        Returns:
            samples (dict): The sampled data.
        """
        if batch_size is None:
            batch_size = self.batch_size
        env_choices = np.random.choice(self.n_envs, batch_size)
        step_choices = np.random.choice(self.size, batch_size)
        samples = {}
        for data_key in self.data_keys:
            if data_key in ['state', 'state_next']:
                samples[data_key] = self.data[data_key][env_choices, step_choices]
                continue
            samples[data_key] = {agt_key: self.data[data_key][agt_key][env_choices, step_choices]
                                 for agt_key in self.model_keys}
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
        buffer_size: buffer size of total experience data.
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
            'terminals': np.zeros((self.buffer_size, self.max_eps_len) + self.done_space, np.bool_),
            'avail_actions': np.ones((self.buffer_size, self.n_agents, self.max_eps_len + 1, self.dim_act), np.bool_),
            'filled': np.zeros((self.buffer_size, self.max_eps_len, 1)).astype(np.bool_)
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
            'terminals': np.zeros((self.n_envs, self.max_eps_len) + self.done_space, dtype=np.bool_),
            'avail_actions': np.ones((self.n_envs, self.n_agents, self.max_eps_len + 1, self.dim_act), dtype=np.bool_),
            'filled': np.zeros((self.n_envs, self.max_eps_len, 1), dtype=np.bool_),
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
        buffer_size: buffer size of total experience data.
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
