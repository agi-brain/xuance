import numpy as np
from abc import ABC, abstractmethod
from xuanpolicy.common.common_tools import discount_cumsum


class BaseBuffer(ABC):
    """
    Basic buffer for MARL algorithms.
    """
    def __init__(self, *args):
        self.n_agents, self.state_space, self.obs_space, self.act_space, self.rew_space, self.done_space, self.n_envs, self.n_size = args
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
                 n_envs, n_size, batch_size, **kwargs):
        super(MARL_OffPolicyBuffer, self).__init__(n_agents, state_space, obs_space, act_space, rew_space, done_space,
                                                   n_envs, n_size)
        self.buffer_size = n_size * n_envs
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
                 n_envs, n_size, batch_size, **kwargs):
        self.max_eps_len = kwargs['max_episode_length']
        self.dim_act = kwargs['dim_act']
        super(MARL_OffPolicyBuffer_RNN, self).__init__(n_agents, state_space, obs_space, act_space, rew_space,
                                                       done_space, n_envs, n_size, batch_size)

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

    def store(self, episode_data, i_env=None):
        for k in self.keys:
            self.data[k][self.ptr] = episode_data[k][i_env]
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = np.min([self.size + 1, self.buffer_size])

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
                 n_envs, n_size, batch_size):
        super(MeanField_OffPolicyBuffer, self).__init__(n_agents, state_space, obs_space, act_space, rew_space,
                                                        done_space, n_envs, n_size, batch_size)
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
    def __init__(self, n_agents, state_space, obs_space, act_space, rew_space, done_space, n_envs, n_size,
                 use_gae, use_advnorm, gamma, gae_lam, **kwargs):
        super(MARL_OnPolicyBuffer, self).__init__(n_agents, state_space, obs_space, act_space, rew_space, done_space,
                                                  n_envs, n_size)
        self.buffer_size = n_size * n_envs
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
    def __init__(self, n_agents, state_space, obs_space, act_space, rew_space, done_space, n_envs, n_size,
                 use_gae, use_advnorm, gamma, gae_lam, **kwargs):
        self.max_eps_len = kwargs['max_episode_length']
        self.dim_act = kwargs['dim_act']
        super(MARL_OnPolicyBuffer_RNN, self).__init__(n_agents, state_space, obs_space, act_space, rew_space,
                                                      done_space, n_envs, n_size, use_gae, use_advnorm, gamma, gae_lam,
                                                      **kwargs)

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
            self.data.update({'state': np.zeros(
                (self.buffer_size, self.max_eps_len + 1) + self.state_space, np.float32)})
        self.ptr, self.size = 0, 0

    def store(self, episode_data, i_env=None):
        episode_data_keys = episode_data.keys()
        for k in self.keys:
            if k in episode_data_keys:
                self.data[k][self.ptr] = episode_data[k][i_env]
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def finish_path(self, value, i_env, episode_data=None, current_t=None, value_normalizer=None):
        """ when an episode is finished. """
        if current_t > self.max_eps_len:
            path_slice = np.arange(0, self.max_eps_len).astype(np.int32)
        else:
            path_slice = np.arange(0, current_t).astype(np.int32)

        # calculate advantages and returns
        rewards = np.array(episode_data['rewards'][i_env, :, path_slice])
        vs = np.append(np.array(episode_data['values'][i_env, :, path_slice]), [value.reshape(self.n_agents, 1)], axis=0)
        dones = np.array(episode_data['terminals'][i_env, path_slice])[:, :, None]
        returns = np.zeros_like(rewards)
        last_gae_lam = 0
        step_nums = len(path_slice)
        use_value_norm = False if (value_normalizer is None) else True

        if self.use_gae:
            for t in reversed(range(step_nums)):
                if use_value_norm:
                    vs_t, vs_next = value_normalizer.denormalize(vs[t]), value_normalizer.denormalize(vs[t+1])
                else:
                    vs_t, vs_next = vs[t], vs[t+1]
                delta = rewards[t] + (1 - dones[t]) * self.gamma * vs_next - vs_t
                last_gae_lam = delta + (1 - dones[t]) * self.gamma * self.gae_lambda * last_gae_lam
                returns[t] = last_gae_lam + vs_t
            advantages = returns - value_normalizer.denormalize(vs[:-1]) if use_value_norm else returns - vs[:-1]
        else:
            returns = np.append(returns, [value.reshape(self.n_agents, 1)], axis=0)
            for t in reversed(range(step_nums)):
                returns[t] = rewards[t] + (1 - dones[t]) * self.gamma * returns[t + 1]
            advantages = returns - value_normalizer.denormalize(vs) if use_value_norm else returns - vs
            advantages = advantages[:-1]

        episode_data['returns'][i_env, :, path_slice] = returns
        episode_data['advantages'][i_env, :, path_slice] = advantages
        self.store(episode_data, i_env)

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


class COMA_Buffer(BaseBuffer, ABC):
    def __init__(self, state_space, obs_space, act_space, act_onehot_space, rew_space, done_space,
                 n_envs, buffer_size, batch_size, max_seq_length):
        super(COMA_Buffer, self).__init__(obs_space, act_space, rew_space, n_envs, buffer_size, batch_size)
        self.state_space = state_space
        self.act_onehot_space = act_onehot_space
        self.done_space = done_space
        self.max_seq_len = max_seq_length
        self.n_agents = act_space[0]
        self.batch_size = batch_size
        self.buffer_size_env = self.buffer_size
        self.buffer_size = self.buffer_size_env * self.n_envs

        self.end_ids = np.zeros(self.n_envs)  # the end index of the episode for each env.

        self.data = {}
        self.clear()
        self.keys = self.data.keys()
        self.data_shapes = {k: self.data[k].shape for k in self.keys}

    def clear(self):
        self.data.update({
            'obs': np.zeros((self.n_envs, self.buffer_size_env, self.max_seq_len,) + self.obs_space).astype(np.float32),
            'state': np.zeros((self.n_envs, self.buffer_size_env, self.max_seq_len,) + self.state_space).astype(
                np.float32),
            'actions': np.zeros((self.n_envs, self.buffer_size_env, self.max_seq_len,) + self.act_space).astype(
                np.float32),
            'actions_onehot': np.zeros(
                (self.n_envs, self.buffer_size_env, self.max_seq_len,) + self.act_onehot_space).astype(np.float32),
            'rewards': np.zeros((self.n_envs, self.buffer_size_env, self.max_seq_len,) + self.rew_space).astype(
                np.float32),
            'terminals': np.zeros((self.n_envs, self.buffer_size_env, self.max_seq_len,) + self.done_space).astype(
                np.bool),
            'agent_mask': np.ones((self.n_envs, self.buffer_size_env, self.max_seq_len, self.n_agents)).astype(np.bool),
        })

        self.buffer_ptrs = np.array(np.zeros(self.n_envs, np.int))
        self.step_ptr = 0  # current pointer
        self.size = 0  # current buffer size
        self.end_ids = np.array(np.zeros(self.n_envs, np.int))
        self.env_index = np.arange(self.n_envs)

    def store(self, step_data):
        for k in self.keys:
            self.data[k][self.env_index, self.buffer_ptrs, self.step_ptr] = step_data[k]
        self.step_ptr = (self.step_ptr + 1) % self.max_seq_len

    def finish_ac_path(self, value=None, i_env=None):
        self.buffer_ptrs[i_env] = (self.buffer_ptrs[i_env] + 1) % self.buffer_size_env
        self.end_ids[i_env] = self.max_seq_len if self.step_ptr == 0 else self.step_ptr
        self.size = np.min([self.size + 1, self.buffer_size])

    def can_sample(self):
        return self.size >= self.batch_size

    def sample(self):
        random_env_index = np.random.choice(self.n_envs, size=self.batch_size)
        random_buffer_index = np.random.choice(self.buffer_size_env, size=self.batch_size)
        samples = {k: self.data[k][random_env_index, random_buffer_index] for k in self.keys}
        samples.update({'batch_size': self.batch_size})
        return samples
