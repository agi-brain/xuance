import numpy as np
from abc import ABC, abstractmethod
from xuance.common import List, Dict, Optional
from gymnasium.spaces import Space
from xuance.common import space2shape, create_memory


class BaseBuffer(ABC):
    """
    Basic buffer for MARL algorithms.
    """

    def __init__(self, *args):
        self.agent_keys, self.state_space, self.obs_space, self.act_space, self.n_envs, self.buffer_size = args
        assert self.buffer_size % self.n_envs == 0, "buffer_size must be divisible by the number of envs (parallels)"
        self.n_size = self.buffer_size // self.n_envs
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

    @abstractmethod
    def finish_path(self, *args, **kwargs):
        raise NotImplementedError


class MARL_OnPolicyBuffer(BaseBuffer):
    """
    Replay buffer for on-policy MARL algorithms.

    Args:
        agent_keys (List[str]): Keys that identify each agent.
        state_space (Dict[str, Space]): Global state space, type: Discrete, Box.
        obs_space (Dict[str, Dict[str, Space]]): Observation space for one agent (suppose same obs space for group agents).
        act_space (Dict[str, Dict[str, Space]]): Action space for one agent (suppose same actions space for group agents).
        n_envs (int): Number of parallel environments.
        buffer_size (int): Buffer size of total experience data.
        use_gae (bool): Whether to use GAE trick.
        use_advnorm (bool): Whether to use Advantage normalization trick.
        gamma (float): Discount factor.
        gae_lam (float): gae lambda.
        **kwargs: Other arguments.

    Example:
        $ state_space=None
        $ obs_space={'agent_0': Box(-inf, inf, (18,), float32),
                     'agent_1': Box(-inf, inf, (18,), float32),
                     'agent_2': Box(-inf, inf, (18,), float32)},
        $ act_space={'agent_0': Box(0.0, 1.0, (5,), float32),
                     'agent_1': Box(0.0, 1.0, (5,), float32),
                     'agent_2': Box(0.0, 1.0, (5,), float32)},
        $ n_envs=16,
        $ buffer_size=1600,
        $ agent_keys=['agent_0', 'agent_1', 'agent_2'],
        $ memory = MARL_OffPolicyBuffer(agent_keys=agent_keys, state_space=state_space, obs_space=obs_space,
                                        act_space=act_space, n_envs=n_envs, buffer_size=buffer_size,
                                        use_gae=False, use_advnorm=False, gamma=0.99, gae_lam=0.95)
    """

    def __init__(self,
                 agent_keys: List[str],
                 state_space: Dict[str, Space] = None,
                 obs_space: Dict[str, Dict[str, Space]] = None,
                 act_space: Dict[str, Dict[str, Space]] = None,
                 n_envs: int = 1,
                 buffer_size: int = 1,
                 use_gae: Optional[bool] = False,
                 use_advnorm: Optional[bool] = False,
                 gamma: Optional[float] = None,
                 gae_lam: Optional[float] = None,
                 **kwargs):
        super(MARL_OnPolicyBuffer, self).__init__(agent_keys, state_space, obs_space, act_space, n_envs, buffer_size)
        self.n_size = buffer_size // self.n_envs
        self.store_global_state = False if self.state_space is None else True
        self.use_actions_mask = kwargs['use_actions_mask'] if 'use_actions_mask' in kwargs else False
        self.avail_actions_shape = kwargs['avail_actions_shape'] if 'avail_actions_shape' in kwargs else None
        self.use_gae = use_gae
        self.use_advantage_norm = use_advnorm
        self.gamma, self.gae_lambda = gamma, gae_lam
        # prepare an empty buffer to store data
        self.data, self.start_ids = {}, None
        self.reward_space = {key: () for key in self.agent_keys}
        self.returns = {key: () for key in self.agent_keys}
        self.values = {key: () for key in self.agent_keys}
        self.log_pi_old = {key: () for key in self.agent_keys}
        self.advantages = {key: () for key in self.agent_keys}
        self.terminal_space = {key: () for key in self.agent_keys}
        self.agent_mask_space = {key: () for key in self.agent_keys}
        self.clear()
        self.data_keys = self.data.keys()

    def clear(self):
        """
        Clears the memory data in the replay buffer.

        Example:
        An example shows the data shape: (n_env=16, buffer_size=1600, agent_keys=['agent_0', 'agent_1', 'agent_2']).
        self.data: {'obs': {'agent_0': shape=[16, 100, 18],
                            'agent_1': shape=[16, 100, 18],
                            'agent_2': shape=[16, 100, 18]},  # dim_obs: 18
                    'actions': {'agent_0': shape=[16, 100, 5],
                                'agent_1': shape=[16, 100, 5],
                                'agent_2': shape=[16, 100, 5]},  # dim_act: 5
                     ...}
        """
        self.data = {
            'obs': create_memory(space2shape(self.obs_space), self.n_envs, self.n_size),
            'actions': create_memory(space2shape(self.act_space), self.n_envs, self.n_size),
            'rewards': create_memory(self.reward_space, self.n_envs, self.n_size),
            'returns': create_memory(self.reward_space, self.n_envs, self.n_size),
            'values': create_memory(self.reward_space, self.n_envs, self.n_size),
            'log_pi_old': create_memory(self.reward_space, self.n_envs, self.n_size),
            'advantages': create_memory(self.reward_space, self.n_envs, self.n_size),
            'terminals': create_memory(self.terminal_space, self.n_envs, self.n_size, np.bool_),
            'agent_mask': create_memory(self.agent_mask_space, self.n_envs, self.n_size, np.bool_),
        }

        if self.store_global_state:
            self.data.update({
                'state': create_memory(space2shape(self.state_space), self.n_envs, self.n_size)
            })
        if self.use_actions_mask:
            self.data.update({
                "avail_actions": create_memory(self.avail_actions_shape, self.n_envs, self.n_size, np.bool_),
            })
        self.ptr, self.size = 0, 0
        self.start_ids = np.zeros(self.n_envs, np.int64)  # the start index of the last episode for each env.

    def store(self, **step_data):
        """ Stores a step of data into the replay buffer. """
        for data_key, data_value in step_data.items():
            if data_key in ['state']:
                self.data[data_key][:, self.ptr] = data_value
                continue
            for agt_key in self.agent_keys:
                self.data[data_key][agt_key][:, self.ptr] = data_value[agt_key]
        self.ptr = (self.ptr + 1) % self.n_size
        self.size = np.min([self.size + 1, self.n_size])

    def finish_path(self,
                    i_env: Optional[int] = None,
                    value_next: Optional[dict] = None,
                    value_normalizer=None):
        """
        Calculates and stores the returns and advantages when an episode is finished.

        Parameters:
            i_env (int): The index of environment.
            value_next (dict): The critic values of the terminal state.
            value_normalizer: The value normalizer method, default is None.
        """
        if self.size == 0:
            return
        if self.full:
            path_slice = np.arange(self.start_ids[i_env], self.n_size).astype(np.int32)
        else:
            path_slice = np.arange(self.start_ids[i_env], self.ptr).astype(np.int32)

        # calculate advantages and returns
        use_value_norm = False if (value_normalizer is None) else True
        use_parameter_sharing = False
        if use_value_norm:
            if value_normalizer.keys() != set(self.agent_keys):
                use_parameter_sharing = True
        for key in self.agent_keys:
            rewards = self.data['rewards'][key][i_env, path_slice]
            vs = np.append(self.data['values'][key][i_env, path_slice], [value_next[key]], axis=0)
            dones = self.data['terminals'][key][i_env, path_slice]
            returns = np.zeros_like(rewards)
            last_gae_lam = 0
            step_nums = len(path_slice)
            key_vn = self.agent_keys[0] if use_parameter_sharing else key

            if self.use_gae:
                for t in reversed(range(step_nums)):
                    if use_value_norm:
                        vs_t, vs_next = value_normalizer[key_vn].denormalize(vs[t]), value_normalizer[
                            key_vn].denormalize(vs[t + 1])
                    else:
                        vs_t, vs_next = vs[t], vs[t + 1]
                    delta = rewards[t] + (1 - dones[t]) * self.gamma * vs_next - vs_t
                    last_gae_lam = delta + (1 - dones[t]) * self.gamma * self.gae_lambda * last_gae_lam
                    returns[t] = last_gae_lam + vs_t
                advantages = returns - value_normalizer[key_vn].denormalize(
                    vs[:-1]) if use_value_norm else returns - vs[:-1]
            else:
                returns_ = np.append(returns, [value_next[key]], axis=0)
                for t in reversed(range(step_nums)):
                    returns_[t] = rewards[t] + (1 - dones[t]) * self.gamma * returns_[t + 1]
                advantages = returns_ - value_normalizer[key_vn].denormalize(vs) if use_value_norm else returns_ - vs
                advantages = advantages[:-1]
                returns = returns_[:-1]

            self.data['returns'][key][i_env, path_slice] = returns
            self.data['advantages'][key][i_env, path_slice] = advantages
        self.start_ids[i_env] = self.ptr

    def sample(self, indexes: Optional[np.ndarray] = None):
        """
        Samples a batch of data from the replay buffer.

        Parameters:
            indexes (int): The indexes of the data in the buffer that will be sampled.

        Returns:
            samples_dict (dict): The sampled data.
        """
        assert self.full, "Not enough transitions for on-policy buffer to random sample."
        samples_dict = {}
        env_choices, step_choices = divmod(indexes, self.n_size)
        for data_key in self.data_keys:
            if data_key == "advantages":
                adv_batch_dict = {}
                for agt_key in self.agent_keys:
                    adv_batch = self.data[data_key][agt_key][env_choices, step_choices]
                    if self.use_advantage_norm:
                        adv_batch_dict[agt_key] = (adv_batch - np.mean(adv_batch)) / (np.std(adv_batch) + 1e-8)
                    else:
                        adv_batch_dict[agt_key] = adv_batch
                samples_dict[data_key] = adv_batch_dict
            elif data_key == "state":
                samples_dict[data_key] = self.data[data_key][env_choices, step_choices]
            else:
                samples_dict[data_key] = {k: self.data[data_key][k][env_choices, step_choices] for k in self.agent_keys}
        samples_dict['batch_size'] = len(indexes)
        return samples_dict


class MARL_OnPolicyBuffer_RNN(MARL_OnPolicyBuffer):
    """
    Replay buffer for on-policy MARL algorithms with DRQN trick.

    Args:
        agent_keys (List[str]): Keys that identify each agent.
        state_space (Dict[str, Space]): Global state space, type: Discrete, Box.
        obs_space (Dict[str, Dict[str, Space]]): Observation space for one agent (suppose same obs space for group agents).
        act_space (Dict[str, Dict[str, Space]]): Action space for one agent (suppose same actions space for group agents).
        n_envs (int): Number of parallel environments.
        buffer_size (int): Buffer size of total experience data.
        max_episode_steps (int): The sequence length of each episode data.
        use_gae (bool): Whether to use GAE trick.
        use_advnorm (bool): Whether to use Advantage normalization trick.
        gamma (float): Discount factor.
        gae_lam (float): gae lambda.
        **kwargs: Other arguments.

    Example:
        >> state_space=None
        >> obs_space={'agent_0': Box(-inf, inf, (18,), float32),
                      'agent_1': Box(-inf, inf, (18,), float32),
                      'agent_2': Box(-inf, inf, (18,), float32)},
        >> act_space={'agent_0': Box(0.0, 1.0, (5,), float32),
                      'agent_1': Box(0.0, 1.0, (5,), float32),
                      'agent_2': Box(0.0, 1.0, (5,), float32)},
        >> n_envs=16,
        >> buffer_size=1600,
        >> agent_keys=['agent_0', 'agent_1', 'agent_2'],
        >> max_episode_steps = 100
        >> memory = MARL_OffPolicyBuffer(agent_keys=agent_keys, state_space=state_space, obs_space=obs_space,
                                         act_space=act_space, n_envs=n_envs, buffer_size=buffer_size,
                                         max_episode_steps=max_episode_steps,
                                         use_gae=False, use_advnorm=False, gamma=0.99, gae_lam=0.95)
    """

    def __init__(self,
                 agent_keys: List[str],
                 state_space: Dict[str, Space] = None,
                 obs_space: Dict[str, Dict[str, Space]] = None,
                 act_space: Dict[str, Dict[str, Space]] = None,
                 n_envs: int = 1,
                 buffer_size: int = 1,
                 max_episode_steps: int = 1,
                 use_gae: Optional[bool] = False,
                 use_advnorm: Optional[bool] = False,
                 gamma: Optional[float] = None,
                 gae_lam: Optional[float] = None,
                 **kwargs):
        self.max_eps_len = max_episode_steps
        self.n_actions = kwargs['n_actions'] if 'n_actions' in kwargs else None
        self.obs_shape = {k: space2shape(obs_space[k]) for k in agent_keys}
        self.act_shape = {k: space2shape(act_space[k]) for k in agent_keys}
        super(MARL_OnPolicyBuffer_RNN, self).__init__(agent_keys, state_space, obs_space, act_space, n_envs,
                                                      buffer_size, use_gae, use_advnorm, gamma, gae_lam, **kwargs)
        self.episode_data = {}
        self.clear_episodes()

    @property
    def full(self):
        return self.size >= self.buffer_size

    def clear(self):
        self.data = {
            'obs': {k: np.zeros((self.buffer_size, self.max_eps_len) + self.obs_shape[k], np.float32)
                    for k in self.agent_keys},
            'actions': {k: np.zeros((self.buffer_size, self.max_eps_len) + self.act_shape[k], np.float32)
                        for k in self.agent_keys},
            'rewards': {k: np.zeros((self.buffer_size, self.max_eps_len), np.float32) for k in self.agent_keys},
            'returns': {k: np.zeros((self.buffer_size, self.max_eps_len), np.float32) for k in self.agent_keys},
            'values': {k: np.zeros((self.buffer_size, self.max_eps_len), np.float32) for k in self.agent_keys},
            'advantages': {k: np.zeros((self.buffer_size, self.max_eps_len), np.float32) for k in self.agent_keys},
            'log_pi_old': {k: np.zeros((self.buffer_size, self.max_eps_len), np.float32) for k in self.agent_keys},
            'terminals': {k: np.zeros((self.buffer_size, self.max_eps_len), np.bool_) for k in self.agent_keys},
            'agent_mask': {k: np.zeros((self.buffer_size, self.max_eps_len), np.bool_) for k in self.agent_keys},
            'filled': np.zeros((self.buffer_size, self.max_eps_len), np.bool_)
        }
        if self.store_global_state:
            self.data.update({
                'state': np.zeros((self.buffer_size, self.max_eps_len) + self.state_space.shape, np.float32)
            })
        if self.use_actions_mask:
            self.data.update({
                'avail_actions': {k: np.zeros((self.buffer_size, self.max_eps_len) + self.avail_actions_shape[k],
                                              dtype=np.bool_) for k in self.agent_keys}
            })
        self.ptr, self.size = 0, 0

    def clear_episodes(self):
        self.episode_data = {
            'obs': {k: np.zeros((self.n_envs, self.max_eps_len) + self.obs_shape[k], np.float32)
                    for k in self.agent_keys},
            'actions': {k: np.zeros((self.n_envs, self.max_eps_len) + self.act_shape[k], np.float32)
                        for k in self.agent_keys},
            'rewards': {k: np.zeros((self.n_envs, self.max_eps_len), np.float32) for k in self.agent_keys},
            'returns': {k: np.zeros((self.n_envs, self.max_eps_len), np.float32) for k in self.agent_keys},
            'values': {k: np.zeros((self.n_envs, self.max_eps_len), np.float32) for k in self.agent_keys},
            'advantages': {k: np.zeros((self.n_envs, self.max_eps_len), np.float32) for k in self.agent_keys},
            'log_pi_old': {k: np.zeros((self.n_envs, self.max_eps_len), np.float32) for k in self.agent_keys},
            'terminals': {k: np.zeros((self.n_envs, self.max_eps_len), np.bool_) for k in self.agent_keys},
            'agent_mask': {k: np.zeros((self.n_envs, self.max_eps_len), np.bool_) for k in self.agent_keys},
            'filled': np.zeros((self.n_envs, self.max_eps_len), np.bool_)
        }
        if self.store_global_state:
            self.episode_data.update({
                'state': np.zeros((self.n_envs, self.max_eps_len) + self.state_space.shape, np.float32)
            })
        if self.use_actions_mask:
            self.episode_data.update({
                'avail_actions': {k: np.zeros((self.n_envs, self.max_eps_len) + self.avail_actions_shape[k],
                                              dtype=np.bool_) for k in self.agent_keys}
            })

    def store(self, **step_data):
        """
        Stores a step of data for each environment.

        Parameters:
            step_data (dict): A dict of step data that to be stored into self.episode_data.
        """
        envs_step = step_data['episode_steps']
        envs_choice = range(self.n_envs)
        self.episode_data["filled"][envs_choice, envs_step] = True
        for data_key, data_value in step_data.items():
            if data_key == "episode_steps":
                continue
            if data_key == 'state':
                self.episode_data[data_key][envs_choice, envs_step] = data_value
                continue
            for agt_key in self.agent_keys:
                self.episode_data[data_key][agt_key][envs_choice, envs_step] = data_value[agt_key]

    def store_episodes(self, i_env):
        """
        Stores the episode of data for ith environment into the self.data.

        Parameters:
            i_env (int): The ith environment.
        """
        for data_key in self.data_keys:
            if data_key == "filled":
                self.data["filled"][self.ptr] = self.episode_data["filled"][i_env].copy()
                continue
            if data_key in ['state']:
                self.data[data_key][self.ptr] = self.episode_data[data_key][i_env].copy()
                continue
            for agt_key in self.agent_keys:
                self.data[data_key][agt_key][self.ptr] = self.episode_data[data_key][agt_key][i_env].copy()
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = np.min([self.size + 1, self.buffer_size])
        # clear the filled values for ith env.
        self.episode_data['filled'][i_env] = np.zeros(self.max_eps_len, dtype=np.bool_)

    def finish_path(self,
                    i_env: Optional[int] = None,
                    i_step: Optional[int] = None,
                    value_next: Optional[dict] = None,
                    value_normalizer: Optional[dict] = None):
        """
        Calculates and stores the returns and advantages when an episode is finished.

        Parameters:
            i_env (int): The index of environment.
            i_step (int): The index of step for current environment.
            value_next (Optional[dict]): The critic values of the terminal state.
            value_normalizer (Optional[dict]): The value normalizer method, default is None.
        """
        env_step = i_step if i_step < self.max_eps_len else self.max_eps_len
        path_slice = np.arange(0, env_step).astype(np.int32)

        # calculate advantages and returns
        use_value_norm = False if (value_normalizer is None) else True
        use_parameter_sharing = False
        if use_value_norm:
            if value_normalizer.keys() != set(self.agent_keys):
                use_parameter_sharing = True
        for key in self.agent_keys:
            rewards = np.array(self.episode_data['rewards'][key][i_env, path_slice])
            vs = np.append(np.array(self.episode_data['values'][key][i_env, path_slice]), [value_next[key]], axis=0)
            dones = np.array(self.episode_data['terminals'][key][i_env, path_slice])
            returns = np.zeros_like(rewards)
            last_gae_lam = 0
            step_nums = len(path_slice)
            key_vn = self.agent_keys[0] if use_parameter_sharing else key

            if self.use_gae:
                for t in reversed(range(step_nums)):
                    if use_value_norm:
                        vs_t = value_normalizer[key_vn].denormalize(vs[t])
                        vs_next = value_normalizer[key_vn].denormalize(vs[t + 1])
                    else:
                        vs_t, vs_next = vs[t], vs[t + 1]
                    delta = rewards[t] + (1 - dones[t]) * self.gamma * vs_next - vs_t
                    last_gae_lam = delta + (1 - dones[t]) * self.gamma * self.gae_lambda * last_gae_lam
                    returns[t] = last_gae_lam + vs_t
                advantages = returns - value_normalizer[key_vn].denormalize(
                    vs[:-1]) if use_value_norm else returns - vs[:-1]
            else:
                returns_ = np.append(returns, [value_next[key]], axis=0)
                for t in reversed(range(step_nums)):
                    returns_[t] = rewards[t] + (1 - dones[t]) * self.gamma * returns_[t + 1]
                advantages = returns_ - value_normalizer[key_vn].denormalize(vs) if use_value_norm else returns_ - vs
                advantages = advantages[:-1]
                returns = returns_[:-1]

            self.episode_data['returns'][key][i_env, path_slice] = returns
            self.episode_data['advantages'][key][i_env, path_slice] = advantages
        self.store_episodes(i_env)

    def sample(self, indexes: Optional[np.ndarray] = None):
        """
        Samples a batch of data from the replay buffer.

        Parameters:
            indexes (int): The indexes of the data in the buffer that will be sampled.

        Returns:
            samples_dict (dict): The sampled data.
        """
        assert self.full, "Not enough transitions for on-policy buffer to random sample"
        episode_choices = indexes
        samples_dict = {}
        for data_key in self.data_keys:
            if data_key == "filled":
                samples_dict["filled"] = self.data['filled'][episode_choices]
                continue
            if data_key in ['state', 'state_next']:
                samples_dict[data_key] = self.data[data_key][episode_choices]
                continue
            samples_dict[data_key] = {k: self.data[data_key][k][episode_choices] for k in self.agent_keys}
        samples_dict['batch_size'] = len(indexes)
        samples_dict['sequence_length'] = self.max_eps_len
        return samples_dict


class IC3Net_OnPolicyBuffer_RNN(MARL_OnPolicyBuffer_RNN):
    def __init__(self,
                 agent_keys: List[str],
                 state_space: Dict[str, Space] = None,
                 obs_space: Dict[str, Dict[str, Space]] = None,
                 act_space: Dict[str, Dict[str, Space]] = None,
                 n_envs: int = 1,
                 buffer_size: int = 1,
                 max_episode_steps: int = 1,
                 use_gae: Optional[bool] = False,
                 use_advnorm: Optional[bool] = False,
                 gamma: Optional[float] = None,
                 gae_lam: Optional[float] = None,
                 **kwargs):
        super(IC3Net_OnPolicyBuffer_RNN, self).__init__(agent_keys, state_space, obs_space, act_space,
                                                        n_envs, buffer_size, max_episode_steps, use_gae, use_advnorm, gamma, gae_lam, **kwargs)

    def clear(self):
        self.data = {
            'obs': {k: np.zeros((self.buffer_size, self.max_eps_len) + self.obs_shape[k], np.float32)
                    for k in self.agent_keys},
            'actions': {k: np.zeros((self.buffer_size, self.max_eps_len) + self.act_shape[k], np.float32)
                        for k in self.agent_keys},
            'rewards': {k: np.zeros((self.buffer_size, self.max_eps_len), np.float32) for k in self.agent_keys},
            'returns': {k: np.zeros((self.buffer_size, self.max_eps_len), np.float32) for k in self.agent_keys},
            'values': {k: np.zeros((self.buffer_size, self.max_eps_len), np.float32) for k in self.agent_keys},
            'advantages': {k: np.zeros((self.buffer_size, self.max_eps_len), np.float32) for k in self.agent_keys},
            'gate_log_pi_old': {k: np.zeros((self.buffer_size, self.max_eps_len), np.float32) for k in self.agent_keys},
            'log_pi_old': {k: np.zeros((self.buffer_size, self.max_eps_len), np.float32) for k in self.agent_keys},
            'terminals': {k: np.zeros((self.buffer_size, self.max_eps_len), np.bool_) for k in self.agent_keys},
            'agent_mask': {k: np.zeros((self.buffer_size, self.max_eps_len), np.bool_) for k in self.agent_keys},
            'filled': np.zeros((self.buffer_size, self.max_eps_len), np.bool_)
        }
        if self.store_global_state:
            self.data.update({
                'state': np.zeros((self.buffer_size, self.max_eps_len) + self.state_space.shape, np.float32)
            })
        if self.use_actions_mask:
            self.data.update({
                'avail_actions': {k: np.zeros((self.buffer_size, self.max_eps_len) + self.avail_actions_shape[k],
                                              dtype=np.bool_) for k in self.agent_keys}
            })
        self.ptr, self.size = 0, 0

    def clear_episodes(self):
        self.episode_data = {
            'obs': {k: np.zeros((self.n_envs, self.max_eps_len) + self.obs_shape[k], np.float32)
                    for k in self.agent_keys},
            'actions': {k: np.zeros((self.n_envs, self.max_eps_len) + self.act_shape[k], np.float32)
                        for k in self.agent_keys},
            'rewards': {k: np.zeros((self.n_envs, self.max_eps_len), np.float32) for k in self.agent_keys},
            'returns': {k: np.zeros((self.n_envs, self.max_eps_len), np.float32) for k in self.agent_keys},
            'values': {k: np.zeros((self.n_envs, self.max_eps_len), np.float32) for k in self.agent_keys},
            'advantages': {k: np.zeros((self.n_envs, self.max_eps_len), np.float32) for k in self.agent_keys},
            'gate_log_pi_old': {k: np.zeros((self.n_envs, self.max_eps_len), np.float32) for k in self.agent_keys},
            'log_pi_old': {k: np.zeros((self.n_envs, self.max_eps_len), np.float32) for k in self.agent_keys},
            'terminals': {k: np.zeros((self.n_envs, self.max_eps_len), np.bool_) for k in self.agent_keys},
            'agent_mask': {k: np.zeros((self.n_envs, self.max_eps_len), np.bool_) for k in self.agent_keys},
            'filled': np.zeros((self.n_envs, self.max_eps_len), np.bool_)
        }
        if self.store_global_state:
            self.episode_data.update({
                'state': np.zeros((self.n_envs, self.max_eps_len) + self.state_space.shape, np.float32)
            })
        if self.use_actions_mask:
            self.episode_data.update({
                'avail_actions': {k: np.zeros((self.n_envs, self.max_eps_len) + self.avail_actions_shape[k],
                                              dtype=np.bool_) for k in self.agent_keys}
            })


class MeanField_OnPolicyBuffer(MARL_OnPolicyBuffer):
    """
    Replay buffer for Mean Field Actor-Critic algorithm.
    """

    def __init__(self, *args, **kwargs):
        self.n_actions_max = kwargs.get('n_actions_max')
        self.prob_space = (self.n_actions_max,)
        super(MeanField_OnPolicyBuffer, self).__init__(*args, **kwargs)

    def clear(self):
        super().clear()
        self.data["actions_mean"] = {k: create_memory(space2shape(self.prob_space), self.n_envs, self.n_size)
                                     for k in self.agent_keys}


class MeanField_OnPolicyBuffer_RNN(MARL_OnPolicyBuffer_RNN):
    def __init__(self, *args, **kwargs):
        self.n_actions_max = kwargs.get('n_actions_max')
        self.prob_space = (self.n_actions_max,)
        super(MeanField_OnPolicyBuffer_RNN, self).__init__(*args, **kwargs)

    def clear(self):
        super().clear()
        self.data['actions_mean'] = {k: np.zeros((self.buffer_size, self.max_eps_len) + self.prob_space,
                                                 dtype=np.float32) for k in self.agent_keys}

    def clear_episodes(self):
        super().clear_episodes()
        self.episode_data['actions_mean'] = {k: np.zeros((self.n_envs, self.max_eps_len) + self.prob_space,
                                                         dtype=np.float32) for k in self.agent_keys}


class MARL_OffPolicyBuffer(BaseBuffer):
    """
    Replay buffer for off-policy MARL algorithms.

    Args:
        agent_keys (List[str]): Keys that identify each agent.
        state_space (Dict[str, Space]): Global state space, type: Discrete, Box.
        obs_space (Dict[str, Dict[str, Space]]): Observation space for one agent (suppose same obs space for group agents).
        act_space (Dict[str, Dict[str, Space]]): Action space for one agent (suppose same actions space for group agents).
        n_envs (int): Number of parallel environments.
        buffer_size (int): Buffer size of total experience data.
        batch_size (int): Batch size of transition data for a sample.
        **kwargs: Other arguments.

    Example:
        >> state_space=None
        >> obs_space={'agent_0': Box(-inf, inf, (18,), float32),
                      'agent_1': Box(-inf, inf, (18,), float32),
                      'agent_2': Box(-inf, inf, (18,), float32)},
        >> act_space={'agent_0': Box(0.0, 1.0, (5,), float32),
                      'agent_1': Box(0.0, 1.0, (5,), float32),
                      'agent_2': Box(0.0, 1.0, (5,), float32)},
        >> n_envs=50,
        >> buffer_size=10000,
        >> batch_size=256,
        >> agent_keys=['agent_0', 'agent_1', 'agent_2'],
        >> memory = MARL_OffPolicyBuffer(agent_keys=agent_keys, state_space=state_space, obs_space=obs_space,
                                         act_space=act_space, n_envs=n_envs, buffer_size=buffer_size,
                                         batch_size=batch_size)
    """

    def __init__(self,
                 agent_keys: List[str],
                 state_space: Dict[str, Space] = None,
                 obs_space: Dict[str, Dict[str, Space]] = None,
                 act_space: Dict[str, Dict[str, Space]] = None,
                 n_envs: int = 1,
                 buffer_size: int = 1,
                 batch_size: int = 1,
                 **kwargs):
        super(MARL_OffPolicyBuffer, self).__init__(agent_keys, state_space, obs_space, act_space, n_envs, buffer_size)
        self.batch_size = batch_size
        self.store_global_state = False if self.state_space is None else True
        self.use_actions_mask = kwargs['use_actions_mask'] if 'use_actions_mask' in kwargs else False
        self.avail_actions_shape = kwargs['avail_actions_shape'] if 'avail_actions_shape' in kwargs else None
        self.data = {}
        self.clear()
        self.data_keys = self.data.keys()

    def clear(self):
        """
        Clears the memory data in the replay buffer.

        Example:
        An example shows the data shape: (n_env=50, buffer_size=10000, agent_keys=['agent_0', 'agent_1', 'agent_2']).
        self.data: {'obs': {'agent_0': shape=[50, 200, 18],
                            'agent_1': shape=[50, 200, 18],
                            'agent_2': shape=[50, 200, 18]},  # dim_obs: 18
                    'actions': {'agent_0': shape=[50, 200, 5],
                                'agent_1': shape=[50, 200, 5],
                                'agent_2': shape=[50, 200, 5]},  # dim_act: 5
                     ...}
        """
        reward_space = {key: () for key in self.agent_keys}
        terminal_space = {key: () for key in self.agent_keys}
        agent_mask_space = {key: () for key in self.agent_keys}

        self.data = {
            'obs': create_memory(space2shape(self.obs_space), self.n_envs, self.n_size),
            'actions': create_memory(space2shape(self.act_space), self.n_envs, self.n_size),
            'obs_next': create_memory(space2shape(self.obs_space), self.n_envs, self.n_size),
            'rewards': create_memory(reward_space, self.n_envs, self.n_size),
            'terminals': create_memory(terminal_space, self.n_envs, self.n_size, np.bool_),
            'agent_mask': create_memory(agent_mask_space, self.n_envs, self.n_size, np.bool_),
        }
        if self.store_global_state:
            self.data.update({
                'state': create_memory(space2shape(self.state_space), self.n_envs, self.n_size),
                'state_next': create_memory(space2shape(self.state_space), self.n_envs, self.n_size)
            })
        if self.use_actions_mask:
            self.data.update({
                "avail_actions": create_memory(self.avail_actions_shape, self.n_envs, self.n_size, np.bool_),
                "avail_actions_next": create_memory(self.avail_actions_shape, self.n_envs, self.n_size, np.bool_)
            })
        self.ptr, self.size = 0, 0

    def store(self, **step_data):
        """ Stores a step of data into the replay buffer. """
        for data_key, data_values in step_data.items():
            if data_key in ['state', 'state_next']:
                self.data[data_key][:, self.ptr] = data_values
                continue
            for agt_key in self.agent_keys:
                self.data[data_key][agt_key][:, self.ptr] = data_values[agt_key]
        self.ptr = (self.ptr + 1) % self.n_size
        self.size = np.min([self.size + 1, self.n_size])

    def sample(self, batch_size=None):
        """
        Samples a batch of data from the replay buffer.

        Parameters:
            batch_size (int): The size of the batch data to be sampled.

        Returns:
            samples_dict (dict): The sampled data.
        """
        assert self.size > 0, "Not enough transitions for off-policy buffer to random sample."
        if batch_size is None:
            batch_size = self.batch_size
        env_choices = np.random.choice(self.n_envs, batch_size)
        step_choices = np.random.choice(self.size, batch_size)
        samples_dict = {}
        for data_key in self.data_keys:
            if data_key in ['state', 'state_next']:
                samples_dict[data_key] = self.data[data_key][env_choices, step_choices]
                continue
            samples_dict[data_key] = {k: self.data[data_key][k][env_choices, step_choices] for k in self.agent_keys}
        samples_dict['batch_size'] = batch_size
        return samples_dict

    def finish_path(self, *args, **kwargs):
        return


class MARL_OffPolicyBuffer_RNN(MARL_OffPolicyBuffer):
    """
    Replay buffer for off-policy MARL algorithms with DRQN trick.

    Args:
        agent_keys (List[str]): Keys that identify each agent.
        state_space (Dict[str, Space]): Global state space, type: Discrete, Box.
        obs_space (Dict[str, Dict[str, Space]]): Observation space for one agent (suppose same obs space for group agents).
        act_space (Dict[str, Dict[str, Space]]): Action space for one agent (suppose same actions space for group agents).
        n_envs (int): Number of parallel environments.
        buffer_size (int): Buffer size of total experience data.
        batch_size (int): Batch size of episodes for a sample.
        max_episode_steps (int): The sequence length of each episode data.
        **kwargs: Other arguments.

    Example:
        $ state_space=None
        $ obs_space={'agent_0': Box(-inf, inf, (18,), float32),
                     'agent_1': Box(-inf, inf, (18,), float32),
                     'agent_2': Box(-inf, inf, (18,), float32)},
        $ act_space={'agent_0': Box(0.0, 1.0, (5,), float32),
                     'agent_1': Box(0.0, 1.0, (5,), float32),
                     'agent_2': Box(0.0, 1.0, (5,), float32)},
        $ n_envs=50,
        $ buffer_size=10000,
        $ batch_size=256,
        $ agent_keys=['agent_0', 'agent_1', 'agent_2'],
        $ max_episode_steps=60
        $ memory = MARL_OffPolicyBuffer_RNN(agent_keys=agent_keys, state_space=state_space,
                                            obs_space=obs_space, act_space=act_space,
                                            n_envs=n_envs, buffer_size=buffer_size, batch_size=batch_size,
                                            max_episode_steps=max_episode_steps,
                                            agent_keys=agent_keys)
    """

    def __init__(self,
                 agent_keys: List[str],
                 state_space: Dict[str, Space] = None,
                 obs_space: Dict[str, Dict[str, Space]] = None,
                 act_space: Dict[str, Dict[str, Space]] = None,
                 n_envs: int = 1,
                 buffer_size: int = 1,
                 batch_size: int = 1,
                 max_episode_steps: int = 1,
                 **kwargs):
        self.max_eps_len = max_episode_steps
        self.obs_shape = {k: space2shape(obs_space[k]) for k in agent_keys}
        self.act_shape = {k: space2shape(act_space[k]) for k in agent_keys}
        super(MARL_OffPolicyBuffer_RNN, self).__init__(agent_keys, state_space, obs_space, act_space,
                                                       n_envs, buffer_size, batch_size, **kwargs)
        self.episode_data = {}
        self.clear_episodes()
        self.episode_data_keys = self.episode_data.keys()

    def clear(self):
        """
        Clears the memory data in the replay buffer.

        Example:
        An example shows the data shape: (buffer_size=10000, max_eps_len=60,
                                          agent_keys=['agent_0', 'agent_1', 'agent_2']).
        self.data: {'obs': {'agent_0': shape=[10000, 61, 18],
                            'agent_1': shape=[10000, 61, 18],
                            'agent_2': shape=[10000, 61, 18]},  # dim_obs: 18
                    'actions': {'agent_0': shape=[10000, 60, 5],
                                'agent_1': shape=[10000, 60, 5],
                                'agent_2': shape=[10000, 60, 5]},  # dim_act: 5
                     ...
                     'filled': shape=[10000, 60],  # Step mask values. True means current step is not terminated.
                     }
        """
        self.data = {
            'obs': {k: np.zeros((self.buffer_size, self.max_eps_len + 1) + self.obs_shape[k], dtype=np.float32)
                    for k in self.agent_keys},
            'actions': {k: np.zeros((self.buffer_size, self.max_eps_len) + self.act_shape[k], dtype=np.float32)
                        for k in self.agent_keys},
            'rewards': {k: np.zeros((self.buffer_size, self.max_eps_len), dtype=np.float32) for k in self.agent_keys},
            'terminals': {k: np.zeros((self.buffer_size, self.max_eps_len), dtype=np.bool_) for k in self.agent_keys},
            'agent_mask': {k: np.zeros((self.buffer_size, self.max_eps_len), dtype=np.bool_) for k in self.agent_keys},
            'filled': np.zeros((self.buffer_size, self.max_eps_len), dtype=np.bool_),
        }

        if self.store_global_state:
            state_shape = (self.buffer_size, self.max_eps_len + 1) + space2shape(self.state_space)
            self.data.update({'state': np.zeros(state_shape, dtype=np.float32)})
        if self.use_actions_mask:
            self.data.update({
                'avail_actions': {k: np.zeros((self.buffer_size, self.max_eps_len + 1) + self.avail_actions_shape[k],
                                              dtype=np.bool_) for k in self.agent_keys}})
        self.ptr, self.size = 0, 0

    def clear_episodes(self):
        """
        Clears an episode of data for multiple environments in the replay buffer.

        Example:
        An example shows the data shape: (n_envs=16, max_eps_len=60, agent_keys=['agent_0', 'agent_1', 'agent_2']).
        self.data: {'obs': {'agent_0': shape=[16, 61, 18],
                            'agent_1': shape=[16, 61, 18],
                            'agent_2': shape=[16, 61, 18]},  # dim_obs: 18
                    'actions': {'agent_0': shape=[16, 60, 5],
                                'agent_1': shape=[16, 60, 5],
                                'agent_2': shape=[16, 60, 5]},  # dim_act: 5
                     ...
                     'filled': shape=[16, 60],  # Step mask values. True means current step is not terminated.
                     }
        """
        self.episode_data = {
            'obs': {k: np.zeros((self.n_envs, self.max_eps_len + 1) + self.obs_shape[k], dtype=np.float32)
                    for k in self.agent_keys},
            'actions': {k: np.zeros((self.n_envs, self.max_eps_len) + self.act_shape[k], dtype=np.float32)
                        for k in self.agent_keys},
            'rewards': {k: np.zeros((self.n_envs, self.max_eps_len), dtype=np.float32) for k in self.agent_keys},
            'terminals': {k: np.zeros((self.n_envs, self.max_eps_len), dtype=np.bool_) for k in self.agent_keys},
            'agent_mask': {k: np.zeros((self.n_envs, self.max_eps_len), dtype=np.bool_) for k in self.agent_keys},
            'filled': np.zeros((self.n_envs, self.max_eps_len), dtype=np.bool_),
        }

        if self.store_global_state:
            state_shape = (self.n_envs, self.max_eps_len + 1) + space2shape(self.state_space)
            self.episode_data.update({'state': np.zeros(state_shape, dtype=np.float32)})
        if self.use_actions_mask:
            self.episode_data.update({
                "avail_actions": {k: np.zeros((self.n_envs, self.max_eps_len + 1) + self.avail_actions_shape[k],
                                              dtype=np.bool_) for k in self.agent_keys}})

    def store(self, **step_data):
        """
        Stores a step of data for each environment.

        Parameters:
            step_data (dict): A dict of step data that to be stored into self.episode_data.
        """
        envs_step = step_data['episode_steps']
        envs_choice = range(self.n_envs)
        self.episode_data["filled"][envs_choice, envs_step] = True
        for data_key, data_value in step_data.items():
            if data_key not in self.episode_data:
                continue
            if data_key in ['state', 'state_next']:
                self.episode_data[data_key][envs_choice, envs_step] = data_value
                continue
            for agt_key in self.agent_keys:
                self.episode_data[data_key][agt_key][envs_choice, envs_step] = data_value[agt_key]

    def store_episodes(self, i_env):
        """
        Stores the episode of data for ith environment into the self.data.

        Parameters:
            i_env (int): The ith environment.
        """
        for data_key in self.data_keys:
            if data_key == "filled":
                self.data["filled"][self.ptr] = self.episode_data["filled"][i_env].copy()
                continue
            if data_key in ['state', 'state_next']:
                self.data[data_key][self.ptr] = self.episode_data[data_key][i_env].copy()
                continue
            for agt_key in self.agent_keys:
                self.data[data_key][agt_key][self.ptr] = self.episode_data[data_key][agt_key][i_env].copy()
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = np.min([self.size + 1, self.buffer_size])
        # clear the filled values for ith env.
        self.episode_data['filled'][i_env] = np.zeros(self.max_eps_len, dtype=np.bool_)

    def finish_path(self, i_env, **terminal_data):
        """
        Address the terminal states, including store the terminal observations, avail_actions, and others.

        Parameters:
            i_env (int): The i-th environment.
            terminal_data (dict): The terminal states.
        """
        env_step = terminal_data['episode_step']
        # Store terminal data into self.episode_data.
        if self.store_global_state:
            self.episode_data['state'][i_env, env_step] = terminal_data['state']
        for agt_key in self.agent_keys:
            self.episode_data['obs'][agt_key][i_env, env_step] = terminal_data['obs'][agt_key]
            if self.use_actions_mask:
                self.episode_data['avail_actions'][agt_key][i_env, env_step] = terminal_data['avail_actions'][agt_key]
        # Store the episode data of ith env into self.data.
        self.store_episodes(i_env)

    def sample(self, batch_size=None):
        """
        Samples a batch of data for model training.

        Parameters:
            batch_size (int): The size of the data batch, default is self.batch_size (recommended).

        Returns:
            samples_dict (dict): A dict of sampled data.
        """
        assert self.size > 0, "You need to first store experience data into the buffer!"
        if batch_size is None:
            batch_size = self.batch_size
        episode_choices = np.random.choice(self.size, batch_size)
        samples_dict = {}
        for data_key in self.data_keys:
            if data_key == "filled":
                samples_dict["filled"] = self.data['filled'][episode_choices]
                continue
            if data_key in ['state', 'state_next']:
                samples_dict[data_key] = self.data[data_key][episode_choices]
                continue
            samples_dict[data_key] = {k: self.data[data_key][k][episode_choices] for k in self.agent_keys}
        samples_dict['batch_size'] = batch_size
        samples_dict['sequence_length'] = self.max_eps_len
        return samples_dict


class MeanField_OffPolicyBuffer(MARL_OffPolicyBuffer):
    """
    Replay buffer for off-policy Mean-Field MARL algorithms (Mean-Field Q-Learning).

    Args:
        agent_keys (List[str]): Keys that identify each agent.
        state_space (Dict[str, Space]): Global state space, type: Discrete, Box.
        obs_space (Dict[str, Dict[str, Space]]): Observation space for one agent (suppose same obs space for group agents).
        act_space (Dict[str, Dict[str, Space]]): Action space for one agent (suppose same actions space for group agents).
        n_envs (int): Number of parallel environments.
        buffer_size (int): Buffer size of total experience data.
        batch_size (int): Batch size of transition data for a sample.
        **kwargs: Other arguments.

    Example:
        >> state_space=None
        >> obs_space={'agent_0': Box(-inf, inf, (18,), float32),
                      'agent_1': Box(-inf, inf, (18,), float32),
                      'agent_2': Box(-inf, inf, (18,), float32)},
        >> act_space={'agent_0': Box(0.0, 1.0, (5,), float32),
                      'agent_1': Box(0.0, 1.0, (5,), float32),
                      'agent_2': Box(0.0, 1.0, (5,), float32)},
        >> n_envs=50,
        >> buffer_size=10000,
        >> batch_size=256,
        >> agent_keys=['agent_0', 'agent_1', 'agent_2'],
        >> memory = MARL_OffPolicyBuffer(agent_keys=agent_keys, state_space=state_space, obs_space=obs_space,
                                         act_space=act_space, n_envs=n_envs, buffer_size=buffer_size,
                                         batch_size=batch_size)
    """

    def __init__(self,
                 agent_keys: List[str],
                 state_space: Dict[str, Space] = None,
                 obs_space: Dict[str, Dict[str, Space]] = None,
                 act_space: Dict[str, Dict[str, Space]] = None,
                 n_envs: int = 1,
                 buffer_size: int = 1,
                 batch_size: int = 1,
                 **kwargs):
        self.n_actions_max = kwargs.get('n_actions_max')
        self.prob_space = (self.n_actions_max,)
        super(MeanField_OffPolicyBuffer, self).__init__(agent_keys, state_space, obs_space, act_space,
                                                        n_envs, buffer_size, batch_size, **kwargs)

    def clear(self):
        super().clear()
        self.data["actions_mean"] = {k: create_memory(space2shape(self.prob_space), self.n_envs, self.n_size)
                                     for k in self.agent_keys}
        self.data["actions_mean_next"] = {k: create_memory(space2shape(self.prob_space), self.n_envs, self.n_size)
                                          for k in self.agent_keys}


class MeanField_OffPolicyBuffer_RNN(MARL_OffPolicyBuffer_RNN):
    def __init__(self, *args, **kwargs) -> None:
        self.n_actions_max = kwargs.get('n_actions_max')
        self.prob_space = (self.n_actions_max,)
        super(MeanField_OffPolicyBuffer_RNN, self).__init__(*args, **kwargs)

    def clear(self):
        super().clear()
        self.data['actions_mean'] = {k: np.zeros((self.buffer_size, self.max_eps_len + 1) + self.prob_space,
                                                 dtype=np.float32) for k in self.agent_keys}

    def clear_episodes(self):
        super().clear_episodes()
        self.episode_data['actions_mean'] = {k: np.zeros((self.n_envs, self.max_eps_len + 1) + self.prob_space,
                                                         dtype=np.float32) for k in self.agent_keys}

    def finish_path(self, i_env, **terminal_data):
        """
        Address the terminal states, including store the terminal observations, avail_actions, and others.

        Parameters:
            i_env (int): The i-th environment.
            terminal_data (dict): The terminal states.
        """
        env_step = terminal_data['episode_step']
        # Store terminal data into self.episode_data.
        if self.store_global_state:
            self.episode_data['state'][i_env, env_step] = terminal_data['state']
        for agt_key in self.agent_keys:
            self.episode_data['obs'][agt_key][i_env, env_step] = terminal_data['obs'][agt_key]
            self.episode_data['actions_mean'][agt_key][i_env, env_step] = terminal_data['actions_mean'][agt_key]
            if self.use_actions_mask:
                self.episode_data['avail_actions'][agt_key][i_env, env_step] = terminal_data['avail_actions'][agt_key]
        # Store the episode data of ith env into self.data.
        self.store_episodes(i_env)
