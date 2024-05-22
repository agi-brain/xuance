import numpy as np
from baselines.offpolicy.utils.util import get_dim_from_space
from baselines.offpolicy.utils.segment_tree import SumSegmentTree, MinSegmentTree


def _cast(x):
    return x.transpose(2, 0, 1, 3)


class RecReplayBuffer(object):
    def __init__(
        self,
        policy_info,
        policy_agents,
        buffer_size,
        episode_length,
        use_same_share_obs,
        use_avail_acts,
        use_reward_normalization=False,
    ):
        """
        Replay buffer class for training RNN policies. Stores entire episodes rather than single transitions.

        :param policy_info: (dict) maps policy id to a dict containing information about corresponding policy.
        :param policy_agents: (dict) maps policy id to list of agents controled by corresponding policy.
        :param buffer_size: (int) max number of transitions to store in the buffer.
        :param use_same_share_obs: (bool) whether all agents share the same centralized observation.
        :param use_avail_acts: (bool) whether to store what actions are available.
        :param use_reward_normalization: (bool) whether to use reward normalization.
        """
        self.policy_info = policy_info

        self.policy_buffers = {
            p_id: RecPolicyBuffer(
                buffer_size,
                episode_length,
                len(policy_agents[p_id]),
                self.policy_info[p_id]["obs_space"],
                self.policy_info[p_id]["share_obs_space"],
                self.policy_info[p_id]["act_space"],
                use_same_share_obs,
                use_avail_acts,
                use_reward_normalization,
            )
            for p_id in self.policy_info.keys()
        }

    def __len__(self):
        return self.policy_buffers["policy_0"].filled_i

    def insert(
        self,
        num_insert_episodes,
        obs,
        share_obs,
        acts,
        rewards,
        dones,
        dones_env,
        avail_acts,
    ):
        """
        Insert a set of episodes into buffer. If the buffer size overflows, old episodes are dropped.

        :param num_insert_episodes: (int) number of episodes to be added to buffer
        :param obs: (dict) maps policy id to numpy array of observations of agents corresponding to that policy
        :param share_obs: (dict) maps policy id to numpy array of centralized observation corresponding to that policy
        :param acts: (dict) maps policy id to numpy array of actions of agents corresponding to that policy
        :param rewards: (dict) maps policy id to numpy array of rewards of agents corresponding to that policy
        :param dones: (dict) maps policy id to numpy array of terminal status of agents corresponding to that policy
        :param dones_env: (dict) maps policy id to numpy array of terminal status of env
        :param valid_transition: (dict) maps policy id to numpy array of whether the corresponding transition is valid of agents corresponding to that policy
        :param avail_acts: (dict) maps policy id to numpy array of available actions of agents corresponding to that policy

        :return: (np.ndarray) indexes in which the new transitions were placed.
        """
        for p_id in self.policy_info.keys():
            idx_range = self.policy_buffers[p_id].insert(
                num_insert_episodes,
                np.array(obs[p_id]),
                np.array(share_obs[p_id]),
                np.array(acts[p_id]),
                np.array(rewards[p_id]),
                np.array(dones[p_id]),
                np.array(dones_env[p_id]),
                np.array(avail_acts[p_id]),
            )
        return idx_range

    def sample(self, batch_size):
        """
        Sample a set of episodes from buffer, uniformly at random.
        :param batch_size: (int) number of episodes to sample from buffer.

        :return: obs: (dict) maps policy id to sampled observations corresponding to that policy
        :return: share_obs: (dict) maps policy id to sampled observations corresponding to that policy
        :return: acts: (dict) maps policy id to sampled actions corresponding to that policy
        :return: rewards: (dict) maps policy id to sampled rewards corresponding to that policy
        :return: dones: (dict) maps policy id to sampled terminal status of agents corresponding to that policy
        :return: dones_env: (dict) maps policy id to sampled environment terminal status corresponding to that policy
        :return: valid_transition: (dict) maps policy_id to whether each sampled transition is valid or not (invalid if corresponding agent is dead)
        :return: avail_acts: (dict) maps policy_id to available actions corresponding to that policy
        """
        inds = np.random.choice(self.__len__(), batch_size)
        obs, share_obs, acts, rewards, dones, dones_env, avail_acts = (
            {},
            {},
            {},
            {},
            {},
            {},
            {},
        )
        for p_id in self.policy_info.keys():
            (
                obs[p_id],
                share_obs[p_id],
                acts[p_id],
                rewards[p_id],
                dones[p_id],
                dones_env[p_id],
                avail_acts[p_id],
            ) = self.policy_buffers[p_id].sample_inds(inds)

        return obs, share_obs, acts, rewards, dones, dones_env, avail_acts, None, None


class RecPolicyBuffer(object):
    def __init__(
        self,
        buffer_size,
        episode_length,
        num_agents,
        obs_space,
        share_obs_space,
        act_space,
        use_same_share_obs,
        use_avail_acts,
        use_reward_normalization=False,
    ):
        """
        Buffer class containing buffer data corresponding to a single policy.

        :param buffer_size: (int) max number of episodes to store in buffer.
        :param episode_length: (int) max length of an episode.
        :param num_agents: (int) number of agents controlled by the policy.
        :param obs_space: (gym.Space) observation space of the environment.
        :param share_obs_space: (gym.Space) centralized observation space of the environment.
        :param act_space: (gym.Space) action space of the environment.
        :use_same_share_obs: (bool) whether all agents share the same centralized observation.
        :use_avail_acts: (bool) whether to store what actions are available.
        :param use_reward_normalization: (bool) whether to use reward normalization.
        """
        self.buffer_size = buffer_size
        self.episode_length = episode_length
        self.num_agents = num_agents
        self.use_same_share_obs = use_same_share_obs
        self.use_avail_acts = use_avail_acts
        self.use_reward_normalization = use_reward_normalization
        self.filled_i = 0
        self.current_i = 0

        # obs
        if obs_space.__class__.__name__ == "Box":
            obs_shape = obs_space.shape
            share_obs_shape = share_obs_space.shape
        elif obs_space.__class__.__name__ == "list":
            obs_shape = obs_space
            share_obs_shape = share_obs_space
        else:
            raise NotImplementedError

        self.obs = np.zeros(
            (self.episode_length + 1, self.buffer_size, self.num_agents, obs_shape[0]),
            dtype=np.float32,
        )

        if self.use_same_share_obs:
            self.share_obs = np.zeros(
                (self.episode_length + 1, self.buffer_size, share_obs_shape[0]),
                dtype=np.float32,
            )
        else:
            self.share_obs = np.zeros(
                (
                    self.episode_length + 1,
                    self.buffer_size,
                    self.num_agents,
                    share_obs_shape[0],
                ),
                dtype=np.float32,
            )

        # action
        act_dim = np.sum(get_dim_from_space(act_space))
        self.acts = np.zeros(
            (self.episode_length, self.buffer_size, self.num_agents, act_dim),
            dtype=np.float32,
        )
        if self.use_avail_acts:
            self.avail_acts = np.ones(
                (self.episode_length + 1, self.buffer_size, self.num_agents, act_dim),
                dtype=np.float32,
            )

        # rewards
        self.rewards = np.zeros(
            (self.episode_length, self.buffer_size, self.num_agents, 1),
            dtype=np.float32,
        )

        # default to done being True
        self.dones = np.ones_like(self.rewards, dtype=np.float32)
        self.dones_env = np.ones(
            (self.episode_length, self.buffer_size, 1), dtype=np.float32
        )

    def __len__(self):
        return self.filled_i

    def insert(
        self,
        num_insert_episodes,
        obs,
        share_obs,
        acts,
        rewards,
        dones,
        dones_env,
        avail_acts=None,
    ):
        """
        Insert a set of episodes corresponding to this policy into buffer. If the buffer size overflows, old transitions are dropped.

        :param num_insert_steps: (int) number of transitions to be added to buffer
        :param obs: (np.ndarray) observations of agents corresponding to this policy.
        :param share_obs: (np.ndarray) centralized observations of agents corresponding to this policy.
        :param acts: (np.ndarray) actions of agents corresponding to this policy.
        :param rewards: (np.ndarray) rewards of agents corresponding to this policy.
        :param dones: (np.ndarray) terminal status of agents corresponding to this policy.
        :param dones_env: (np.ndarray) environment terminal status.
        :param valid_transition: (np.ndarray) whether each transition is valid or not (invalid if agent was dead during transition)
        :param avail_acts: (np.ndarray) available actions of agents corresponding to this policy.

        :return: (np.ndarray) indexes of the buffer the new transitions were placed in.
        """

        # obs: [step, episode, agent, dim]
        episode_length = acts.shape[0]
        assert episode_length == self.episode_length, "different dimension!"

        if self.current_i + num_insert_episodes <= self.buffer_size:
            idx_range = np.arange(self.current_i, self.current_i + num_insert_episodes)
        else:
            num_left_episodes = self.current_i + num_insert_episodes - self.buffer_size
            idx_range = np.concatenate(
                (
                    np.arange(self.current_i, self.buffer_size),
                    np.arange(num_left_episodes),
                )
            )

        if self.use_same_share_obs:
            # remove agent dimension since all agents share centralized observation
            share_obs = share_obs[:, :, 0]

        self.obs[:, idx_range] = obs.copy()
        self.share_obs[:, idx_range] = share_obs.copy()
        self.acts[:, idx_range] = acts.copy()
        self.rewards[:, idx_range] = rewards.copy()
        self.dones[:, idx_range] = dones.copy()
        self.dones_env[:, idx_range] = dones_env.copy()

        if self.use_avail_acts:
            self.avail_acts[:, idx_range] = avail_acts.copy()

        self.current_i = idx_range[-1] + 1
        self.filled_i = min(self.filled_i + len(idx_range), self.buffer_size)

        return idx_range

    def sample_inds(self, sample_inds):
        """
        Sample a set of transitions from buffer from the specified indices.
        :param sample_inds: (np.ndarray) indices of samples to return from buffer.

        :return: obs: (np.ndarray) sampled observations corresponding to that policy
        :return: share_obs: (np.ndarray) sampled observations corresponding to that policy
        :return: acts: (np.ndarray) sampled actions corresponding to that policy
        :return: rewards: (np.ndarray) sampled rewards corresponding to that policy
        :return: dones: (np.ndarray) sampled terminal status of agents corresponding to that policy
        :return: dones_env: (np.ndarray) sampled environment terminal status corresponding to that policy
        :return: valid_transition: (np.ndarray) whether each sampled transition in episodes are valid or not (invalid if corresponding agent is dead)
        :return: avail_acts: (np.ndarray) sampled available actions corresponding to that policy
        """

        obs = _cast(self.obs[:, sample_inds])
        acts = _cast(self.acts[:, sample_inds])
        if self.use_reward_normalization:
            # mean std
            # [length, envs, agents, 1]
            # [length, envs, 1]
            all_dones_env = np.tile(
                np.expand_dims(self.dones_env[:, : self.filled_i], -1),
                (1, 1, self.num_agents, 1),
            )
            first_step_dones_env = np.zeros((1, self.filled_i, self.num_agents, 1))
            curr_dones_env = np.concatenate(
                (first_step_dones_env, all_dones_env[: self.episode_length - 1])
            )
            temp_rewards = self.rewards[:, : self.filled_i].copy()
            temp_rewards[curr_dones_env == 1.0] = np.nan

            mean_reward = np.nanmean(temp_rewards)
            std_reward = np.nanstd(temp_rewards)
            rewards = _cast((self.rewards[:, sample_inds] - mean_reward) / std_reward)
        else:
            rewards = _cast(self.rewards[:, sample_inds])

        if self.use_same_share_obs:
            share_obs = self.share_obs[:, sample_inds]
        else:
            share_obs = _cast(self.share_obs[:, sample_inds])

        dones = _cast(self.dones[:, sample_inds])
        dones_env = self.dones_env[:, sample_inds]

        if self.use_avail_acts:
            avail_acts = _cast(self.avail_acts[:, sample_inds])
        else:
            avail_acts = None

        return obs, share_obs, acts, rewards, dones, dones_env, avail_acts


class PrioritizedRecReplayBuffer(RecReplayBuffer):
    def __init__(
        self,
        alpha,
        policy_info,
        policy_agents,
        buffer_size,
        episode_length,
        use_same_share_obs,
        use_avail_acts,
        use_reward_normalization=False,
    ):
        """Prioritized replay buffer class for training RNN policies. See parent class."""
        super(PrioritizedRecReplayBuffer, self).__init__(
            policy_info,
            policy_agents,
            buffer_size,
            episode_length,
            use_same_share_obs,
            use_avail_acts,
            use_reward_normalization,
        )
        self.alpha = alpha
        self.policy_info = policy_info
        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sums = {
            p_id: SumSegmentTree(it_capacity) for p_id in self.policy_info.keys()
        }
        self._it_mins = {
            p_id: MinSegmentTree(it_capacity) for p_id in self.policy_info.keys()
        }
        self.max_priorities = {p_id: 1.0 for p_id in self.policy_info.keys()}

    def insert(
        self,
        num_insert_episodes,
        obs,
        share_obs,
        acts,
        rewards,
        dones,
        dones_env,
        avail_acts=None,
    ):
        """See parent class."""
        idx_range = super().insert(
            num_insert_episodes,
            obs,
            share_obs,
            acts,
            rewards,
            dones,
            dones_env,
            avail_acts,
        )
        for idx in range(idx_range[0], idx_range[1]):
            for p_id in self.policy_info.keys():
                self._it_sums[p_id][idx] = self.max_priorities[p_id] ** self.alpha
                self._it_mins[p_id][idx] = self.max_priorities[p_id] ** self.alpha

        return idx_range

    def _sample_proportional(self, batch_size, p_id=None):
        total = self._it_sums[p_id].sum(0, len(self) - 1)
        mass = np.random.random(size=batch_size) * total
        idx = self._it_sums[p_id].find_prefixsum_idx(mass)
        return idx

    def sample(self, batch_size, beta=0, p_id=None):
        """
        Sample a set of episodes from buffer; probability of choosing a given episode is proportional to its priority.
        :param batch_size: (int) number of episodes to sample.
        :param beta: (float) controls the amount of prioritization to apply.
        :param p_id: (str) policy which will be updated using the samples.

        :return: See parent class.
        """
        assert (
            len(self) > batch_size
        ), "Cannot sample with no completed episodes in the buffer!"
        assert beta > 0

        batch_inds = self._sample_proportional(batch_size, p_id)

        p_min = self._it_mins[p_id].min() / self._it_sums[p_id].sum()
        max_weight = (p_min * len(self)) ** (-beta)
        p_sample = self._it_sums[p_id][batch_inds] / self._it_sums[p_id].sum()
        weights = (p_sample * len(self)) ** (-beta) / max_weight

        obs, share_obs, acts, rewards, dones, dones_env, avail_acts = (
            {},
            {},
            {},
            {},
            {},
            {},
            {},
        )
        for p_id in self.policy_info.keys():
            p_buffer = self.policy_buffers[p_id]
            (
                obs[p_id],
                share_obs[p_id],
                acts[p_id],
                rewards[p_id],
                dones[p_id],
                dones_env[p_id],
                avail_acts[p_id],
            ) = p_buffer.sample_inds(batch_inds)

        return (
            obs,
            share_obs,
            acts,
            rewards,
            dones,
            dones_env,
            avail_acts,
            weights,
            batch_inds,
        )

    def update_priorities(self, idxes, priorities, p_id=None):
        """
        Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self)

        self._it_sums[p_id][idxes] = priorities**self.alpha
        self._it_mins[p_id][idxes] = priorities**self.alpha

        self.max_priorities[p_id] = max(self.max_priorities[p_id], np.max(priorities))
