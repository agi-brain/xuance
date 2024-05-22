import numpy as np
import torch
import random


class NStepReplayBuffer:
    def __init__(
        self,
        max_size,
        episode_len,
        n,
        policy_ids,
        agent_ids,
        policy_agents,
        policy_obs_dim,
        policy_act_dim,
        gamma,
    ):
        self.max_size = max_size
        self.episode_len = episode_len
        # n for n-step returns
        self.n = n
        self.policy_ids = policy_ids
        self.agent_ids = agent_ids
        self.policy_agents = policy_agents
        self.policy_buffers = {
            p_id: NStepPolicyBuffer(
                p_id,
                self.max_size,
                episode_len,
                n,
                self.policy_agents[p_id],
                policy_obs_dim[p_id],
                policy_act_dim[p_id],
                gamma,
            )
            for p_id in self.policy_ids
        }
        self.num_episodes = 0
        self.num_transitions = 0

    def push(
        self,
        t_env,
        observation_batch,
        action_batch,
        reward_batch,
        next_observation_batch,
        dones_batch,
        finish_episodes,
    ):
        batch_size = observation_batch.shape[0]

        observations = {
            a_id: np.vstack([obs[a_id] for obs in observation_batch])
            for a_id in self.agent_ids
        }
        actions = {
            a_id: np.vstack([act[a_id] for act in action_batch])
            for a_id in self.agent_ids
        }
        rewards = {
            a_id: np.vstack([rew[a_id] for rew in reward_batch])
            for a_id in self.agent_ids
        }
        n_observations = {
            a_id: np.vstack([nobs[a_id] for nobs in next_observation_batch])
            for a_id in self.agent_ids
        }
        if finish_episodes:
            dones = {
                a_id: np.ones_like(rewards[a_id]).astype(bool)
                for a_id in self.agent_ids
            }
        else:
            dones = {
                a_id: np.vstack([done[a_id] for done in dones_batch])
                for a_id in self.agent_ids
            }

        for p_id in self.policy_ids:
            self.policy_buffers[p_id].push(
                batch_size,
                t_env,
                observations,
                actions,
                rewards,
                n_observations,
                dones,
                finish_episodes,
            )

        assert (
            len(
                set(
                    [p_buffer.num_episodes for p_buffer in self.policy_buffers.values()]
                )
            )
            == 1
        )
        assert (
            len(
                set(
                    [
                        p_buffer.num_transitions
                        for p_buffer in self.policy_buffers.values()
                    ]
                )
            )
            == 1
        )

        self.num_episodes = self.policy_buffers[self.policy_ids[0]].num_episodes
        self.num_transitions = self.policy_buffers[self.policy_ids[0]].num_transitions

    def sample(self, batch_size):
        assert (
            self.num_transitions > batch_size
        ), "Cannot sample with no completed episodes in the buffer!"

        chunk_starts = np.random.choice(self.episode_len, batch_size)
        batch_inds = np.random.choice(self.num_episodes, batch_size)

        obs = {}
        act = {}
        rew = {}
        nobs = {}
        dones = {}

        for p_id in self.policy_ids:
            p_buffer = self.policy_buffers[p_id]
            o, a, r, no, d = p_buffer.get(batch_inds, chunk_starts)
            obs[p_id] = o
            act[p_id] = a
            rew[p_id] = r
            nobs[p_id] = no
            dones[p_id] = d

        return obs, act, rew, nobs, dones


class NStepPolicyBuffer:
    def __init__(
        self,
        policy_id,
        max_size,
        episode_len,
        n,
        policy_agents,
        obs_dim,
        act_dim,
        gamma,
    ):
        self.max_size = max_size
        self.n = n
        self.num_agents = len(policy_agents)
        self.policy_id = policy_id
        self.episode_len = episode_len
        self.agent_ids = policy_agents
        self.gamma = gamma
        random.shuffle(self.agent_ids)

        self.observations = np.zeros((self.num_agents, max_size, episode_len, obs_dim))
        self.actions = np.zeros((self.num_agents, max_size, episode_len, act_dim))
        self.rewards = np.zeros((self.num_agents, max_size, episode_len + n - 1, 1))
        self.next_observations = np.zeros(
            (self.num_agents, max_size, episode_len + n - 1, obs_dim)
        )
        self.dones = np.ones((self.num_agents, max_size, episode_len + n - 1, 1))

        self.num_episodes = 0
        self.num_transitions = 0

    def push(
        self,
        num_envs,
        t_env,
        observation_batch,
        action_batch,
        reward_batch,
        next_observation_batch,
        dones_batch,
        finish_episodes,
    ):
        assert t_env < self.episode_len

        if t_env == 0:
            # shuffle the agent ids at the start of a new episode batch
            random.shuffle(self.agent_ids)

        if t_env == 0 and self.num_episodes + num_envs > self.max_size:
            diff = self.num_episodes + num_envs - self.max_size
            self.observations = np.roll(self.observations, -diff, axis=1)
            self.actions = np.roll(self.actions, -diff, axis=1)
            self.rewards = np.roll(self.rewards, -diff, axis=1)
            self.next_observations = np.roll(self.next_observations, -diff, axis=1)
            self.dones = np.roll(self.dones, -diff, axis=1)

            self.num_episodes -= diff

        for i in range(self.num_agents):
            if finish_episodes:
                dones = np.ones_like(dones_batch[self.agent_ids[i]])
            else:
                dones = dones_batch[self.agent_ids[i]]

            self.observations[
                i, self.num_episodes : self.num_episodes + num_envs, t_env, :
            ] = observation_batch[self.agent_ids[i]]
            self.actions[
                i, self.num_episodes : self.num_episodes + num_envs, t_env, :
            ] = action_batch[self.agent_ids[i]]
            self.rewards[
                i, self.num_episodes : self.num_episodes + num_envs, t_env, :
            ] = reward_batch[self.agent_ids[i]]
            self.next_observations[
                i, self.num_episodes : self.num_episodes + num_envs, t_env, :
            ] = next_observation_batch[self.agent_ids[i]]
            self.dones[
                i, self.num_episodes : self.num_episodes + num_envs, t_env, :
            ] = dones

        self.num_transitions += num_envs
        if finish_episodes:
            self.num_episodes += num_envs

    def get(self, batch_inds, start_inds):
        batch_inds_col = batch_inds[:, None]
        start_inds_col = start_inds[:, None]
        nstep_inds = start_inds_col + np.arange(self.n)

        obs = self.observations[:, batch_inds, start_inds, :]
        acts = self.actions[:, batch_inds, start_inds, :]

        # get the n-step rewards and weight each by exponentiated discounts
        rews = self.rewards[:, batch_inds_col, nstep_inds, 0]
        rews = rews * np.power((np.ones(self.n) * self.gamma), np.arange(self.n))
        # sum the n-step rewards: rewards for terminal states are pre-set to 0, so don't need to mask
        rews = np.sum(rews, axis=2).reshape(self.num_agents, len(batch_inds), 1)
        # get the nobs of the nth
        nobs = self.next_observations[:, batch_inds, start_inds + self.n - 1, :]
        dones = self.dones[:, batch_inds, start_inds + self.n - 1, :]

        return (
            torch.from_numpy(obs),
            torch.from_numpy(acts),
            torch.from_numpy(rews),
            torch.from_numpy(nobs),
            torch.from_numpy(dones),
        )
