import torch
import gym
import argparse
import numpy as np
from numpy import ndarray as arr
from typing import Optional, Tuple, Generator
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


class GraphReplayBuffer(object):
    """
    Buffer to store training data. For graph-based environments
    args: (argparse.Namespace)
        arguments containing relevant model, policy, and env information.
    num_agents: (int)
        number of agents in the env.
    num_entities: (int)
        number of entities in the env. This will be used for the `edge_list`
        size and `node_feats`
    obs_space: (gym.Space)
        observation space of agents.
    cent_obs_space: (gym.Space)
        centralized observation space of agents.
    node_obs_space: (gym.Space)
        node observation space of agents.
    agent_id_space: (gym.Space)
        observation space of agent ids.
    share_agent_id_space: (gym.Space)
        centralised observation space of agent ids.
    adj_space: (gym.Space)
        observation space of adjacency matrix.
    act_space: (gym.Space)
        action space for agents.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        num_agents: int,
        obs_space: gym.Space,
        cent_obs_space: gym.Space,
        node_obs_space: gym.Space,
        agent_id_space: gym.Space,
        share_agent_id_space: gym.Space,
        adj_space: gym.Space,
        act_space: gym.Space,
    ):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits

        # get shapes of observations
        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)
        node_obs_shape = get_shape_from_obs_space(node_obs_space)
        agent_id_shape = get_shape_from_obs_space(agent_id_space)
        if args.use_centralized_V:
            share_agent_id_shape = get_shape_from_obs_space(share_agent_id_space)
        else:
            share_agent_id_shape = get_shape_from_obs_space(agent_id_space)
        adj_shape = get_shape_from_obs_space(adj_space)
        ####################

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        self.share_obs = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                *share_obs_shape,
            ),
            dtype=np.float32,
        )
        self.obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape),
            dtype=np.float32,
        )
        # graph related stuff
        self.node_obs = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                *node_obs_shape,
            ),
            dtype=np.float32,
        )
        self.adj = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, *adj_shape),
            dtype=np.float32,
        )
        self.agent_id = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                *agent_id_shape,
            ),
            dtype=np.int,
        )
        self.share_agent_id = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                *share_agent_id_shape,
            ),
            dtype=np.int,
        )
        ####################

        self.rnn_states = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        self.rnn_states_critic = np.zeros_like(self.rnn_states)

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1),
            dtype=np.float32,
        )
        self.returns = np.zeros_like(self.value_preds)

        if act_space.__class__.__name__ == "Discrete":
            self.available_actions = np.ones(
                (
                    self.episode_length + 1,
                    self.n_rollout_threads,
                    num_agents,
                    act_space.n,
                ),
                dtype=np.float32,
            )
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape),
            dtype=np.float32,
        )
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape),
            dtype=np.float32,
        )
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1),
            dtype=np.float32,
        )

        self.masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1),
            dtype=np.float32,
        )
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0

    def insert(
        self,
        share_obs: arr,
        obs: arr,
        node_obs: arr,
        adj: arr,
        agent_id: arr,
        share_agent_id: arr,
        rnn_states_actor: arr,
        rnn_states_critic: arr,
        actions: arr,
        action_log_probs: arr,
        value_preds: arr,
        rewards: arr,
        masks: arr,
        bad_masks: arr = None,
        active_masks: arr = None,
        available_actions: arr = None,
    ) -> None:
        """
        Insert data into the buffer.
        share_obs: (argparse.Namespace)
            arguments containing relevant model, policy, and env information.
        obs: (np.ndarray)
            local agent observations. [num_rollouts, num_agents, obs_shape]
        node_obs: (np.ndarray)
            node features for the graph.
        adj: (np.ndarray)
            adjacency matrix for the graph.
            NOTE: needs post-processing to split
            into edge_feat and edge_attr
        agent_id: (np.ndarray)
            the agent id †o which the observation belong to
        share_agent_id: (np.ndarray)
            the agent id to which the shared_observations belong to
        rnn_states_actor: (np.ndarray)
            RNN states for actor network.
        rnn_states_critic: (np.ndarray)
            RNN states for critic network.
        actions:(np.ndarray)
            actions taken by agents.
        action_log_probs:(np.ndarray)
            log probs of actions taken by agents
        value_preds: (np.ndarray)
            value function prediction at each step.
        rewards: (np.ndarray)
            reward collected at each step.
        masks: (np.ndarray)
            denotes whether the environment has terminated or not.
        bad_masks: (np.ndarray)
            action space for agents.
        active_masks: (np.ndarray)
            denotes whether an agent is active or dead in the env.
        available_actions: (np.ndarray)
            actions available to each agent.
            If None, all actions are available.
        """
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.node_obs[self.step + 1] = node_obs.copy()
        self.adj[self.step + 1] = adj.copy()
        self.agent_id[self.step + 1] = agent_id.copy()
        self.share_agent_id[self.step + 1] = share_agent_id.copy()
        self.rnn_states[self.step + 1] = rnn_states_actor.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self) -> None:
        """Copy last timestep data to first index. Called after update to model."""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.node_obs[0] = self.node_obs[-1].copy()
        self.adj[0] = self.adj[-1].copy()
        self.agent_id[0] = self.agent_id[-1].copy()
        self.share_agent_id[0] = self.share_agent_id[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def compute_returns(
        self, next_value: arr, value_normalizer: Optional[PopArt] = None
    ) -> None:
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        next_value: (np.ndarray)
            value predictions for the step after the last episode step.
        value_normalizer: (PopArt)
            If not None, PopArt value normalizer instance.
        """
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        # step + 1
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(self.value_preds[step + 1])
                            * self.masks[step + 1]
                            - value_normalizer.denormalize(self.value_preds[step])
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * gae * self.masks[step + 1]
                        )
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * self.value_preds[step + 1]
                            * self.masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        self.returns[step] = (
                            self.returns[step + 1] * self.gamma * self.masks[step + 1]
                            + self.rewards[step]
                        ) * self.bad_masks[step + 1] + (
                            1 - self.bad_masks[step + 1]
                        ) * value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:
                        self.returns[step] = (
                            self.returns[step + 1] * self.gamma * self.masks[step + 1]
                            + self.rewards[step]
                        ) * self.bad_masks[step + 1] + (
                            1 - self.bad_masks[step + 1]
                        ) * self.value_preds[
                            step
                        ]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(self.value_preds[step + 1])
                            * self.masks[step + 1]
                            - value_normalizer.denormalize(self.value_preds[step])
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        self.returns[step] = gae + value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * self.value_preds[step + 1]
                            * self.masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = (
                        self.returns[step + 1] * self.gamma * self.masks[step + 1]
                        + self.rewards[step]
                    )

    def feed_forward_generator(
        self,
        advantages: arr,
        num_mini_batch: Optional[int] = None,
        mini_batch_size: Optional[int] = None,
    ) -> Generator[
        Tuple[
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
        ],
        None,
        None,
    ]:
        """
        Yield training data for MLP policies.
        advantages: (np.ndarray)
            advantage estimates.
        num_mini_batch: (int)
            number of minibatches to split the batch into.
        mini_batch_size: (int)
            number of samples in each minibatch.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                f"PPO requires the number of processes ({n_rollout_threads}) "
                f"* number of steps ({episode_length}) * number of agents "
                f"({num_agents}) = {n_rollout_threads*episode_length*num_agents} "
                "to be greater than or equal to the number of "
                f"PPO mini batches ({num_mini_batch})."
            )
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[3:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
        node_obs = self.node_obs[:-1].reshape(-1, *self.node_obs.shape[3:])
        adj = self.adj[:-1].reshape(-1, *self.adj.shape[3:])
        agent_id = self.agent_id[:-1].reshape(-1, *self.agent_id.shape[3:])
        share_agent_id = self.share_agent_id[:-1].reshape(
            -1, *self.share_agent_id.shape[3:]
        )
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(
            -1, *self.rnn_states_critic.shape[3:]
        )
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(
                -1, self.available_actions.shape[-1]
            )
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(
            -1, self.action_log_probs.shape[-1]
        )
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N M Dim]-->[T N M Dim]-->[T*N*M,Dim]-->[index,Dim]
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            node_obs_batch = node_obs[indices]
            adj_batch = adj[indices]
            agent_id_batch = agent_id[indices]
            share_agent_id_batch = share_agent_id[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield share_obs_batch, obs_batch, node_obs_batch, adj_batch, agent_id_batch, share_agent_id_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

    def naive_recurrent_generator(
        self, advantages: arr, num_mini_batch: int
    ) -> Generator[
        Tuple[
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
        ],
        None,
        None,
    ]:
        """
        Yield training data for non-chunked RNN training.
        advantages: (np.ndarray)
            advantage estimates.
        num_mini_batch: (int)
            number of minibatches to split the batch into.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * num_agents
        assert n_rollout_threads * num_agents >= num_mini_batch, (
            "PPO requires the number of processes ({})* number of agents ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(
                n_rollout_threads, num_agents, num_mini_batch
            )
        )
        num_envs_per_batch = batch_size // num_mini_batch
        perm = torch.randperm(batch_size).numpy()

        share_obs = self.share_obs.reshape(-1, batch_size, *self.share_obs.shape[3:])
        obs = self.obs.reshape(-1, batch_size, *self.obs.shape[3:])
        node_obs = self.node_obs.reshape(-1, batch_size, *self.node_obs.shape[3:])
        adj = self.adj.reshape(-1, batch_size, *self.adj.shape[3:])
        agent_id = self.agent_id.reshape(-1, batch_size, *self.agent_id.shape[3:])
        share_agent_id = self.share_agent_id.reshape(
            -1, batch_size, *self.share_agent_id.shape[3:]
        )
        rnn_states = self.rnn_states.reshape(-1, batch_size, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic.reshape(
            -1, batch_size, *self.rnn_states_critic.shape[3:]
        )
        actions = self.actions.reshape(-1, batch_size, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions.reshape(
                -1, batch_size, self.available_actions.shape[-1]
            )
        value_preds = self.value_preds.reshape(-1, batch_size, 1)
        returns = self.returns.reshape(-1, batch_size, 1)
        masks = self.masks.reshape(-1, batch_size, 1)
        active_masks = self.active_masks.reshape(-1, batch_size, 1)
        action_log_probs = self.action_log_probs.reshape(
            -1, batch_size, self.action_log_probs.shape[-1]
        )
        advantages = advantages.reshape(-1, batch_size, 1)

        for start_ind in range(0, batch_size, num_envs_per_batch):
            share_obs_batch = []
            obs_batch = []
            node_obs_batch = []
            adj_batch = []
            agent_id_batch = []
            share_agent_id_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                share_obs_batch.append(share_obs[:-1, ind])
                obs_batch.append(obs[:-1, ind])
                node_obs_batch.append(node_obs[:-1, ind])
                adj_batch.append(adj[:-1, ind])
                agent_id_batch.append(agent_id[:-1, ind])
                share_agent_id_batch.append(share_agent_id[:-1, ind])
                rnn_states_batch.append(rnn_states[0:1, ind])
                rnn_states_critic_batch.append(rnn_states_critic[0:1, ind])
                actions_batch.append(actions[:, ind])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[:-1, ind])
                value_preds_batch.append(value_preds[:-1, ind])
                return_batch.append(returns[:-1, ind])
                masks_batch.append(masks[:-1, ind])
                active_masks_batch.append(active_masks[:-1, ind])
                old_action_log_probs_batch.append(action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            # [N[T, dim]]
            T, N = self.episode_length, num_envs_per_batch
            # These are all from_numpys of size (T, N, -1)
            share_obs_batch = np.stack(share_obs_batch, 1)
            obs_batch = np.stack(obs_batch, 1)
            node_obs_batch = np.stack(node_obs_batch, 1)
            adj_batch = np.stack(adj_batch, 1)
            agent_id_batch = np.stack(agent_id_batch, 1)
            share_agent_id_batch = np.stack(share_agent_id_batch, 1)
            actions_batch = np.stack(actions_batch, 1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, 1)
            value_preds_batch = np.stack(value_preds_batch, 1)
            return_batch = np.stack(return_batch, 1)
            masks_batch = np.stack(masks_batch, 1)
            active_masks_batch = np.stack(active_masks_batch, 1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, 1)
            adv_targ = np.stack(adv_targ, 1)

            # States is just a (N, dim) from_numpy [N[1,dim]]
            rnn_states_batch = np.stack(rnn_states_batch).reshape(
                N, *self.rnn_states.shape[3:]
            )
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(
                N, *self.rnn_states_critic.shape[3:]
            )

            # Flatten the (T, N, ...) from_numpys to (T * N, ...)
            share_obs_batch = _flatten(T, N, share_obs_batch)
            obs_batch = _flatten(T, N, obs_batch)
            node_obs_batch = _flatten(T, N, node_obs_batch)
            adj_batch = _flatten(T, N, adj_batch)
            agent_id_batch = _flatten(T, N, agent_id_batch)
            share_agent_id_batch = _flatten(T, N, share_agent_id_batch)
            actions_batch = _flatten(T, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(T, N, value_preds_batch)
            return_batch = _flatten(T, N, return_batch)
            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(T, N, old_action_log_probs_batch)
            adv_targ = _flatten(T, N, adv_targ)

            yield share_obs_batch, obs_batch, node_obs_batch, adj_batch, agent_id_batch, share_agent_id_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

    def recurrent_generator(
        self, advantages: arr, num_mini_batch: int, data_chunk_length: int
    ) -> Generator[
        Tuple[
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
        ],
        None,
        None,
    ]:
        """
        Yield training data for chunked RNN training.
        advantages: (np.ndarray)
            advantage estimates.
        num_mini_batch: (int)
            number of minibatches to split the batch into.
        data_chunk_length: (int)
            length of sequence chunks with which to train RNN.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]
        mini_batch_size = data_chunks // num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]

        if len(self.share_obs.shape) > 4:
            share_obs = (
                self.share_obs[:-1]
                .transpose(1, 2, 0, 3, 4, 5)
                .reshape(-1, *self.share_obs.shape[3:])
            )
            obs = (
                self.obs[:-1]
                .transpose(1, 2, 0, 3, 4, 5)
                .reshape(-1, *self.obs.shape[3:])
            )
        else:
            share_obs = _cast(self.share_obs[:-1])
            obs = _cast(self.obs[:-1])

        node_obs = (
            self.node_obs[:-1]
            .transpose(1, 2, 0, 3, 4)
            .reshape(-1, *self.node_obs.shape[3:])
        )
        adj = self.adj[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.adj.shape[3:])

        agent_id = _cast(self.agent_id[:-1])
        share_agent_id = _cast(self.share_agent_id[:-1])

        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])
        # rnn_states = _cast(self.rnn_states[:-1])
        # rnn_states_critic = _cast(self.rnn_states_critic[:-1])
        rnn_states = (
            self.rnn_states[:-1]
            .transpose(1, 2, 0, 3, 4)
            .reshape(-1, *self.rnn_states.shape[3:])
        )
        rnn_states_critic = (
            self.rnn_states_critic[:-1]
            .transpose(1, 2, 0, 3, 4)
            .reshape(-1, *self.rnn_states_critic.shape[3:])
        )

        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1])

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            node_obs_batch = []
            adj_batch = []
            agent_id_batch = []
            share_agent_id_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:
                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                share_obs_batch.append(share_obs[ind : ind + data_chunk_length])
                obs_batch.append(obs[ind : ind + data_chunk_length])
                node_obs_batch.append(node_obs[ind : ind + data_chunk_length])
                adj_batch.append(adj[ind : ind + data_chunk_length])
                agent_id_batch.append(agent_id[ind : ind + data_chunk_length])
                share_agent_id_batch.append(
                    share_agent_id[ind : ind + data_chunk_length]
                )
                actions_batch.append(actions[ind : ind + data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(
                        available_actions[ind : ind + data_chunk_length]
                    )
                value_preds_batch.append(value_preds[ind : ind + data_chunk_length])
                return_batch.append(returns[ind : ind + data_chunk_length])
                masks_batch.append(masks[ind : ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind : ind + data_chunk_length])
                old_action_log_probs_batch.append(
                    action_log_probs[ind : ind + data_chunk_length]
                )
                adv_targ.append(advantages[ind : ind + data_chunk_length])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim)
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            obs_batch = np.stack(obs_batch, axis=1)
            node_obs_batch = np.stack(node_obs_batch, axis=1)
            adj_batch = np.stack(adj_batch, axis=1)
            agent_id_batch = np.stack(agent_id_batch, axis=1)
            share_agent_id_batch = np.stack(share_agent_id_batch, axis=1)

            actions_batch = np.stack(actions_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(
                N, *self.rnn_states.shape[3:]
            )
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(
                N, *self.rnn_states_critic.shape[3:]
            )

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            share_obs_batch = _flatten(L, N, share_obs_batch)
            obs_batch = _flatten(L, N, obs_batch)
            node_obs_batch = _flatten(L, N, node_obs_batch)
            adj_batch = _flatten(L, N, adj_batch)
            agent_id_batch = _flatten(L, N, agent_id_batch)
            share_agent_id_batch = _flatten(L, N, share_agent_id_batch)
            actions_batch = _flatten(L, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)

            yield share_obs_batch, obs_batch, node_obs_batch, adj_batch, agent_id_batch, share_agent_id_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch


def create_generator():
    mylist = range(3)
    for i in mylist:
        yield i * i
