from typing import Dict
import numpy as np
import torch
import time

from baselines.offpolicy.runner.rnn.base_runner import RecRunner


class MPERunner(RecRunner):
    """Runner class for Multiagent Particle Envs (MPE). See parent class for more information."""

    def __init__(self, config):
        super(MPERunner, self).__init__(config)
        self.collecter = (
            self.shared_collect_rollout
            if self.share_policy
            else self.separated_collect_rollout
        )
        # fill replay buffer with random actions
        # fill replay buffer with random actions
        num_warmup_episodes = max((self.batch_size, self.args.num_random_episodes))
        self.warmup(num_warmup_episodes)
        self.start = time.time()
        self.log_clear()

    def eval(self):
        """Collect episodes to evaluate the policy."""
        self.trainer.prep_rollout()
        eval_infos = {}
        eval_infos["average_episode_rewards"] = []

        for i in range(self.args.num_eval_episodes):
            env_info = self.collecter(
                explore=False, training_episode=False, warmup=False
            )
            if i == 0:
                eval_infos = env_info
                # convert scalar to list for average_episode_rewards
                eval_infos["average_episode_rewards"] = [
                    eval_infos["average_episode_rewards"]
                ]
            else:
                for k, v in env_info.items():
                    if type(v) != list:
                        eval_infos[k].append(v)
                    else:
                        eval_infos[k].extend(v)

        self.log_env(eval_infos, suffix="eval_")

    # for mpe-simple_spread and mpe-simple_reference
    @torch.no_grad()
    def shared_collect_rollout(self, explore=True, training_episode=True, warmup=False):
        """
        Collect a rollout and store it in the buffer. All agents share a single policy.
        :param explore: (bool) whether to use an exploration strategy when collecting the episoide.
        :param training_episode: (bool) whether this episode is used for evaluation or training.
        :param warmup: (bool) whether this episode is being collected during warmup phase.

        :return env_info: (dict) contains information about the rollout (total rewards, etc).
        """
        env_info = {}
        # only 1 policy since all agents share weights
        p_id = "policy_0"
        policy = self.policies[p_id]

        env = self.env if training_episode or warmup else self.eval_env

        obs = env.reset()

        rnn_states_batch = np.zeros(
            (self.num_envs * self.num_agents, self.hidden_size), dtype=np.float32
        )
        last_acts_batch = np.zeros(
            (self.num_envs * self.num_agents, policy.output_dim), dtype=np.float32
        )

        # initialize variables to store episode information.
        episode_obs = {
            p_id: np.zeros(
                (
                    self.episode_length + 1,
                    self.num_envs,
                    self.num_agents,
                    policy.obs_dim,
                ),
                dtype=np.float32,
            )
            for p_id in self.policy_ids
        }
        episode_share_obs = {
            p_id: np.zeros(
                (
                    self.episode_length + 1,
                    self.num_envs,
                    self.num_agents,
                    policy.central_obs_dim,
                ),
                dtype=np.float32,
            )
            for p_id in self.policy_ids
        }
        episode_acts = {
            p_id: np.zeros(
                (
                    self.episode_length,
                    self.num_envs,
                    self.num_agents,
                    policy.output_dim,
                ),
                dtype=np.float32,
            )
            for p_id in self.policy_ids
        }
        episode_rewards = {
            p_id: np.zeros(
                (self.episode_length, self.num_envs, self.num_agents, 1),
                dtype=np.float32,
            )
            for p_id in self.policy_ids
        }
        episode_dones = {
            p_id: np.ones(
                (self.episode_length, self.num_envs, self.num_agents, 1),
                dtype=np.float32,
            )
            for p_id in self.policy_ids
        }
        episode_dones_env = {
            p_id: np.ones((self.episode_length, self.num_envs, 1), dtype=np.float32)
            for p_id in self.policy_ids
        }
        episode_avail_acts = {p_id: None for p_id in self.policy_ids}

        t = 0
        while t < self.episode_length:
            share_obs = obs.reshape(self.num_envs, -1)
            # group observations from parallel envs into one batch to process at once
            obs_batch = np.concatenate(obs)
            # get actions for all agents to step the env
            if warmup:
                # completely random actions in pre-training warmup phase
                acts_batch = policy.get_random_actions(obs_batch)
                # get new rnn hidden state
                _, rnn_states_batch, _ = policy.get_actions(
                    obs_batch, last_acts_batch, rnn_states_batch
                )
            else:
                # get actions with exploration noise (eps-greedy/Gaussian)
                acts_batch, rnn_states_batch, _ = policy.get_actions(
                    obs_batch,
                    last_acts_batch,
                    rnn_states_batch,
                    t_env=self.total_env_steps,
                    explore=explore,
                )
            acts_batch = (
                acts_batch
                if isinstance(acts_batch, np.ndarray)
                else acts_batch.cpu().detach().numpy()
            )
            # update rnn hidden state
            rnn_states_batch = (
                rnn_states_batch
                if isinstance(rnn_states_batch, np.ndarray)
                else rnn_states_batch.cpu().detach().numpy()
            )
            last_acts_batch = acts_batch

            env_acts = np.split(acts_batch, self.num_envs)
            # env step and store the relevant episode information
            next_obs, rewards, dones, infos = env.step(env_acts)
            if training_episode:
                self.total_env_steps += self.num_envs

            dones_env = np.all(dones, axis=1)
            terminate_episodes = np.any(dones_env) or t == self.episode_length - 1

            episode_obs[p_id][t] = obs
            episode_share_obs[p_id][t] = share_obs
            episode_acts[p_id][t] = np.stack(env_acts)
            episode_rewards[p_id][t] = rewards
            episode_dones[p_id][t] = dones
            episode_dones_env[p_id][t] = dones_env
            t += 1

            obs = next_obs

            if terminate_episodes:
                break

        episode_obs[p_id][t] = obs
        episode_share_obs[p_id][t] = obs.reshape(self.num_envs, -1)

        if explore:
            self.num_episodes_collected += self.num_envs
            # push all episodes collected in this rollout step to the buffer
            self.buffer.insert(
                self.num_envs,
                episode_obs,
                episode_share_obs,
                episode_acts,
                episode_rewards,
                episode_dones,
                episode_dones_env,
                episode_avail_acts,
            )

        infos = self.process_env_infos(infos)
        env_info.update(infos)
        average_episode_rewards = np.mean(np.sum(episode_rewards[p_id], axis=0))
        env_info["average_episode_rewards"] = average_episode_rewards

        return env_info

    # for mpe-simple_speaker_listener
    @torch.no_grad()
    def separated_collect_rollout(
        self, explore=True, training_episode=True, warmup=False
    ):
        """
        Collect a rollout and store it in the buffer. Each agent has its own policy.
        :param explore: (bool) whether to use an exploration strategy when collecting the episoide.
        :param training_episode: (bool) whether this episode is used for evaluation or training.
        :param warmup: (bool) whether this episode is being collected during warmup phase.

        :return env_info: (dict) contains information about the rollout (total rewards, etc).
        """
        env_info = {}
        env = self.env if training_episode or warmup else self.eval_env

        obs = env.reset()

        rnn_states = np.zeros(
            (self.num_agents, self.num_envs, self.hidden_size), dtype=np.float32
        )

        last_acts = {
            p_id: np.zeros(
                (
                    self.num_envs,
                    len(self.policy_agents[p_id]),
                    self.policies[p_id].output_dim,
                ),
                dtype=np.float32,
            )
            for p_id in self.policy_ids
        }
        episode_obs = {
            p_id: np.zeros(
                (
                    self.episode_length + 1,
                    self.num_envs,
                    len(self.policy_agents[p_id]),
                    self.policies[p_id].obs_dim,
                ),
                dtype=np.float32,
            )
            for p_id in self.policy_ids
        }
        episode_share_obs = {
            p_id: np.zeros(
                (
                    self.episode_length + 1,
                    self.num_envs,
                    len(self.policy_agents[p_id]),
                    self.policies[p_id].central_obs_dim,
                ),
                dtype=np.float32,
            )
            for p_id in self.policy_ids
        }
        episode_acts = {
            p_id: np.zeros(
                (
                    self.episode_length,
                    self.num_envs,
                    len(self.policy_agents[p_id]),
                    self.policies[p_id].output_dim,
                ),
                dtype=np.float32,
            )
            for p_id in self.policy_ids
        }
        episode_rewards = {
            p_id: np.zeros(
                (self.episode_length, self.num_envs, len(self.policy_agents[p_id]), 1),
                dtype=np.float32,
            )
            for p_id in self.policy_ids
        }
        episode_dones = {
            p_id: np.ones(
                (self.episode_length, self.num_envs, len(self.policy_agents[p_id]), 1),
                dtype=np.float32,
            )
            for p_id in self.policy_ids
        }
        episode_dones_env = {
            p_id: np.ones((self.episode_length, self.num_envs, 1), dtype=np.float32)
            for p_id in self.policy_ids
        }
        episode_avail_acts = {p_id: None for p_id in self.policy_ids}

        t = 0
        while t < self.episode_length:
            for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
                policy = self.policies[p_id]
                agent_obs = np.stack(obs[:, agent_id])
                share_obs = (
                    np.concatenate([obs[0, i] for i in range(self.num_agents)])
                    .reshape(self.num_envs, -1)
                    .astype(np.float32)
                )
                # get actions for all agents to step the env
                if warmup:
                    # completely random actions in pre-training warmup phase
                    # [parallel envs, agents, dim]
                    act = policy.get_random_actions(agent_obs)
                    # get new rnn hidden state
                    _, rnn_state, _ = policy.get_actions(
                        agent_obs, last_acts[p_id][:, 0], rnn_states[agent_id]
                    )
                else:
                    # get actions with exploration noise (eps-greedy/Gaussian)
                    if self.algorithm_name == "rmasac":
                        act, rnn_state, _ = policy.get_actions(
                            agent_obs,
                            last_acts[p_id],
                            rnn_states[agent_id],
                            sample=explore,
                        )
                    else:
                        act, rnn_state, _ = policy.get_actions(
                            agent_obs,
                            last_acts[p_id].squeeze(axis=0),
                            rnn_states[agent_id],
                            t_env=self.total_env_steps,
                            explore=explore,
                        )
                # update rnn hidden state
                rnn_states[agent_id] = (
                    rnn_state
                    if isinstance(rnn_state, np.ndarray)
                    else rnn_state.cpu().detach().numpy()
                )
                last_acts[p_id] = (
                    np.expand_dims(act, axis=1)
                    if isinstance(act, np.ndarray)
                    else np.expand_dims(act.cpu().detach().numpy(), axis=1)
                )

                episode_obs[p_id][t] = agent_obs
                episode_share_obs[p_id][t] = share_obs
                episode_acts[p_id][t] = act

            env_acts = []
            for i in range(self.num_envs):
                env_act = []
                for p_id in self.policy_ids:
                    env_act.append(last_acts[p_id][i, 0])
                env_acts.append(env_act)

            # env step and store the relevant episode information
            next_obs, rewards, dones, infos = env.step(env_acts)

            dones_env = np.all(dones, axis=1)
            terminate_episodes = np.any(dones_env) or t == self.episode_length - 1
            if terminate_episodes:
                dones_env = np.ones_like(dones_env).astype(bool)

            for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
                episode_rewards[p_id][t] = np.expand_dims(rewards[:, agent_id], axis=1)
                episode_dones[p_id][t] = np.expand_dims(dones[:, agent_id], axis=1)
                episode_dones_env[p_id][t] = dones_env

            obs = next_obs
            t += 1

            if training_episode:
                self.total_env_steps += self.num_envs

            if terminate_episodes:
                break

        for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
            episode_obs[p_id][t] = np.stack(obs[:, agent_id])
            episode_share_obs[p_id][t] = (
                np.concatenate([obs[0, i] for i in range(self.num_agents)])
                .reshape(self.num_envs, -1)
                .astype(np.float32)
            )

        if explore:
            self.num_episodes_collected += self.num_envs
            self.buffer.insert(
                self.num_envs,
                episode_obs,
                episode_share_obs,
                episode_acts,
                episode_rewards,
                episode_dones,
                episode_dones_env,
                episode_avail_acts,
            )

        average_episode_rewards = []
        for p_id in self.policy_ids:
            average_episode_rewards.append(
                np.mean(np.sum(episode_rewards[p_id], axis=0))
            )

        infos = self.process_env_infos(infos)
        env_info.update(infos)
        env_info["average_episode_rewards"] = np.mean(average_episode_rewards)

        return env_info

    def log(self):
        """See parent class."""
        end = time.time()
        print(
            f"Timesteps: {self.total_env_steps}/{self.num_env_steps} \t"
            f"FPS {int(self.total_env_steps / (end - self.start))} \t"
            f"Percentage complete {self.total_env_steps / self.num_env_steps * 100:.3f}"
        )
        for p_id, train_info in zip(self.policy_ids, self.train_infos):
            self.log_train(p_id, train_info)

        self.log_env(self.env_infos)
        self.log_clear()

    def log_clear(self):
        """See parent class."""
        self.env_infos = {}

        self.env_infos["average_episode_rewards"] = []

    def process_env_infos(self, infos: Dict):
        env_infos = {}
        for agent_id in range(self.num_agents):
            idv_rews = []
            dist_goals, time_to_goals, min_times_to_goal = [], [], []
            idv_collisions, obst_collisions = [], []
            # iterate through rollouts
            for info in infos:
                if "individual_reward" in info[agent_id].keys():
                    idv_rews.append(info[agent_id]["individual_reward"])
                if "Dist_to_goal" in info[agent_id].keys():
                    dist_goals.append(info[agent_id]["Dist_to_goal"])
                if "Time_req_to_goal" in info[agent_id].keys():
                    times = info[agent_id]["Time_req_to_goal"]
                    if times == -1:
                        times = self.args.episode_length * 0.1  # NOTE: Hardcoding `dt`
                    time_to_goals.append(times)
                if "Num_agent_collisions" in info[agent_id].keys():
                    idv_collisions.append(info[agent_id]["Num_agent_collisions"])
                if "Num_obst_collisions" in info[agent_id].keys():
                    obst_collisions.append(info[agent_id]["Num_obst_collisions"])
                if "Min_time_to_goal" in info[agent_id].keys():
                    min_times_to_goal.append(info[agent_id]["Min_time_to_goal"])

            agent_rew = f"agent{agent_id}/individual_rewards"
            times = f"agent{agent_id}/time_to_goal"
            dists = f"agent{agent_id}/dist_to_goal"
            agent_col = f"agent{agent_id}/num_agent_collisions"
            obst_col = f"agent{agent_id}/num_obstacle_collisions"
            min_times = f"agent{agent_id}/min_time_to_goal"

            env_infos[agent_rew] = idv_rews
            env_infos[times] = time_to_goals
            env_infos[min_times] = min_times_to_goal
            env_infos[dists] = dist_goals
            env_infos[agent_col] = idv_collisions
            env_infos[obst_col] = obst_collisions

        return env_infos
