from typing import Dict
import wandb
import numpy as np
from itertools import chain
import torch
import time
from baselines.offpolicy.utils.util import is_multidiscrete
from baselines.offpolicy.runner.mlp.base_runner import MlpRunner


class MPERunner(MlpRunner):
    def __init__(self, config):
        """
        Runner class for the Multi-Agent Particle Env (MPE)  environment.
        See parent class for more information.
        """
        super(MPERunner, self).__init__(config)
        self.collecter = (
            self.shared_collect_rollout
            if self.share_policy
            else self.separated_collect_rollout
        )
        # fill replay buffer with random actions
        self.finish_first_train_reset = False
        num_warmup_episodes = max(
            (
                int(self.batch_size // self.episode_length) + 1,
                self.args.num_random_episodes,
            )
        )
        self.warmup(num_warmup_episodes)
        self.start = time.time()
        self.log_clear()

    @torch.no_grad()
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
    def shared_collect_rollout(self, explore=True, training_episode=True, warmup=False):
        """
        Collect a rollout and store it in the buffer. All agents share a single policy. Do training steps when appropriate
        :param explore: (bool) whether to use an exploration strategy when collecting the episoide.
        :param training_episode: (bool) whether this episode is used for evaluation or training.
        :param warmup: (bool) whether this episode is being collected during warmup phase.

        :return env_info: (dict) contains information about the rollout (total rewards, etc).
        """
        env_info = {}
        p_id = "policy_0"
        policy = self.policies[p_id]

        env = self.env if explore else self.eval_env
        n_rollout_threads = self.num_envs if explore else self.num_eval_envs

        if not explore:
            obs = env.reset()
            share_obs = obs.reshape(n_rollout_threads, -1)
        else:
            if self.finish_first_train_reset:
                obs = self.obs
                share_obs = self.share_obs
            else:
                obs = env.reset()
                share_obs = obs.reshape(n_rollout_threads, -1)
                self.finish_first_train_reset = True

        # init
        episode_rewards = []
        step_obs = {}
        step_share_obs = {}
        step_acts = {}
        step_rewards = {}
        step_next_obs = {}
        step_next_share_obs = {}
        step_dones = {}
        step_dones_env = {}
        valid_transition = {}
        step_avail_acts = {}
        step_next_avail_acts = {}

        for step in range(self.episode_length):
            obs_batch = np.concatenate(obs)
            # get actions for all agents to step the env
            if warmup:
                # completely random actions in pre-training warmup phase
                acts_batch = policy.get_random_actions(obs_batch)
            else:
                # get actions with exploration noise (eps-greedy/Gaussian)
                acts_batch, _ = policy.get_actions(
                    obs_batch, t_env=self.total_env_steps, explore=explore
                )

            if not isinstance(acts_batch, np.ndarray):
                acts_batch = acts_batch.cpu().detach().numpy()
            env_acts = np.split(acts_batch, n_rollout_threads)

            # env step and store the relevant episode information
            next_obs, rewards, dones, infos = env.step(env_acts)

            episode_rewards.append(rewards)
            dones_env = np.all(dones, axis=1)

            if explore and n_rollout_threads == 1 and np.all(dones_env):
                next_obs = env.reset()

            if not explore and np.all(dones_env):
                infos = self.process_env_infos(infos)
                env_info.update(infos)
                average_episode_rewards = np.mean(np.sum(episode_rewards, axis=0))
                env_info["average_episode_rewards"] = average_episode_rewards
                return env_info

            next_share_obs = next_obs.reshape(n_rollout_threads, -1)

            step_obs[p_id] = obs
            step_share_obs[p_id] = share_obs
            step_acts[p_id] = env_acts
            step_rewards[p_id] = rewards
            step_next_obs[p_id] = next_obs
            step_next_share_obs[p_id] = next_share_obs
            step_dones[p_id] = np.zeros_like(dones)
            step_dones_env[p_id] = dones_env
            valid_transition[p_id] = np.ones_like(dones)
            step_avail_acts[p_id] = None
            step_next_avail_acts[p_id] = None

            obs = next_obs
            share_obs = next_share_obs

            if explore:
                self.obs = obs
                self.share_obs = share_obs
                # push all episodes collected in this rollout step to the buffer
                self.buffer.insert(
                    n_rollout_threads,
                    step_obs,
                    step_share_obs,
                    step_acts,
                    step_rewards,
                    step_next_obs,
                    step_next_share_obs,
                    step_dones,
                    step_dones_env,
                    valid_transition,
                    step_avail_acts,
                    step_next_avail_acts,
                )

            # train
            if training_episode:
                self.total_env_steps += n_rollout_threads
                if (
                    self.last_train_T == 0
                    or (
                        (self.total_env_steps - self.last_train_T) / self.train_interval
                    )
                    >= 1
                ):
                    self.train()
                    self.total_train_steps += 1
                    self.last_train_T = self.total_env_steps

        infos = self.process_env_infos(infos)
        env_info.update(infos)
        average_episode_rewards = np.mean(np.sum(episode_rewards, axis=0))
        env_info["average_episode_rewards"] = average_episode_rewards

        return env_info

    # for mpe-simple_speaker_listener
    def separated_collect_rollout(
        self, explore=True, training_episode=True, warmup=False
    ):
        """
        Collect a rollout and store it in the buffer. Each agent has its own policy.. Do training steps when appropriate.
        :param explore: (bool) whether to use an exploration strategy when collecting the episoide.
        :param training_episode: (bool) whether this episode is used for evaluation or training.
        :param warmup: (bool) whether this episode is being collected during warmup phase.

        :return env_info: (dict) contains information about the rollout (total rewards, etc).
        """
        env_info = {}
        env = self.env if explore else self.eval_env
        n_rollout_threads = self.num_envs if explore else self.num_eval_envs

        if not explore:
            obs = env.reset()
            share_obs = []
            for o in obs:
                share_obs.append(list(chain(*o)))
            share_obs = np.array(share_obs)
        else:
            if self.finish_first_train_reset:
                obs = self.obs
                share_obs = self.share_obs
            else:
                obs = env.reset()
                share_obs = []
                for o in obs:
                    share_obs.append(list(chain(*o)))
                share_obs = np.array(share_obs)
                self.finish_first_train_reset = True

        agent_obs = []
        for agent_id in range(self.num_agents):
            env_obs = []
            for o in obs:
                env_obs.append(o[agent_id])
            env_obs = np.array(env_obs)
            agent_obs.append(env_obs)

        # [agents, parallel envs, dim]
        episode_rewards = []
        step_obs = {}
        step_share_obs = {}
        step_acts = {}
        step_rewards = {}
        step_next_obs = {}
        step_next_share_obs = {}
        step_dones = {}
        step_dones_env = {}
        valid_transition = {}
        step_avail_acts = {}
        step_next_avail_acts = {}

        acts = []
        for p_id in self.policy_ids:
            if is_multidiscrete(self.policy_info[p_id]["act_space"]):
                self.sum_act_dim = int(np.sum(self.policy_act_dim[p_id]))
            else:
                self.sum_act_dim = self.policy_act_dim[p_id]
            temp_act = np.zeros((n_rollout_threads, self.sum_act_dim))
            acts.append(temp_act)

        for step in range(self.episode_length):
            for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
                policy = self.policies[p_id]
                # get actions for all agents to step the env
                if warmup:
                    # completely random actions in pre-training warmup phase
                    # [parallel envs, agents, dim]
                    act = policy.get_random_actions(agent_obs[agent_id])
                else:
                    # get actions with exploration noise (eps-greedy/Gaussian)
                    act, _ = policy.get_actions(
                        agent_obs[agent_id], t_env=self.total_env_steps, explore=explore
                    )

                if not isinstance(act, np.ndarray):
                    act = act.cpu().detach().numpy()
                acts[agent_id] = act

            env_acts = []
            for i in range(n_rollout_threads):
                env_act = []
                for agent_id in range(self.num_agents):
                    env_act.append(acts[agent_id][i])
                env_acts.append(env_act)

            # env step and store the relevant episode information
            next_obs, rewards, dones, infos = env.step(env_acts)

            episode_rewards.append(rewards)
            dones_env = np.all(dones, axis=1)

            if explore and n_rollout_threads == 1 and np.all(dones_env):
                next_obs = env.reset()

            if not explore and np.all(dones_env):
                infos = self.process_env_infos(infos)
                env_info.update(infos)
                average_episode_rewards = np.mean(np.sum(episode_rewards, axis=0))
                env_info["average_episode_rewards"] = average_episode_rewards
                return env_info

            next_share_obs = []
            for no in next_obs:
                next_share_obs.append(list(chain(*no)))
            next_share_obs = np.array(next_share_obs)

            next_agent_obs = []
            for agent_id in range(self.num_agents):
                next_env_obs = []
                for no in next_obs:
                    next_env_obs.append(no[agent_id])
                next_env_obs = np.array(next_env_obs)
                next_agent_obs.append(next_env_obs)

            for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
                step_obs[p_id] = np.expand_dims(agent_obs[agent_id], axis=1)
                step_share_obs[p_id] = share_obs
                step_acts[p_id] = np.expand_dims(acts[agent_id], axis=1)
                step_rewards[p_id] = np.expand_dims(rewards[:, agent_id], axis=1)
                step_next_obs[p_id] = np.expand_dims(next_agent_obs[agent_id], axis=1)
                step_next_share_obs[p_id] = next_share_obs
                step_dones[p_id] = np.zeros_like(
                    np.expand_dims(dones[:, agent_id], axis=1)
                )
                step_dones_env[p_id] = dones_env
                valid_transition[p_id] = np.ones_like(
                    np.expand_dims(dones[:, agent_id], axis=1)
                )
                step_avail_acts[p_id] = None
                step_next_avail_acts[p_id] = None

            obs = next_obs
            agent_obs = next_agent_obs
            share_obs = next_share_obs

            if explore:
                self.obs = obs
                self.share_obs = share_obs
                self.buffer.insert(
                    n_rollout_threads,
                    step_obs,
                    step_share_obs,
                    step_acts,
                    step_rewards,
                    step_next_obs,
                    step_next_share_obs,
                    step_dones,
                    step_dones_env,
                    valid_transition,
                    step_avail_acts,
                    step_next_avail_acts,
                )

            # train
            if training_episode:
                self.total_env_steps += n_rollout_threads
                if (
                    self.last_train_T == 0
                    or (
                        (self.total_env_steps - self.last_train_T) / self.train_interval
                    )
                    >= 1
                ):
                    self.train()
                    self.total_train_steps += 1
                    self.last_train_T = self.total_env_steps

        infos = self.process_env_infos(infos)
        env_info.update(infos)
        average_episode_rewards = np.mean(np.sum(episode_rewards, axis=0))
        env_info["average_episode_rewards"] = average_episode_rewards

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

    def log_env(self, env_info, suffix=None):
        """See parent class."""
        for k, v in env_info.items():
            if type(v) == list:
                v = np.mean(v)
            suffix_k = k if suffix is None else suffix + k
            # hardcoded only for average_episode_rewards
            if "average_episode_rewards" in suffix_k:
                print(suffix_k + " is " + str(v))
            if self.use_wandb:
                wandb.log({suffix_k: v}, step=self.total_env_steps)
            else:
                self.writter.add_scalars(suffix_k, {suffix_k: v}, self.total_env_steps)

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

    @torch.no_grad()
    def warmup(self, num_warmup_episodes):
        # fill replay buffer with enough episodes to begin training
        self.trainer.prep_rollout()
        warmup_rewards = []
        print(f"Warming up for {num_warmup_episodes} episodes")
        for _ in range(int(num_warmup_episodes // self.num_envs) + 1):
            env_info = self.collecter(explore=True, training_episode=False, warmup=True)
            warmup_rewards.append(env_info["average_episode_rewards"])
        warmup_reward = np.mean(warmup_rewards)
        print(f"Warmup average episode rewards: {warmup_reward}")
