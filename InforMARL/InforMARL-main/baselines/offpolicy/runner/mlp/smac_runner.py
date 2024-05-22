import numpy as np
import torch
import time

from baselines.offpolicy.runner.mlp.base_runner import MlpRunner


class SMACRunner(MlpRunner):
    def __init__(self, config):
        """Runner class for the StarcraftII (SMAC) environment. See parent class for more information."""
        super(SMACRunner, self).__init__(config)
        # fill replay buffer with random actions
        self.finish_first_train_reset = False
        num_warmup_episodes = max(
            (self.batch_size / self.episode_length, self.args.num_random_episodes)
        )
        self.warmup(num_warmup_episodes)
        self.start = time.time()
        self.log_clear()

    @torch.no_grad()
    def eval(self):
        """Collect episodes to evaluate the policy."""
        self.trainer.prep_rollout()

        eval_infos = {}
        eval_infos["win_rate"] = []
        eval_infos["average_step_rewards"] = []

        for _ in range(self.args.num_eval_episodes):
            env_info = self.collecter(
                explore=False, training_episode=False, warmup=False
            )

            for k, v in env_info.items():
                eval_infos[k].append(v)

        self.log_env(eval_infos, suffix="eval_")

    def collect_rollout(self, explore=True, training_episode=True, warmup=False):
        """
        Collect a rollout and store it in the buffer. All agents share a single policy. Do training steps when appropriate
        :param explore: (bool) whether to use an exploration strategy when collecting the episoide.
        :param training_episode: (bool) whether this episode is used for evaluation or training.
        :param warmup: (bool) whether this episode is being collected during warmup phase.

        :return env_info: (dict) contains information about the rollout (total rewards, etc).
        """
        assert (
            self.share_policy
        ), "SC2 does not support individual agent policies currently!"
        env_info = {}
        p_id = "policy_0"
        policy = self.policies[p_id]

        env = self.env if explore else self.eval_env
        n_rollout_threads = self.num_envs if explore else self.num_eval_envs

        if not explore:
            obs, share_obs, avail_acts = env.reset()
        else:
            if self.finish_first_train_reset:
                obs = self.obs
                share_obs = self.share_obs
                avail_acts = self.avail_acts
            else:
                obs, share_obs, avail_acts = env.reset()
                self.finish_first_train_reset = True

        # init
        agent_deaths = np.zeros((self.num_envs, self.num_agents, 1))
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
            avail_acts_batch = np.concatenate(avail_acts)
            # get actions for all agents to step the env
            if warmup:
                # completely random actions in pre-training warmup phase
                acts_batch = policy.get_random_actions(obs_batch, avail_acts_batch)
            else:
                # get actions with exploration noise (eps-greedy/Gaussian)
                acts_batch, _ = policy.get_actions(
                    obs_batch,
                    avail_acts_batch,
                    t_env=self.total_env_steps,
                    explore=explore,
                )
            if not isinstance(acts_batch, np.ndarray):
                acts_batch = acts_batch.cpu().detach().numpy()
            env_acts = np.split(acts_batch, n_rollout_threads)

            # env step and store the relevant episode information
            next_obs, next_share_obs, rewards, dones, infos, next_avail_acts = env.step(
                env_acts
            )

            episode_rewards.append(rewards)
            dones_env = np.all(dones, axis=1)

            if explore and n_rollout_threads == 1 and np.all(dones_env):
                next_obs, next_share_obs, next_avail_acts = env.reset()

            if not explore and np.any(dones_env):
                assert (
                    n_rollout_threads == 1
                ), "only support one env for evaluation in smac domain."
                for i in range(n_rollout_threads):
                    if "won" in infos[i][0].keys():
                        if infos[i][0]["won"]:  # take one agent
                            env_info["win_rate"] = 1 if infos[i][0]["won"] else 0
                env_info["average_step_rewards"] = np.mean(episode_rewards)
                return env_info

            step_obs[p_id] = obs
            step_share_obs[p_id] = share_obs
            step_acts[p_id] = env_acts
            step_rewards[p_id] = rewards
            step_next_obs[p_id] = next_obs
            step_next_share_obs[p_id] = next_share_obs
            step_dones[p_id] = dones
            step_dones_env[p_id] = dones_env
            valid_transition[p_id] = 1 - agent_deaths
            step_avail_acts[p_id] = avail_acts
            step_next_avail_acts[p_id] = next_avail_acts

            obs = next_obs
            share_obs = next_share_obs
            avail_acts = next_avail_acts
            agent_deaths = dones

            if explore:
                self.obs = obs
                self.share_obs = share_obs
                self.avail_acts = avail_acts
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

        env_info["average_step_rewards"] = np.mean(episode_rewards)
        return env_info

    def log(self):
        """See parent class."""
        end = time.time()
        print(
            "\n Env {} Map {} Algo {} Exp {} runs total num timesteps {}/{}, FPS {}. \n".format(
                self.env_name,
                self.args.map_name,
                self.algorithm_name,
                self.args.experiment_name,
                self.total_env_steps,
                self.num_env_steps,
                int(self.total_env_steps / (end - self.start)),
            )
        )
        for p_id, train_info in zip(self.policy_ids, self.train_infos):
            self.log_train(p_id, train_info)

        self.log_env(self.env_infos)
        self.log_clear()

    def log_clear(self):
        """See parent class."""
        self.env_infos = {}

        self.env_infos["average_step_rewards"] = []
