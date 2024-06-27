import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from operator import itemgetter
from argparse import Namespace
from typing import List, Optional
from xuance.environment import DummyVecMultiAgentEnv
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.learners import ISAC_Learner
from xuance.torch.agents import MARLAgents
from xuance.torch.agents.multi_agent_rl.iddpg_agents import IDDPG_Agents
from xuance.common import MARL_OffPolicyBuffer, MARL_OffPolicyBuffer_RNN


class ISAC_Agents(IDDPG_Agents, MARLAgents):
    """The implementation of Independent SAC agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMultiAgentEnv):
        MARLAgents.__init__(self, config, envs)
        self.continuous_control = False
        # build policy, optimizers, schedulers
        self.policy = self._build_policy()
        optimizer, scheduler = {}, {}
        for key in self.model_keys:
            optimizer[key] = [torch.optim.Adam(self.policy.parameters_actor[key], self.config.lr_a, eps=1e-5),
                              torch.optim.Adam(self.policy.parameters_critic[key], self.config.lr_c, eps=1e-5)]
            scheduler[key] = [torch.optim.lr_scheduler.LinearLR(optimizer[key][0], start_factor=1.0, end_factor=0.5,
                                                                total_iters=self.config.running_steps),
                              torch.optim.lr_scheduler.LinearLR(optimizer[key][1], start_factor=1.0, end_factor=0.5,
                                                                total_iters=self.config.running_steps)]

        # create experience replay buffer
        buffer = MARL_OffPolicyBuffer_RNN if self.use_rnn else MARL_OffPolicyBuffer
        self.memory = buffer(agent_keys=self.agent_keys,
                             obs_space=self.observation_space,
                             act_space=self.action_space,
                             n_envs=self.n_envs,
                             buffer_size=self.config.buffer_size,
                             batch_size=self.config.batch_size,
                             n_actions=None,
                             use_actions_mask=False,
                             max_episode_steps=envs.max_episode_steps)

        # create learner
        self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, envs.max_episode_steps,
                                           self.policy, optimizer, scheduler)

    def _build_policy(self):
        """
        Build representation(s) and policy(ies) for agent(s)

        Returns:
            policy (torch.nn.Module): A dict of policies.
        """
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device
        agent = self.config.agent

        # build representations
        actor_representation = self._build_representation(self.config.representation, self.config)
        critic_representation = self._build_representation(self.config.representation, self.config)

        # build policies
        if self.config.policy == "Gaussian_ISAC_Policy":
            policy = REGISTRY_Policy["Gaussian_ISAC_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                actor_representation=actor_representation, critic_representation=critic_representation,
                actor_hidden_size=self.config.actor_hidden_size,
                critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                activation_action=ActivationFunctions[self.config.activation_action],
                device=device, use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
            self.continuous_control = True
        else:
            raise AttributeError(f"{agent} currently does not support the policy named {self.config.policy}.")

        return policy

    def _build_learner(self, config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler):
        return ISAC_Learner(config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler)

    def action(self,
               obs_dict: List[dict],
               rnn_hidden: Optional[dict] = None,
               test_mode: Optional[bool] = False):
        """
        Returns actions for agents.

        Parameters:
            obs_dict (List[dict]): Observations for each agent in self.agent_keys.
            rnn_hidden (Optional[dict]): The hidden variables of the RNN.
            test_mode (bool): True for testing without noises.

        Returns:
            rnn_hidden_state (dict): The new hidden states for RNN (if self.use_rnn=True).
            actions_dict (dict): The output actions.
        """
        batch_size = len(obs_dict)

        obs_input, agents_id, avail_actions_input = self._build_inputs(obs_dict)
        hidden_state, actions, _ = self.policy(observation=obs_input, agent_ids=agents_id, rnn_hidden=rnn_hidden)

        if self.use_parameter_sharing:
            key = self.model_keys[0]
            actions[key] = actions[key].reshape(batch_size, self.n_agents, -1).cpu().detach().numpy()
            actions_dict = [{k: actions[key][e, i] for i, k in enumerate(self.agent_keys)} for e in range(batch_size)]
        else:
            for key in self.agent_keys:
                actions[key] = actions[key].reshape(batch_size, -1).cpu().detach().numpy()
            actions_dict = [{k: actions[k][i] for k in self.agent_keys} for i in range(batch_size)]

        return hidden_state, actions_dict

    def train(self, n_steps):
        """
        Train the model for numerous steps.

        Parameters:
            n_steps (int): The number of steps to train the model.
        """
        if self.use_rnn:
            with tqdm(total=n_steps) as process_bar:
                step_start, step_last = deepcopy(self.current_step), deepcopy(self.current_step)
                n_steps_all = n_steps * self.n_envs
                while step_last - step_start < n_steps_all:
                    self.run_episodes(None, n_episodes=self.n_envs, test_mode=False)
                    if self.current_step >= self.start_training:
                        train_info = self.train_epochs(n_epochs=self.n_epochs)
                        self.log_infos(train_info, self.current_step)
                    process_bar.update((self.current_step - step_last) // self.n_envs)
                    step_last = deepcopy(self.current_step)
                process_bar.update(n_steps - process_bar.last_print_n)
            return

        obs_dict = self.envs.buf_obs
        for _ in tqdm(range(n_steps)):
            step_info = {}
            if self.current_step < self.start_training:
                actions_dict = [{k: self.action_space[k].sample() for k in self.agent_keys} for _ in range(self.n_envs)]
            else:
                _, actions_dict = self.action(obs_dict=obs_dict, test_mode=False)
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = self.envs.step(actions_dict)
            self.store_experience(obs_dict, actions_dict, next_obs_dict, rewards_dict, terminated_dict, info)
            if self.current_step >= self.start_training and self.current_step % self.training_frequency == 0:
                train_info = self.train_epochs(n_epochs=self.n_epochs)
                self.log_infos(train_info, self.current_step)
            obs_dict = deepcopy(next_obs_dict)

            for i in range(self.n_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    obs_dict[i] = info[i]["reset_obs"]
                    self.envs.buf_obs[i] = info[i]["reset_obs"]
                    if self.use_wandb:
                        step_info["Train-Results/Episode-Steps/env-%d" % i] = info[i]["episode_step"]
                        step_info["Train-Results/Episode-Rewards/env-%d" % i] = info[i]["episode_score"]
                    else:
                        step_info["Train-Results/Episode-Steps"] = {"env-%d" % i: info[i]["episode_step"]}
                        step_info["Train-Results/Episode-Rewards"] = {
                            "env-%d" % i: np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"]))}
                    self.log_infos(step_info, self.current_step)

            self.current_step += self.n_envs

    def run_episodes(self, env_fn=None, n_episodes: int = 1, test_mode: bool = False):
        """
        Run some episodes when use RNN.

        Parameters:
            env_fn: The function that can make some testing environments.
            n_episodes (int): Number of episodes.
            test_mode (bool): Whether to test the model.

        Returns:
            Scores: The episode scores.
        """
        envs = self.envs if env_fn is None else env_fn()
        num_envs = envs.num_envs
        videos, episode_videos = [[] for _ in range(num_envs)], []
        episode_count, scores, best_score = 0, [0.0 for _ in range(num_envs)], -np.inf
        obs_dict, info = envs.reset()
        if test_mode:
            if self.config.render_mode == "rgb_array" and self.render:
                images = envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)
        else:
            if self.use_rnn:
                self.memory.clear_episodes()
        rnn_hidden = self.init_rnn_hidden(num_envs)

        while episode_count < n_episodes:
            step_info = {}
            if not test_mode and (self.current_step < self.start_training):
                actions_dict = [{k: self.action_space[k].sample() for k in self.agent_keys} for _ in range(num_envs)]
            else:
                rnn_hidden, actions_dict = self.action(obs_dict=obs_dict, rnn_hidden=rnn_hidden, test_mode=test_mode)
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = envs.step(actions_dict)
            if test_mode:
                if self.config.render_mode == "rgb_array" and self.render:
                    images = envs.render(self.config.render_mode)
                    for idx, img in enumerate(images):
                        videos[idx].append(img)
            else:
                self.store_experience(obs_dict, actions_dict, next_obs_dict, rewards_dict, terminated_dict, info)
            obs_dict = deepcopy(next_obs_dict)

            for i in range(num_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    episode_count += 1
                    obs_dict[i] = info[i]["reset_obs"]
                    envs.buf_obs[i] = info[i]["reset_obs"]
                    if self.use_rnn:
                        rnn_hidden = self.init_hidden_item(i_env=i, rnn_hidden=rnn_hidden)
                        if not test_mode:
                            terminal_data = {'obs': next_obs_dict[i],
                                             'episode_step': info[i]['episode_step']}
                            self.memory.finish_path(i, **terminal_data)
                    episode_score = float(np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"])))
                    scores.append(episode_score)
                    if test_mode:
                        if best_score < episode_score:
                            best_score = episode_score
                            episode_videos = videos[i].copy()
                        if self.config.test_mode:
                            print("Episode: %d, Score: %.2f" % (episode_count, episode_score))
                    else:
                        if self.use_wandb:
                            step_info["Train-Results/Episode-Steps/env-%d" % i] = info[i]["episode_step"]
                            step_info["Train-Results/Episode-Rewards/env-%d" % i] = info[i]["episode_score"]
                        else:
                            step_info["Train-Results/Episode-Steps"] = {"env-%d" % i: info[i]["episode_step"]}
                            step_info["Train-Results/Episode-Rewards"] = {
                                "env-%d" % i: np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"]))}
                        self.current_step += info[i]["episode_step"]
                        self.log_infos(step_info, self.current_step)

        if test_mode:
            if self.config.render_mode == "rgb_array" and self.render:
                # time, height, width, channel -> time, channel, height, width
                videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
                self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)

            if self.config.test_mode:
                print("Best Score: %.2f" % best_score)

            test_info = {
                "Test-Results/Episode-Rewards": np.mean(scores),
                "Test-Results/Episode-Rewards-Std": np.std(scores),
            }

            self.log_infos(test_info, self.current_step)
            if env_fn is not None:
                envs.close()
        return scores

    def train_epochs(self, n_epochs=1):
        """
        Train the model for numerous epochs.

        Parameters:
            n_epochs (int): The number of epochs to train.

        Returns:
            info_train (dict): The information of training.
        """
        info_train = {}
        for i_epoch in range(n_epochs):
            sample = self.memory.sample()
            if self.use_rnn:
                info_train = self.learner.update_rnn(sample)
            else:
                info_train = self.learner.update(sample)
        return info_train
