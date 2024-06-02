import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from operator import itemgetter
from argparse import Namespace
from typing import Optional, List
from xuance.environment import DummyVecMultiAgentEnv
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.representations import REGISTRY_Representation
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.learners import IDDPG_Learner
from xuance.torch.agents import MARLAgents
from xuance.common import MARL_OffPolicyBuffer, space2shape


class IDDPG_Agents(MARLAgents):
    """The implementation of Independent DDPG agents.

    Args:
        config: The Namespace variable that provides hyper-parameters and other settings.
        envs: The vectorized environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMultiAgentEnv):
        super(IDDPG_Agents, self).__init__(config, envs)

        self.start_noise = config.start_noise
        self.end_noise = config.end_noise
        self.noise_scale = config.start_noise
        self.delta_noise = (self.start_noise - self.end_noise) / (config.running_steps / self.n_envs)

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
        self.memory = MARL_OffPolicyBuffer(agent_keys=self.agent_keys,
                                           obs_space=self.observation_space,
                                           act_space=self.action_space,
                                           n_envs=self.n_envs,
                                           buffer_size=self.config.buffer_size,
                                           batch_size=self.config.batch_size)

        # create learner
        self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, envs.max_episode_length,
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

        # build representations
        representation = {key: None for key in self.model_keys}
        for key in self.model_keys:
            input_shape = space2shape(self.observation_space[key])
            if self.config.representation == "Basic_Identical":
                representation[key] = REGISTRY_Representation["Basic_Identical"](input_shape=input_shape,
                                                                                 device=self.device)
            elif self.config.representation == "Basic_MLP":
                representation[key] = REGISTRY_Representation["Basic_MLP"](
                    input_shape=input_shape, hidden_sizes=self.config.representation_hidden_size,
                    normalize=normalize_fn, initialize=initializer, activation=activation, device=device)
            else:
                raise AttributeError(f"IDDPG currently does not support {self.config.representation} representation.")

        # build policies
        if self.config.policy == "Independent_DDPG_Policy":
            policy = REGISTRY_Policy["Independent_DDPG_Policy"](
                action_space=self.action_space, n_agents=self.n_agents, representation=representation,
                actor_hidden_size=self.config.actor_hidden_size,
                critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
                activation_action=ActivationFunctions[self.config.activation_action],
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys)
        else:
            raise AttributeError(f"IDDPG currently does not support the policy named {self.config.policy}.")

        return policy

    def _build_learner(self, config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler):
        return IDDPG_Learner(config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler)

    def store_experience(self, obs_dict, actions_dict, obs_next_dict, rewards_dict, terminals_dict, info):
        """
        Store experience data into replay buffer.

        Parameters:
            obs_dict (List[dict]): Observations for each agent in self.agent_keys.
            actions_dict (List[dict]): Actions for each agent in self.agent_keys.
            obs_next_dict (List[dict]): Next observations for each agent in self.agent_keys.
            rewards_dict (List[dict]): Rewards for each agent in self.agent_keys.
            terminals_dict (List[dict]): Terminated values for each agent in self.agent_keys.
            info (List[dict]): Other information for the environment at current step.
        """
        experience_data = {
            'obs': {k: np.array([data[k] for data in obs_dict]) for k in self.agent_keys},
            'actions': {k: np.array([data[k] for data in actions_dict]) for k in self.agent_keys},
            'obs_next': {k: np.array([data[k] for data in obs_next_dict]) for k in self.agent_keys},
            'rewards': {k: np.array([data[k] for data in rewards_dict]) for k in self.agent_keys},
            'terminals': {k: np.array([data[k] for data in terminals_dict]) for k in self.agent_keys},
            'agent_mask': {k: np.array([data['agent_mask'][k] for data in info]) for k in self.agent_keys},
        }
        self.memory.store(**experience_data)

    def action(self,
               obs_dict: List[dict],
               test_mode: Optional[bool] = False):
        """
        Returns actions for agents.

        Parameters:
            obs_dict (List[dict]): Observations for each agent in self.agent_keys.
            test_mode (bool): True for testing without noises.

        Returns:
            actions_dict (dict): The output actions.
        """
        n_env = len(obs_dict)
        if self.use_parameter_sharing:
            obs_input = {self.agent_keys[0]: np.array([itemgetter(*self.agent_keys)(env_obs) for env_obs in obs_dict])}
            agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(n_env, -1, -1).to(self.device)
            actions = self.policy(obs_input, agents_id)[self.agent_keys[0]]
            actions = actions.cpu().detach().numpy()
            if not test_mode:
                actions += np.random.normal(0, self.noise_scale, size=actions.shape)
            actions_dict = [{key: actions[e, agt] for agt, key in enumerate(self.agent_keys)} for e in range(n_env)]
        else:
            obs_input = {key: np.array([itemgetter(key)(env_obs) for env_obs in obs_dict]) for key in self.agent_keys}
            actions_dict_ = self.policy(obs_input)
            for key in self.agent_keys:
                actions_dict_[key] = actions_dict_[key].cpu().detach().numpy()
                if not test_mode:
                    actions_dict_[key] += np.random.normal(0, self.noise_scale, size=actions_dict_[key].shape)
            actions_dict = [{key: actions_dict_[key][i] for key in self.agent_keys} for i in range(n_env)]
        return actions_dict

    def train_epochs(self, n_epoch=1):
        """
        Train the model for numerous epochs.

        Parameters:
            n_epoch (int): The number of epochs to train.

        Returns:
            info_train (dict): The information of training.
        """
        info_train = {}
        for i_epoch in range(n_epoch):
            sample = self.memory.sample()
            if self.use_rnn:
                info_train = self.learner.update_rnn(sample)
            else:
                info_train = self.learner.update(sample)
        info_train["noise_scale"] = self.noise_scale
        return info_train

    def train(self, n_steps):
        """
        Train the model for numerous steps.

        Parameters:
            n_steps (int): The number of steps to train the model.
        """
        obs_dict = self.envs.buf_obs
        for _ in tqdm(range(n_steps)):
            step_info = {}
            if self.current_step < self.start_training:
                actions_dict = [{k: self.action_space[k].sample() for k in self.agent_keys} for _ in range(self.n_envs)]
            else:
                actions_dict = self.action(obs_dict, test_mode=False)
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = self.envs.step(actions_dict)
            self.store_experience(obs_dict, actions_dict, next_obs_dict, rewards_dict, terminated_dict, info)
            if self.current_step >= self.start_training and self.current_step % self.training_frequency == 0:
                train_info = self.train_epochs(n_epoch=1)
                self.log_infos(train_info, self.current_step)
            obs_dict = deepcopy(next_obs_dict)

            for i in range(self.n_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    obs_dict[i] = info[i]["reset_obs"]
                    self.envs.buf_obs[i] = info[i]["reset_obs"]
                    if self.use_wandb:
                        step_info["Episode-Steps/env-%d" % i] = info[i]["episode_step"]
                        step_info["Train-Episode-Rewards/env-%d" % i] = info[i]["episode_score"]
                    else:
                        step_info["Episode-Steps"] = {"env-%d" % i: info[i]["episode_step"]}
                        step_info["Train-Episode-Rewards"] = {
                            "env-%d" % i: np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"]))}
                    self.log_infos(step_info, self.current_step)

            self.current_step += self.n_envs
            if self.noise_scale >= self.end_noise:
                self.noise_scale = self.noise_scale - self.delta_noise

    def test(self, env_fn, n_episodes):
        """
        Test the model for some episodes.

        Parameters:
            env_fn: The function that can make some testing environments.
            n_episodes (int): Number of episodes to test.

        Returns:
            scores (List(float)): A list of cumulative rewards for each episode.
        """
        test_envs = env_fn()
        num_envs = test_envs.num_envs
        videos, episode_videos = [[] for _ in range(num_envs)], []
        current_episode, scores, best_score = 0, [0.0 for _ in range(num_envs)], -np.inf
        obs_dict, info = test_envs.reset()
        if self.config.render_mode == "rgb_array" and self.render:
            images = test_envs.render(self.config.render_mode)
            for idx, img in enumerate(images):
                videos[idx].append(img)

        while current_episode < n_episodes:
            actions_dict = self.action(obs_dict, test_mode=True)
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = test_envs.step(actions_dict)
            if self.config.render_mode == "rgb_array" and self.render:
                images = test_envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)

            obs_dict = deepcopy(next_obs_dict)
            for i in range(num_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    obs_dict[i] = info[i]["reset_obs"]
                    episode_score = float(np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"])))
                    scores.append(episode_score)
                    current_episode += 1
                    if best_score < episode_score:
                        best_score = episode_score
                        episode_videos = videos[i].copy()
                    if self.config.test_mode:
                        print("Episode: %d, Score: %.2f" % (current_episode, episode_score))

        if self.config.render_mode == "rgb_array" and self.render:
            # time, height, width, channel -> time, channel, height, width
            videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
            self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)

        if self.config.test_mode:
            print("Best Score: %.2f" % best_score)

        test_info = {
            "Test-Episode-Rewards/Mean-Score": np.mean(scores),
            "Test-Episode-Rewards/Std-Score": np.std(scores),
        }
        self.log_infos(test_info, self.current_step)
        test_envs.close()
        return scores
