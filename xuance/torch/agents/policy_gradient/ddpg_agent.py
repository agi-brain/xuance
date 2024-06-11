import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from xuance.environment import DummyVecEnv
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.learners import DDPG_Learner
from xuance.torch.agents import Agent
from xuance.common import DummyOffPolicyBuffer


class DDPG_Agent(Agent):
    """The implementation of DDPG agent.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv):
        super(DDPG_Agent, self).__init__(config, envs)
        self.start_noise, self.end_noise = config.start_noise, config.end_noise
        self.noise_scale = config.start_noise
        self.delta_noise = (self.start_noise - self.end_noise) / (config.running_steps / self.n_envs)

        # build policy, optimizer, lr_scheduler
        self.policy = self._build_policy()
        optimizers = {
            'actor': torch.optim.Adam(self.policy.actor_parameters, self.config.actor_learning_rate),
            'critic': torch.optim.Adam(self.policy.critic_parameters, self.config.critic_learning_rate)}
        lr_schedulers = {
            'actor': torch.optim.lr_scheduler.LinearLR(optimizers['actor'], start_factor=1.0, end_factor=0.25,
                                                       total_iters=self.config.running_steps),
            'critic': torch.optim.lr_scheduler.LinearLR(optimizers['critic'], start_factor=1.0, end_factor=0.25,
                                                        total_iters=self.config.running_steps)}

        # crate memory
        self.auxiliary_info_shape = {}
        self.memory = DummyOffPolicyBuffer(observation_space=self.observation_space,
                                           action_space=self.action_space,
                                           auxiliary_shape=self.auxiliary_info_shape,
                                           n_envs=self.n_envs,
                                           buffer_size=config.buffer_size,
                                           batch_size=config.batch_size)
        self.learner = self._build_learner(self.config, envs.max_episode_steps, self.policy, optimizers, lr_schedulers)

    def _build_policy(self):
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device

        # build representations.
        representation = self._build_representation(self.config.representation, self.config)

        # build policy
        if self.config.policy == "DDPG_Policy":
            policy = REGISTRY_Policy["DDPG_Policy"](
                action_space=self.action_space, representation=representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, device=device,
                activation=activation, activation_action=ActivationFunctions[self.config.activation_action])
        else:
            raise AttributeError(f"DDPG currently does not support the policy named {self.config.policy}.")

        return policy

    def _build_learner(self, *args):
        return DDPG_Learner(*args)

    def action(self, obs, noise_scale=0.0):
        _, action = self.policy(obs)
        action = action.detach().cpu().numpy()
        action = action + np.random.normal(size=action.shape) * noise_scale
        return np.clip(action, -1, 1)

    def train_epochs(self, n_epochs):
        train_info = {}
        for _ in range(n_epochs):
            samples = self.memory.sample()
            train_info = self.learner.update(**samples)
        train_info["noise_scale"] = self.noise_scale
        return train_info

    def train(self, train_steps):
        obs = self.envs.buf_obs
        for _ in tqdm(range(train_steps)):
            step_info = {}
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            acts = self.action(obs, self.noise_scale)
            if self.current_step < self.start_training:
                acts = [self.action_space.sample() for _ in range(self.n_envs)]
            next_obs, rewards, terminals, trunctions, infos = self.envs.step(acts)
            self.memory.store(obs, acts, self._process_reward(rewards), terminals, self._process_observation(next_obs))
            if self.current_step > self.start_training and self.current_step % self.training_frequency == 0:
                train_info = self.train_epochs(n_epochs=1)
                self.log_infos(train_info, self.current_step)

            self.returns = self.gamma * self.returns + rewards
            obs = deepcopy(next_obs)
            for i in range(self.n_envs):
                if terminals[i] or trunctions[i]:
                    obs[i] = infos[i]["reset_obs"]
                    self.envs.buf_obs[i] = obs[i]
                    self.ret_rms.update(self.returns[i:i + 1])
                    self.returns[i] = 0.0
                    self.current_episode[i] += 1
                    if self.use_wandb:
                        step_info["Episode-Steps/env-%d" % i] = infos[i]["episode_step"]
                        step_info["Train-Episode-Rewards/env-%d" % i] = infos[i]["episode_score"]
                    else:
                        step_info["Episode-Steps"] = {"env-%d" % i: infos[i]["episode_step"]}
                        step_info["Train-Episode-Rewards"] = {"env-%d" % i: infos[i]["episode_score"]}
                    self.log_infos(step_info, self.current_step)

            self.current_step += self.n_envs
            if self.noise_scale >= self.end_noise:
                self.noise_scale = self.noise_scale - self.delta_noise

    def test(self, env_fn, test_episodes):
        test_envs = env_fn()
        num_envs = test_envs.num_envs
        videos, episode_videos = [[] for _ in range(num_envs)], []
        current_episode, scores, best_score = 0, [], -np.inf
        obs, infos = test_envs.reset()
        if self.config.render_mode == "rgb_array" and self.render:
            images = test_envs.render(self.config.render_mode)
            for idx, img in enumerate(images):
                videos[idx].append(img)

        while current_episode < test_episodes:
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            acts = self.action(obs, noise_scale=0.0)
            next_obs, rewards, terminals, trunctions, infos = test_envs.step(acts)
            if self.config.render_mode == "rgb_array" and self.render:
                images = test_envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)

            obs = deepcopy(next_obs)
            for i in range(num_envs):
                if terminals[i] or trunctions[i]:
                    obs[i] = infos[i]["reset_obs"]
                    scores.append(infos[i]["episode_score"])
                    current_episode += 1
                    if best_score < infos[i]["episode_score"]:
                        best_score = infos[i]["episode_score"]
                        episode_videos = videos[i].copy()
                    if self.config.test_mode:
                        print("Episode: %d, Score: %.2f" % (current_episode, infos[i]["episode_score"]))

        if self.config.render_mode == "rgb_array" and self.render:
            # time, height, width, channel -> time, channel, height, width
            videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
            self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)

        if self.config.test_mode:
            print("Best Score: %.2f" % best_score)

        test_info = {
            "Test-Episode-Rewards/Mean-Score": np.mean(scores),
            "Test-Episode-Rewards/Std-Score": np.std(scores)
        }
        self.log_infos(test_info, self.current_step)

        test_envs.close()

        return scores
