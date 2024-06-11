import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from xuance.environment import DummyVecEnv
from xuance.torch.learners import PerDQN_Learner
from xuance.torch.agents.qlearning_family import DQN_Agent
from xuance.common import PerOffPolicyBuffer


class PerDQN_Agent(DQN_Agent):
    """The implementation of Per-DQN agent.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv):
        super(PerDQN_Agent, self).__init__(config, envs)
        self.PER_beta0 = config.PER_beta0
        self.PER_beta = config.PER_beta0

        optimizer = torch.optim.Adam(self.policy.parameters(), self.config.learning_rate, eps=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0,
                                                         total_iters=self.config.running_steps)

        # Create experience replay buffer.
        self.auxiliary_info_shape = {}
        self.atari = True if config.env_name == "Atari" else False
        self.memory = PerOffPolicyBuffer(observation_space=self.observation_space,
                                         action_space=self.action_space,
                                         auxiliary_shape=self.auxiliary_info_shape,
                                         n_envs=self.n_envs,
                                         buffer_size=config.buffer_size,
                                         batch_size=config.batch_size,
                                         alpha=config.PER_alpha)
        self.learner = self._build_learner(self.config, envs.max_episode_steps, self.policy, optimizer, lr_scheduler)

    def _build_learner(self, *args):
        return PerDQN_Learner(*args)

    def train_epochs(self, n_epochs=1):
        train_info = {}
        for _ in range(n_epochs):
            samples = self.memory.sample(self.PER_beta)
            td_error, step_info = self.learner.update(**samples)
            self.memory.update_priorities(samples['step_choices'], td_error)
        train_info["epsilon-greedy"] = self.egreedy
        return train_info

    def train(self, train_steps):
        obs = self.envs.buf_obs
        for _ in tqdm(range(train_steps)):
            step_info = {}
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            acts = self.action(obs, self.egreedy)
            next_obs, rewards, terminals, trunctions, infos = self.envs.step(acts)

            self.memory.store(obs, acts, self._process_reward(rewards), terminals, self._process_observation(next_obs))
            if self.current_step > self.start_training and self.current_step % self.training_frequency == 0:
                train_info = self.train_epochs(n_epochs=1)
                self.log_infos(train_info, self.current_step)
                self.PER_beta += (1 - self.PER_beta0) / train_steps

            obs = deepcopy(next_obs)
            for i in range(self.n_envs):
                if terminals[i] or trunctions[i]:
                    if self.atari and (~trunctions[i]):
                        pass
                    else:
                        obs[i] = infos[i]["reset_obs"]
                        self.envs.buf_obs[i] = obs[i]
                        self.current_episode[i] += 1
                        if self.use_wandb:
                            step_info["Episode-Steps/env-%d" % i] = infos[i]["episode_step"]
                            step_info["Train-Episode-Rewards/env-%d" % i] = infos[i]["episode_score"]
                        else:
                            step_info["Episode-Steps"] = {"env-%d" % i: infos[i]["episode_step"]}
                            step_info["Train-Episode-Rewards"] = {"env-%d" % i: infos[i]["episode_score"]}
                        self.log_infos(step_info, self.current_step)

            self.current_step += self.n_envs
            if self.egreedy > self.end_greedy:
                self.egreedy -= self.delta_egreedy

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
            acts = self.action(obs, egreedy=0.0)
            next_obs, rewards, terminals, trunctions, infos = test_envs.step(acts)
            if self.config.render_mode == "rgb_array" and self.render:
                images = test_envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)

            obs = deepcopy(next_obs)
            for i in range(num_envs):
                if terminals[i] or trunctions[i]:
                    if self.atari and (~trunctions[i]):
                        pass
                    else:
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
            print("Best Score: %.2f" % (best_score))

        test_info = {
            "Test-Episode-Rewards/Mean-Score": np.mean(scores),
            "Test-Episode-Rewards/Std-Score": np.std(scores)
        }
        self.log_infos(test_info, self.current_step)

        test_envs.close()

        return scores
