import numpy as np
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from xuance.common import Optional, Union, DummyOffPolicyBuffer, DummyOffPolicyBuffer_Atari
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch import Module
from xuance.torch.agents.base import Agent, BaseCallback


class OffPolicyAgent(Agent):
    """The core class for off-policy algorithm with single agent.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
                It can be used for logging, early stopping, model saving, or visualization.
                If not provided, a default no-op callback is used.
    """
    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecEnv, SubprocVecEnv],
                 callback: Optional[BaseCallback] = None):
        super(OffPolicyAgent, self).__init__(config, envs, callback)
        self.start_greedy = getattr(config, "start_greedy", None)
        self.end_greedy = getattr(config, "end_greedy", None)
        self.e_greedy = self.start_greedy
        if (self.start_greedy is not None) and (self.end_greedy is not None):
            self.delta_egreedy = (self.start_greedy - self.end_greedy) / (config.decay_step_greedy / self.n_envs)
        else:
            self.delta_egreedy = None

        self.start_noise = getattr(config, "start_noise", None)
        self.end_noise = getattr(config, "end_noise", None)
        self.noise_scale = self.start_noise
        if (self.start_noise is not None) and (self.end_noise is not None):
            self.delta_noise = (self.start_noise - self.end_noise) / (config.running_steps / self.n_envs)
        else:
            self.delta_noise = None
        self.actions_low = getattr(self.action_space, "low", None)
        self.actions_high = getattr(self.action_space, "high", None)

        self.auxiliary_info_shape = None
        self.memory: Optional[DummyOffPolicyBuffer] = None

        self.buffer_size = self.config.buffer_size
        self.batch_size = self.config.batch_size

    def _build_memory(self, auxiliary_info_shape=None):
        self.atari = self.config.env_name == "Atari"
        Buffer = DummyOffPolicyBuffer_Atari if self.atari else DummyOffPolicyBuffer
        input_buffer = dict(observation_space=self.observation_space,
                            action_space=self.action_space,
                            auxiliary_shape=auxiliary_info_shape,
                            n_envs=self.n_envs,
                            buffer_size=self.buffer_size,
                            batch_size=self.batch_size)
        return Buffer(**input_buffer)

    def _build_policy(self) -> Module:
        raise NotImplementedError

    def _update_explore_factor(self):
        if self.e_greedy is not None:
            if self.e_greedy > self.end_greedy:
                self.e_greedy = self.start_greedy - self.current_step * self.delta_egreedy
        elif self.noise_scale is not None:
            if self.noise_scale >= self.end_noise:
                self.noise_scale = self.start_noise - self.current_step * self.delta_noise
        else:
            return

    def exploration(self, pi_actions):
        """Returns the actions for exploration.

        Parameters:
            pi_actions: The original output actions.

        Returns:
            explore_actions: The actions with noisy values.
        """
        if self.e_greedy is not None:
            random_actions = np.random.choice(self.action_space.n, self.n_envs)
            if np.random.rand() < self.e_greedy:
                explore_actions = random_actions
            else:
                explore_actions = pi_actions.detach().cpu().numpy()
        elif self.noise_scale is not None:
            explore_actions = pi_actions + np.random.normal(size=pi_actions.shape) * self.noise_scale
            explore_actions = np.clip(explore_actions, self.actions_low, self.actions_high)
        else:
            explore_actions = pi_actions.detach().cpu().numpy()
        return explore_actions

    def action(self, observations: np.ndarray,
               test_mode: Optional[bool] = False):
        """Returns actions and values.

        Parameters:
            observations (np.ndarray): The observation.
            test_mode (Optional[bool]): True for testing without noises.

        Returns:
            actions: The actions to be executed.
            values: The evaluated values.
            dists: The policy distributions.
            log_pi: Log of stochastic actions.
        """
        _, actions_output, _ = self.policy(observations)
        if test_mode:
            actions = actions_output.detach().cpu().numpy()
        else:
            actions = self.exploration(actions_output)
        return {"actions": actions}

    def train_epochs(self, n_epochs=1):
        train_info = {}
        for _ in range(n_epochs):
            samples = self.memory.sample()
            train_info = self.learner.update(**samples)
        train_info["epsilon-greedy"] = self.e_greedy
        train_info["noise_scale"] = self.noise_scale
        return train_info

    def train(self, train_steps):
        train_info = {}
        obs = self.envs.buf_obs
        for _ in tqdm(range(train_steps)):
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            policy_out = self.action(obs, test_mode=False)
            acts = policy_out['actions']
            next_obs, rewards, terminals, truncations, infos = self.envs.step(acts)

            self.callback.on_train_step(self.current_step, envs=self.envs, policy=self.policy,
                                        obs=obs, policy_out=policy_out, acts=acts, next_obs=next_obs, rewards=rewards,
                                        terminals=terminals, truncations=truncations, infos=infos,
                                        train_steps=train_steps)

            self.memory.store(obs, acts, self._process_reward(rewards), terminals, self._process_observation(next_obs))
            if self.current_step > self.start_training and self.current_step % self.training_frequency == 0:
                update_info = self.train_epochs(n_epochs=self.n_epochs)
                self.log_infos(update_info, self.current_step)
                train_info.update(update_info)
                self.callback.on_train_epochs_end(self.current_step, policy=self.policy, memory=self.memory,
                                                  current_episode=self.current_episode, train_steps=train_steps,
                                                  update_info=update_info)

            self.returns = self.gamma * self.returns + rewards
            obs = deepcopy(next_obs)
            for i in range(self.n_envs):
                if terminals[i] or truncations[i]:
                    if self.atari and (~truncations[i]):
                        pass
                    else:
                        obs[i] = infos[i]["reset_obs"]
                        self.envs.buf_obs[i] = obs[i]
                        self.ret_rms.update(self.returns[i:i + 1])
                        self.returns[i] = 0.0
                        self.current_episode[i] += 1
                        if self.use_wandb:
                            episode_info = {
                                f"Episode-Steps/rank_{self.rank}/env-{i}": infos[i]["episode_step"],
                                f"Train-Episode-Rewards/rank_{self.rank}/env-{i}": infos[i]["episode_score"]
                            }
                        else:
                            episode_info = {
                                f"Episode-Steps/rank_{self.rank}": {f"env-{i}": infos[i]["episode_step"]},
                                f"Train-Episode-Rewards/rank_{self.rank}": {f"env-{i}": infos[i]["episode_score"]}
                            }
                        self.log_infos(episode_info, self.current_step)
                        train_info.update(episode_info)
                        self.callback.on_train_episode_info(envs=self.envs, policy=self.policy, env_id=i,
                                                            infos=infos, rank=self.rank, use_wandb=self.use_wandb,
                                                            current_step=self.current_step,
                                                            current_episode=self.current_episode,
                                                            train_steps=train_steps)

            self.current_step += self.n_envs
            self._update_explore_factor()
            self.callback.on_train_step_end(self.current_step, envs=self.envs, policy=self.policy,
                                            train_steps=train_steps, train_info=train_info)
        return train_info

    def test(self, env_fn, test_episodes: int) -> list:
        test_envs = env_fn()
        num_envs = test_envs.num_envs
        videos, episode_videos, images = [[] for _ in range(num_envs)], [], None
        current_episode, current_step, scores, best_score = 0, 0, [], -np.inf
        obs, infos = test_envs.reset()
        if self.config.render_mode == "rgb_array" and self.render:
            images = test_envs.render(self.config.render_mode)
            for idx, img in enumerate(images):
                videos[idx].append(img)

        while current_episode < test_episodes:
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            policy_out = self.action(obs, test_mode=True)
            next_obs, rewards, terminals, truncations, infos = test_envs.step(policy_out['actions'])
            if self.config.render_mode == "rgb_array" and self.render:
                images = test_envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)

            self.callback.on_test_step(envs=test_envs, policy=self.policy, images=images,
                                       obs=obs, policy_out=policy_out, next_obs=next_obs, rewards=rewards,
                                       terminals=terminals, truncations=truncations, infos=infos,
                                       current_train_step=self.current_step,
                                       current_step=current_step, current_episode=current_episode)

            obs = deepcopy(next_obs)
            for i in range(num_envs):
                if terminals[i] or truncations[i]:
                    if self.atari and (~truncations[i]):
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
            current_step += num_envs

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

        self.callback.on_test_end(envs=test_envs, policy=self.policy,
                                  current_train_step=self.current_step,
                                  current_step=current_step, current_episode=current_episode,
                                  scores=scores, best_score=best_score)

        test_envs.close()

        return scores

