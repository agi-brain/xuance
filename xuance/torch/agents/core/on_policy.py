import numpy as np
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from xuance.common import Optional, Union, DummyOnPolicyBuffer, DummyOnPolicyBuffer_Atari
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch import Module
from xuance.torch.utils import split_distributions
from xuance.torch.agents.base import Agent, BaseCallback


class OnPolicyAgent(Agent):
    """The core class for on-policy algorithm with single agent.

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
        super(OnPolicyAgent, self).__init__(config, envs, callback)
        self.horizon_size = config.horizon_size
        self.n_epochs = config.n_epochs
        self.n_minibatch = config.n_minibatch
        self.gae_lam = config.gae_lambda
        self.auxiliary_info_shape = None
        self.memory: Optional[DummyOnPolicyBuffer] = None

    def _build_memory(self, auxiliary_info_shape=None):
        self.atari = True if self.config.env_name == "Atari" else False
        Buffer = DummyOnPolicyBuffer_Atari if self.atari else DummyOnPolicyBuffer
        self.buffer_size = self.n_envs * self.horizon_size
        self.batch_size = self.buffer_size // self.n_minibatch
        input_buffer = dict(observation_space=self.observation_space,
                            action_space=self.action_space,
                            auxiliary_shape=auxiliary_info_shape,
                            n_envs=self.n_envs,
                            horizon_size=self.horizon_size,
                            use_gae=self.config.use_gae,
                            use_advnorm=self.config.use_advnorm,
                            gamma=self.gamma,
                            gae_lam=self.gae_lam)
        return Buffer(**input_buffer)

    def _build_policy(self) -> Module:
        raise NotImplementedError

    def get_terminated_values(self, observations_next: np.ndarray, rewards: np.ndarray = None) -> np.ndarray:
        """Returns values for terminated states.

        Parameters:
            observations_next (np.ndarray): The terminal observations.
            rewards (np.ndarray): The rewards for terminated states.

        Returns:
            values_next: The values for terminal states.
        """
        policy_out = self.action(self._process_observation(observations_next))
        values_next = policy_out['values']
        return values_next

    def action(self, observations: np.ndarray,
               return_dists: bool = False, return_logpi: bool = False) -> dict:
        """Returns actions and values.

        Parameters:
            observations (np.ndarray): The observation.
            return_dists (bool): Whether to return dists.
            return_logpi (bool): Whether to return log_pi.

        Returns:
            actions: The actions to be executed.
            values: The evaluated values.
            dists: The policy distributions.
            log_pi: Log of stochastic actions.
        """
        _, policy_dists, values = self.policy(observations)
        actions = policy_dists.stochastic_sample()
        log_pi = policy_dists.log_prob(actions).detach().cpu().numpy() if return_logpi else None
        dists = split_distributions(policy_dists) if return_dists else None
        actions = actions.detach().cpu().numpy()
        if values is None:
            values = 0
        else:
            values = values.detach().cpu().numpy()
        return {"actions": actions, "values": values, "dists": dists, "log_pi": log_pi}

    def get_aux_info(self, policy_output: dict = None) -> dict:
        """Returns auxiliary information.

        Parameters:
            policy_output (dict): The output information of the policy.

        Returns:
            aux_info (dict): The auxiliary information.
        """
        return {}

    def train_epochs(self, n_epochs: int = 1) -> dict:
        indexes = np.arange(self.buffer_size)
        train_info = {}
        for _ in range(n_epochs):
            np.random.shuffle(indexes)
            for start in range(0, self.buffer_size, self.batch_size):
                end = start + self.batch_size
                sample_idx = indexes[start:end]
                samples = self.memory.sample(sample_idx)
                train_info = self.learner.update(**samples)
        return train_info

    def train(self, train_steps: int) -> dict:
        train_info = {}
        obs = self.envs.buf_obs
        for _ in tqdm(range(train_steps)):
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            policy_out = self.action(obs, return_dists=False, return_logpi=False)
            acts, vals = policy_out['actions'], policy_out['values']
            next_obs, rewards, terminals, truncations, infos = self.envs.step(acts)
            aux_info = self.get_aux_info()

            self.callback.on_train_step(self.current_step, envs=self.envs, policy=self.policy,
                                        obs=obs, policy_out=policy_out, acts=acts, vals=vals, next_obs=next_obs,
                                        rewards=rewards, terminals=terminals, truncations=truncations,
                                        infos=infos, aux_info=aux_info, train_steps=train_steps)

            self.memory.store(obs, acts, self._process_reward(rewards), vals, terminals, aux_info)
            if self.memory.full:
                vals = self.get_terminated_values(next_obs, rewards)
                for i in range(self.n_envs):
                    if terminals[i]:
                        self.memory.finish_path(0.0, i)
                    else:
                        self.memory.finish_path(vals[i], i)
                update_info = self.train_epochs(self.n_epochs)
                self.log_infos(update_info, self.current_step)
                train_info.update(update_info)
                self.callback.on_train_epochs_end(self.current_step, policy=self.policy, memory=self.memory,
                                                  current_episode=self.current_episode, train_steps=train_steps,
                                                  update_info=update_info)
                self.memory.clear()

            self.returns = self.gamma * self.returns + rewards
            obs = deepcopy(next_obs)
            for i in range(self.n_envs):
                if terminals[i] or truncations[i]:
                    self.ret_rms.update(self.returns[i:i + 1])
                    self.returns[i] = 0.0
                    if self.atari and (~truncations[i]):
                        pass
                    else:
                        if terminals[i]:
                            self.memory.finish_path(0, i)
                        else:
                            vals = self.get_terminated_values(next_obs, rewards)
                            self.memory.finish_path(vals[i], i)
                        obs[i] = infos[i]["reset_obs"]
                        self.envs.buf_obs[i] = obs[i]
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
            policy_out = self.action(obs)
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
            print("Best Score: %.2f" % (best_score))

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

