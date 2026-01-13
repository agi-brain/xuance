import numpy as np
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from gymnasium.spaces import Space
from xuance.common import Optional, DummyOnPolicyBuffer, DummyOnPolicyBuffer_Atari, BaseCallback
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.tensorflow import Module
from xuance.tensorflow.utils import split_distributions
from xuance.tensorflow.agents.base import Agent


class OnPolicyAgent(Agent):
    """Base class for single-agent on-policy reinforcement learning algorithms.

    This class implements the common logic shared by on-policy algorithms (e.g., A2C, PPO, TRPO) in XuanCe.
    It extends the generic `Agent` abstraction with on-policy–specific components such as trajectory buffers, rollout
    collection, and multi-epoch policy/value updates.

    The agent can be used in both training and evaluation-only scenarios.
    When initialized without training environments (`envs=None`), the agent relies on explicitly provided observation
    and action spaces to construct policy networks, which is useful for inference or standalone evaluation.

    Args:
        config (Namespace): Configuration object containing hyperparameters, algorithm settings, and runtime options.
        envs (Optional[DummyVecEnv | SubprocVecEnv]): Vectorized environments
            used for training. If None, the agent will not initialize training
            environments and must be provided with `observation_space` and `action_space`.
        observation_space (Optional[gymnasium.spaces.Space]): Observation space
            specification used to build policy and value networks when `envs` is None.
            Typically obtained from `test_envs.observation_space`.
        action_space (Optional[gymnasium.spaces.Space]): Action space
            specification used to build policy and value networks when `envs` is None.
            Typically obtained from `test_envs.action_space`.
        callback (Optional[BaseCallback]): Optional callback object for injecting custom logic during training or
            evaluation, such as logging, early stopping, model checkpointing, or visualization.

    Notes:
        - On-policy agents collect fresh trajectories from the current policy and update the policy using rollouts
            stored in a trajectory buffer.
        - Training and evaluation environments are conceptually separated; evaluation environments may be created
            and managed externally.
        - In evaluation mode, actions are sampled without exploration schedules specific to training
            (e.g., no epsilon-greedy / action noise).
    """
    def __init__(
            self,
            config: Namespace,
            envs: Optional[DummyVecEnv | SubprocVecEnv] = None,
            observation_space: Optional[Space] = None,
            action_space: Optional[Space] = None,
            callback: Optional[BaseCallback] = None
    ):
        super(OnPolicyAgent, self).__init__(config, envs, observation_space, action_space, callback)
        self.horizon_size = config.horizon_size
        self.n_epochs = config.n_epochs
        self.n_minibatch = config.n_minibatch
        self.gae_lam = config.gae_lambda
        self.auxiliary_info_shape = None
        self.memory: Optional[DummyOnPolicyBuffer] = None

    def _build_memory(self, auxiliary_info_shape=None) -> DummyOnPolicyBuffer:
        """Build and initialize the on-policy trajectory buffer.

        This method creates a trajectory buffer instance used to store rollouts collected from the current policy.
        For Atari environments, a specialized buffer implementation is used to handle image-based observations;
        otherwise, a standard on-policy buffer is constructed.

        Args:
            auxiliary_info_shape (Optional[tuple]): Shape of auxiliary information to be stored alongside transitions
            in the buffer (e.g., additional state features or metadata). If None, no auxiliary information is stored.

        Returns:
            DummyOnPolicyBuffer: An initialized trajectory buffer instance configured with the current observation
                space, action space, number of parallel environments, horizon size, and GAE/advantage settings.

        Notes:
            - The buffer type is selected automatically based on whether the environment is an Atari environment.
            - The buffer stores rollouts of length `horizon_size` for each parallel environment
                and is cleared after each update cycle.
            - When `use_gae` is enabled, the buffer computes advantages using `gamma` and `gae_lam`;
                when `use_advnorm` is enabled, advantages are normalized before updates.
        """
        self.atari = self.config.env_name == "Atari"
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
        """Compute value estimates for terminal/terminated states.

        This method evaluates the value function on terminal observations and returns the value estimates used for
        bootstrapping (e.g., when finishing a trajectory segment).

        Args:
            observations_next (np.ndarray): Observations at the terminal step
                (or the next observations used for bootstrapping).
            rewards (Optional[np.ndarray]): Rewards corresponding to the terminal transitions.
                This argument is reserved for algorithm-specific implementations and may be unused.

        Returns:
            np.ndarray: Value estimates for the provided terminal observations.
        """
        policy_out = self.action(self._process_observation(observations_next))
        values_next = policy_out['values']
        return values_next

    def action(self, observations: np.ndarray,
               return_dists: bool = False, return_logpi: bool = False) -> dict:
        """Compute actions and value estimates for a batch of observations.

        This method performs a forward pass through the current policy to obtain action distributions
        and value predictions. Actions are sampled stochastically from the policy distribution.

        Args:
            observations (np.ndarray): Batch of observations. The array is expected to have shape compatible with
                the underlying policy.
            return_dists (bool): Whether to return the action distributions (split into a Python-friendly structure).
            return_logpi (bool): Whether to return the log-probabilities of the sampled actions.

        Returns:
            dict: A dictionary containing:
                - actions (np.ndarray): Sampled actions to execute in the environment(s).
                - values (np.ndarray): Value estimates for the input observations.
                    If the policy does not produce values, this is set to 0.
                - dists (Optional[Any]): Action distributions (when `return_dists=True`); otherwise None.
                - log_pi (Optional[np.ndarray]): Log-probabilities of sampled actions (when `return_logpi=True`);
                    otherwise None.
        """
        if self.policy.is_continuous:
            _, mu, std, values = self.policy(observations)
            policy_dists = self.policy.actor.distribution(mu=mu, std=std)
        else:
            _, logits, values = self.policy(observations)
            policy_dists = self.policy.actor.distribution(logits=logits)
        actions = policy_dists.stochastic_sample()
        log_pi = policy_dists.log_prob(actions)

        # log_pi = policy_dists.log_prob(actions).numpy() if return_logpi else None
        dists = split_distributions(policy_dists) if return_dists else None
        actions = actions.numpy()
        if values is None:
            values = 0
        else:
            values = values.numpy()
        return {"actions": actions, "values": values, "dists": dists, "log_pi": log_pi}

    def get_aux_info(self, policy_output: dict = None) -> dict:
        """Returns auxiliary information.

        Args:
            policy_output (dict): The output information of the policy.

        Returns:
            aux_info (dict): The auxiliary information.
        """
        return {}

    def train_epochs(self, n_epochs: int = 1) -> dict:
        """Update the policy for multiple epochs using samples from the rollout buffer.

        This method performs multiple passes over the collected rollout data in `self.memory`.  For each epoch,
        it shuffles transition indices and iterates over mini-batches to compute gradient updates via the learner.

        Args:
            n_epochs (int): Number of optimization epochs to perform over the current rollout buffer.

        Returns:
            dict: A dictionary of training metrics returned by the learner from the last mini-batch update
                (e.g., policy loss, value loss, entropy, KL divergence). Implementations may include additional
                diagnostics depending on the algorithm.
        """
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
        """Run the main on-policy training loop.

        This method interacts with the training environments to collect rollouts from the current policy, stores
        transitions in the on-policy trajectory buffer, and triggers policy/value updates when the buffer is full.
        The loop advances in vectorized steps (one iteration corresponds to `self.n_envs` environment steps).

        Args:
            train_steps (int): Number of rollout collection iterations to run. Each iteration steps all vectorized
                environments once, so the total number of environment steps is approximately
                `train_steps * self.n_envs`.

        Returns:
            dict: A dictionary containing aggregated training information and logged metrics collected during training.

        Notes:
            - This method assumes that training environments (`self.train_envs`)
                and the trajectory buffer (`self.memory`) have already been initialized.
            - After collecting `horizon_size` steps per environment, the buffer becomes full and the agent computes
                bootstrapped terminal values, finalizes trajectory segments via `finish_path`, and performs
                `n_epochs` optimization passes over mini-batches using `train_epochs`.
            - Episode termination and reset logic are handled per environment,
                and episode-level statistics are reported via callbacks.
        """
        train_info = {}
        obs = self.train_envs.buf_obs
        for _ in tqdm(range(train_steps)):
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            policy_out = self.action(obs, return_dists=False, return_logpi=False)
            acts, vals = policy_out['actions'], policy_out['values']
            next_obs, rewards, terminals, truncations, infos = self.train_envs.step(acts)
            aux_info = self.get_aux_info()

            self.callback.on_train_step(self.current_step, envs=self.train_envs, policy=self.policy,
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
                        self.train_envs.buf_obs[i] = obs[i]
                        self.current_episode[i] += 1
                        if self.use_wandb:
                            episode_info = {
                                f"Episode-Steps/env-{i}": infos[i]["episode_step"],
                                f"Train-Episode-Rewards/env-{i}": infos[i]["episode_score"]
                            }
                        else:
                            episode_info = {
                                f"Episode-Steps": {f"env-{i}": infos[i]["episode_step"]},
                                f"Train-Episode-Rewards": {f"env-{i}": infos[i]["episode_score"]}
                            }
                        self.log_infos(episode_info, self.current_step)
                        train_info.update(episode_info)
                        self.callback.on_train_episode_info(envs=self.train_envs, policy=self.policy, env_id=i,
                                                            infos=infos, use_wandb=self.use_wandb,
                                                            current_step=self.current_step,
                                                            current_episode=self.current_episode,
                                                            train_steps=train_steps)

            self.current_step += self.n_envs
            self.callback.on_train_step_end(self.current_step, envs=self.train_envs, policy=self.policy,
                                            train_steps=train_steps, train_info=train_info)
        return train_info

    def test(self,
             test_episodes: int,
             test_envs: Optional[DummyVecEnv | SubprocVecEnv] = None,
             close_envs: bool = True) -> list:
        """Evaluate the current policy in a vectorized environment.

        This method runs evaluation episodes using `test_envs` and returns the per-episode scores. Actions are produced
        by the current policy (by default sampled from the policy distribution for on-policy methods), and optional
        RGB-array frames can be recorded for video logging when rendering is enabled.

        Args:
            test_episodes (int): Total number of evaluation episodes to run across all vectorized environments.
            test_envs (Optional[DummyVecEnv | SubprocVecEnv]): Vectorized environments used for evaluation.
                Must not be None.
            close_envs (bool): Whether to close `test_envs` before returning.
                Set this to False if `test_envs` is managed externally and will be reused after evaluation.

        Returns:
            list: A list of episode scores collected during evaluation.

        Notes:
            - This method resets the evaluation environments at the beginning of testing and steps them
                until `test_episodes` episodes are completed.
            - When `render_mode == "rgb_array"` and `self.render` is True, the method records frames and logs
                the best-scoring episode as a video.
            - By default, this implementation updates `obs_rms` during testing. If you want to avoid contaminating
                training statistics, consider guarding this update with a dedicated flag (e.g., `update_rms=False`).
        """
        if test_envs is None:
            raise ValueError("`test_envs` must be provided for evaluation.")
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

            current_step += num_envs

        if self.config.render_mode == "rgb_array" and self.render:
            # time, height, width, channel -> time, channel, height, width
            videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
            self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)

        test_info = {
            "Test-Episode-Rewards/Mean-Score": np.mean(scores),
            "Test-Episode-Rewards/Std-Score": np.std(scores)
        }
        self.log_infos(test_info, self.current_step)

        self.callback.on_test_end(envs=test_envs, policy=self.policy,
                                  current_train_step=self.current_step,
                                  current_step=current_step, current_episode=current_episode,
                                  scores=scores, best_score=best_score)

        if close_envs:
            test_envs.close()

        return scores
