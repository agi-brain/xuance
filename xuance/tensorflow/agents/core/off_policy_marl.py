from tqdm import tqdm
import numpy as np
from copy import deepcopy
from argparse import Namespace
from operator import itemgetter
from gymnasium.spaces import Space
from xuance.common import Optional, List, Union, MARL_OffPolicyBuffer, MARL_OffPolicyBuffer_RNN, MultiAgentBaseCallback
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.tensorflow import Tensor, Module, tf
from xuance.tensorflow.agents.base import MARLAgents


class OffPolicyMARLAgents(MARLAgents):
    """Base class for multi-agent off-policy reinforcement learning algorithms.

    This class implements the common logic shared by multi-agent off-policy algorithms in XuanCe.
    It extends the generic `MARLAgents` abstraction with off-policy–specific components such as replay buffers,
    exploration strategies (e.g., epsilon-greedy or action noise), and update schedules.
    It supports both feed-forward and RNN-based policies and can optionally use parameter sharing across agents.

    The agent group can be used in both training and evaluation-only scenarios.
    When initialized without environments (`envs=None`), the agent group relies on explicitly provided `state_space`,
    `observation_space`, and `action_space` to build networks, which is useful for inference or standalone evaluation.

    Args:
        config (Namespace): Configuration object containing hyperparameters, algorithm settings, and runtime options.
        envs (Optional[DummyVecMultiAgentEnv | SubprocVecMultiAgentEnv]): Vectorized multi-agent environments
            used for training. If None, the agent group will not initialize training environments and must be
            provided with `state_space` (when `use_global_state=True`), `observation_space`, and `action_space`.
        num_agents (Optional[int]): Number of agents in the environment. If None, this value will be inferred from
            `envs` when available.
        agent_keys (Optional[List[str]]): Keys/names that identify each agent in the environment.
            If None, inferred from `envs` when available.
        state_space (Optional[gymnasium.spaces.Space]): Global state space used by centralized critics or global-state
            policies when enabled. Typically obtained from `envs.state_space` (or an equivalent field).
        observation_space (Optional[gymnasium.spaces.Space]): Per-agent observation space specification used to
            construct networks when `envs` is None. Typically obtained from `envs.observation_space`.
        action_space (Optional[gymnasium.spaces.Space]): Per-agent action space specification used to
            construct networks when `envs` is None. Typically obtained from `envs.action_space`.
        callback (Optional[MultiAgentBaseCallback]): Optional callback object for injecting custom logic during
            training or evaluation, such as logging, early stopping, checkpointing, or visualization.

    Notes:
        - Off-policy multi-agent agents maintain a replay buffer to reuse past experience; for RNN-based policies,
            an episode-aware buffer is used.
        - Training and evaluation environments are conceptually separated; evaluation environments may be created
            and managed externally.
        - In evaluation mode, exploration noise is disabled by default by setting `test_mode=True` when calling
            `action()` or `run_episodes()`.
    """

    def __init__(
            self,
            config: Namespace,
            envs: Optional[DummyVecMultiAgentEnv | SubprocVecMultiAgentEnv] = None,
            num_agents: Optional[int] = None,
            agent_keys: Optional[List[str]] = None,
            state_space: Optional[Space] = None,
            observation_space: Optional[Space] = None,
            action_space: Optional[Space] = None,
            callback: Optional[MultiAgentBaseCallback] = None
    ):
        super(OffPolicyMARLAgents, self).__init__(
            config, envs, num_agents, agent_keys, state_space, observation_space, action_space, callback
        )
        self.on_policy = False
        self.start_greedy = getattr(config, "start_greedy", None)
        self.end_greedy = getattr(config, "end_greedy", None)
        self.delta_egreedy: Optional[float] = None
        self.e_greedy: Optional[float] = None

        self.start_noise = getattr(config, "start_noise", None)
        self.end_noise = getattr(config, "end_noise", None)
        self.delta_noise: Optional[float] = None
        self.noise_scale: Optional[float] = None
        self.actions_low = getattr(self.action_space, "low", None)
        self.actions_high = getattr(self.action_space, "high", None)

        self.auxiliary_info_shape = None
        self.memory: Optional[MARL_OffPolicyBuffer, MARL_OffPolicyBuffer_RNN] = None

        self.buffer_size = self.config.buffer_size
        self.batch_size = self.config.batch_size

    def _build_memory(self) -> MARL_OffPolicyBuffer:
        """Build replay buffer for models training
        """
        if self.use_actions_mask:
            avail_actions_shape = {key: (self.action_space[key].n,) for key in self.agent_keys}
        else:
            avail_actions_shape = None
        input_buffer = dict(agent_keys=self.agent_keys,
                            state_space=self.state_space if self.use_global_state else None,
                            obs_space=self.observation_space,
                            act_space=self.action_space,
                            n_envs=self.n_envs,
                            buffer_size=self.buffer_size,
                            batch_size=self.batch_size,
                            avail_actions_shape=avail_actions_shape,
                            use_actions_mask=self.use_actions_mask,
                            max_episode_steps=self.episode_length)
        Buffer = MARL_OffPolicyBuffer_RNN if self.use_rnn else MARL_OffPolicyBuffer
        return Buffer(**input_buffer)

    def _build_policy(self) -> Module:
        raise NotImplementedError

    def store_experience(self, obs_dict, avail_actions, actions_dict, obs_next_dict, avail_actions_next,
                         rewards_dict, terminals_dict, info, **kwargs) -> None:
        """Store a batch of multi-agent transitions into the replay buffer.

        This method converts per-environment dictionaries (one dict per vector environment) into per-agent batched
        arrays and writes them into the replay buffer. It also stores auxiliary fields such as agent masks and,
        when enabled, global state and action masks.

        Args:
            obs_dict (List[dict]): Observations for each parallel environment.
                Each element is a dict keyed by `self.agent_keys`.
            avail_actions (Optional[List[dict]]): Available-action masks for each parallel environment
                when `use_actions_mask=True`. Each element is a dict keyed by `self.agent_keys`.
                Can be None when action masking is disabled.
            actions_dict (List[dict]): Actions executed by each agent for each parallel environment.
                Each element is a dict keyed by `self.agent_keys`.
            obs_next_dict (List[dict]): Next observations for each parallel environment.
                Each element is a dict keyed by `self.agent_keys`.
            avail_actions_next (Optional[List[dict]]): Next-step available-action masks when `use_actions_mask=True`.
                Can be None when action masking is disabled.
            rewards_dict (List[dict]): Rewards for each agent for each parallel environment.
                Each element is a dict keyed by `self.agent_keys`.
            terminals_dict (List[dict]): Termination flags for each agent for each parallel environment.
                Each element is a dict keyed by `self.agent_keys`.
            info (List[dict]): Environment info for each parallel environment at the current step.
                Must contain `agent_mask` for each agent key.
            **kwargs: Optional extra fields. When `use_global_state=True`, this method expects `state` and `next_state`
                to be provided.
        """
        experience_data = {
            'obs': {k: np.array([data[k] for data in obs_dict]) for k in self.agent_keys},
            'actions': {k: np.array([data[k] for data in actions_dict]) for k in self.agent_keys},
            'obs_next': {k: np.array([data[k] for data in obs_next_dict]) for k in self.agent_keys},
            'rewards': {k: np.array([data[k] for data in rewards_dict]) for k in self.agent_keys},
            'terminals': {k: np.array([data[k] for data in terminals_dict]) for k in self.agent_keys},
            'agent_mask': {k: np.array([data['agent_mask'][k] for data in info]) for k in self.agent_keys},
        }
        if self.use_rnn:
            experience_data['episode_steps'] = np.array([data['episode_step'] - 1 for data in info])
        if self.use_global_state:
            experience_data['state'] = np.array(kwargs['state'])
            experience_data['state_next'] = np.array(kwargs['next_state'])
        if self.use_actions_mask:
            experience_data['avail_actions'] = {k: np.array([data[k] for data in avail_actions])
                                                for k in self.agent_keys}
            experience_data['avail_actions_next'] = {k: np.array([data[k] for data in avail_actions_next])
                                                     for k in self.agent_keys}
        self.memory.store(**experience_data)

    def init_rnn_hidden(self, n_envs) -> Optional[dict]:
        """Initialize RNN hidden states for vectorized multi-agent execution.

        This method creates initial hidden states for the RNN-based policy representations
            when `self.use_rnn` is enabled. The batch size depends on whether parameter sharing is used:
            - If `use_parameter_sharing=True`, the batch dimension is `n_envs * n_agents`
                (one hidden state per agent per environment).
            - Otherwise, the batch dimension is `n_envs` (one hidden state per environment per model key).

        Args:
            n_envs (int): Number of parallel environments.

        Returns:
            Optional[dict]: A dictionary of initialized hidden states keyed by `self.model_keys`
                when `self.use_rnn` is True; otherwise None.
        """
        rnn_hidden_states = None
        if self.use_rnn:
            batch = n_envs * self.n_agents if self.use_parameter_sharing else n_envs
            rnn_hidden_states = {k: self.policy.representation[k].init_hidden(batch) for k in self.model_keys}
        return rnn_hidden_states

    def init_hidden_item(self, i_env: int,
                         rnn_hidden: Optional[dict] = None) -> dict:
        """Reset RNN hidden states for a specific environment index.

        This method re-initializes the RNN hidden states corresponding to the `i_env`-th vectorized environment.
        When parameter sharing is enabled, the hidden state batch is arranged as `(n_envs * n_agents, ...)`, so
        this method resets the contiguous slice for all agents in that environment.
        Otherwise, it resets the single hidden-state entry for `i_env` for each model key.

        Args:
            i_env (int): Index of the vectorized environment to reset.
            rnn_hidden (Optional[dict]): Current RNN hidden states keyed by `self.model_keys`.
                This object is updated in-place.

        Returns:
            dict: Updated RNN hidden states with the `i_env` entries reset.
        """
        assert self.use_rnn is True, "This method cannot be called when self.use_rnn is False."
        if self.use_parameter_sharing:
            batch_index = list(range(i_env * self.n_agents, (i_env + 1) * self.n_agents))
        else:
            batch_index = [i_env, ]
        for key in self.model_keys:
            rnn_hidden[key] = self.policy.representation[key].init_hidden_item(batch_index, *rnn_hidden[key])
        return rnn_hidden

    def _update_explore_factor(self):
        if self.e_greedy is not None:
            if self.e_greedy > self.end_greedy:
                self.e_greedy = self.start_greedy - self.delta_egreedy * self.current_step
            else:
                self.e_greedy = self.end_greedy
        elif self.noise_scale is not None:
            if self.noise_scale >= self.end_noise:
                self.noise_scale = self.start_noise - self.delta_noise * self.current_step
            else:
                self.noise_scale = self.end_noise
        else:
            return

    def exploration(self, batch_size: int,
                    pi_actions_dict: Union[List[dict], dict],
                    avail_actions_dict: Optional[List[dict]] = None):
        """Apply exploration strategy to policy actions.

        This method modifies the actions produced by the policy according to the configured exploration mechanism.
        Supported strategies include:
            - Epsilon-greedy exploration for discrete action spaces.
            - Additive Gaussian noise for continuous action spaces.

        The specific strategy is selected automatically based on the agent configuration (`e_greedy` or `noise_scale`).

        Args:
            batch_size (int): Number of parallel environments (batch size).
            pi_actions_dict (Union[List[dict], dict]): Actions produced by the policy before exploration.
                When parameter sharing is enabled, this may be a shared structure across agents.
            avail_actions_dict (Optional[List[dict]]): Available-action masks for each parallel environment
                when `use_actions_mask=True`. Can be None when action masking is disabled.

        Returns:
            Union[List[dict], dict]: Actions after applying the exploration strategy.
                The returned structure matches the format of `pi_actions_dict`.
        """
        if self.e_greedy is not None:
            if np.random.rand() < self.e_greedy:
                if self.use_actions_mask:

                    explore_actions = [{k: tf.random.categorical(Tensor(avail_actions_dict[e][k]),
                                                                 num_samples=1).numpy()
                                        for k in self.agent_keys} for e in range(batch_size)]
                else:
                    explore_actions = [{k: self.action_space[k].sample() for k in self.agent_keys} for _ in
                                       range(batch_size)]
            else:
                explore_actions = pi_actions_dict
        elif self.noise_scale is not None:
            if self.use_parameter_sharing:
                key = self.agent_keys[0]
                pi_actions_dict[key] += np.random.normal(0, self.noise_scale, size=pi_actions_dict[key].shape)
            else:
                for key in self.agent_keys:
                    pi_actions_dict[key] += np.random.normal(0, self.noise_scale, size=pi_actions_dict[key].shape)
            explore_actions = pi_actions_dict
        else:
            explore_actions = pi_actions_dict
        return explore_actions

    def action(self,
               obs_dict: List[dict],
               avail_actions_dict: Optional[List[dict]] = None,
               rnn_hidden: Optional[dict] = None,
               test_mode: Optional[bool] = False,
               **kwargs) -> dict:
        """Compute actions for all agents given vectorized observations.

        This method performs a forward pass through the current multi-agent policy to obtain actions for each agent in
        each parallel environment. When RNN-based representations are enabled, it also consumes and returns recurrent
        hidden states. During training (`test_mode=False`), this method applies the configured exploration strategy
        (epsilon-greedy or additive noise); during evaluation (`test_mode=True`), exploration is disabled.

        Args:
            obs_dict (List[dict]): Observations for each parallel environment.
                Each element is a dict keyed by `self.agent_keys`.
            avail_actions_dict (Optional[List[dict]]): Available-action masks for each parallel environment when
                `use_actions_mask=True`. Each element is a dict keyed by `self.agent_keys`. Can be None when
                action masking is disabled.
            rnn_hidden (Optional[dict]): Current RNN hidden states keyed by `self.model_keys`.
                Required when `self.use_rnn` is True.
            test_mode (bool): Whether to run in evaluation mode. When True, exploration is disabled and actions are
                produced deterministically (or without training-time noise).

        Returns:
            dict: A dictionary containing:
                - hidden_state (Optional[dict]): Updated RNN hidden states when `self.use_rnn` is True;
                    otherwise the value returned by the policy (typically None).
                - actions (List[dict]): Actions for each parallel environment.
                    Each element is a dict keyed by `self.agent_keys`.
        """
        batch_size = len(obs_dict)
        obs_input, agents_id, avail_actions_input = self._build_inputs(obs_dict, avail_actions_dict)
        hidden_state, actions, _ = self.policy(observation=obs_input,
                                               agent_ids=agents_id,
                                               avail_actions=avail_actions_input,
                                               rnn_hidden=rnn_hidden)

        if self.use_parameter_sharing:
            key = self.agent_keys[0]
            actions_out = actions[key].numpy().reshape([batch_size, self.n_agents])
            actions_dict = [{k: actions_out[e, i] for i, k in enumerate(self.agent_keys)} for e in range(batch_size)]
        else:
            actions_out = {k: actions[k].numpy().reshape(batch_size) for k in self.agent_keys}
            actions_dict = [{k: actions_out[k][i] for k in self.agent_keys} for i in range(batch_size)]

        if not test_mode:  # get random actions
            actions_dict = self.exploration(batch_size, actions_dict, avail_actions_dict)
        return {"hidden_state": hidden_state, "actions": actions_dict}

    def train(self, train_steps: int) -> dict:
        """Run the main multi-agent off-policy training loop.

        This method interacts with the training environments to collect multi-agent transitions, stores them in the
        replay buffer, and performs periodic policy updates by sampling mini-batches from the buffer.
        The training loop is step-based and advances in vectorized increments
        (one iteration corresponds to `self.n_envs` environment steps).

        Args:
            train_steps (int): Number of training iterations to run. Each iteration steps all parallel environments
            once, so the total number of environment steps is approximately `train_steps * self.n_envs`.

        Returns:
            dict: A dictionary containing aggregated training information and logged metrics collected during
                training (e.g., losses, episode statistics, exploration factors).

        Notes:
            - This method assumes that training environments (`self.train_envs`) and the replay buffer (`self.memory`)
                have already been initialized.
            - When `self.use_rnn` is enabled, rollout collection and buffer bookkeeping are handled in `run_episodes()`,
                and updates are performed once enough experience is available.
            - Policy updates are triggered after `self.start_training` steps and then periodically according to
                `self.training_frequency`.
        """
        train_info = {}
        if self.use_rnn:
            with tqdm(total=train_steps) as process_bar:
                step_start, step_last = deepcopy(self.current_step), deepcopy(self.current_step)
                n_steps_all = train_steps * self.n_envs
                while step_last - step_start < n_steps_all:
                    self.run_episodes(None, n_episodes=self.n_envs, test_mode=False)
                    if self.current_step >= self.start_training:
                        update_info = self.train_epochs(n_epochs=self.n_epochs)
                        self.log_infos(update_info, self.current_step)
                        train_info.update(update_info)
                        self.callback.on_train_epochs_end(self.current_step, policy=self.policy, memory=self.memory,
                                                          current_episode=self.current_episode, train_steps=train_steps,
                                                          update_info=update_info)
                    process_bar.update((self.current_step - step_last) // self.n_envs)
                    step_last = deepcopy(self.current_step)
                process_bar.update(train_steps - process_bar.last_print_n)
                self.callback.on_train_step_end(self.current_step, envs=self.train_envs, policy=self.policy,
                                                train_steps=train_steps, train_info=train_info)
            return train_info

        obs_dict = self.train_envs.buf_obs
        avail_actions = self.train_envs.buf_avail_actions if self.use_actions_mask else None
        state = self.train_envs.buf_state.copy() if self.use_global_state else None
        for _ in tqdm(range(train_steps)):
            policy_out = self.action(obs_dict=obs_dict, avail_actions_dict=avail_actions, test_mode=False)
            actions_dict = policy_out['actions']
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = self.train_envs.step(actions_dict)
            next_state = self.train_envs.buf_state.copy() if self.use_global_state else None
            next_avail_actions = self.train_envs.buf_avail_actions if self.use_actions_mask else None

            self.callback.on_train_step(self.current_step, envs=self.train_envs, policy=self.policy,
                                        obs=obs_dict, policy_out=policy_out, acts=actions_dict, next_obs=next_obs_dict,
                                        rewards=rewards_dict, state=state, next_state=next_state,
                                        avail_actions=avail_actions, next_avail_actions=next_avail_actions,
                                        terminals=terminated_dict, truncations=truncated, infos=info,
                                        train_steps=train_steps)

            self.store_experience(obs_dict, avail_actions, actions_dict, next_obs_dict, next_avail_actions,
                                  rewards_dict, terminated_dict, info,
                                  **{'state': state, 'next_state': next_state})
            if self.current_step >= self.start_training and self.current_step % self.training_frequency == 0:
                update_info = self.train_epochs(n_epochs=self.n_epochs)
                self.log_infos(update_info, self.current_step)
                train_info.update(update_info)
                self.callback.on_train_epochs_end(self.current_step, policy=self.policy, memory=self.memory,
                                                  current_episode=self.current_episode, train_steps=train_steps,
                                                  update_info=update_info)

            obs_dict = deepcopy(next_obs_dict)
            if self.use_global_state:
                state = deepcopy(next_state)
            if self.use_actions_mask:
                avail_actions = deepcopy(next_avail_actions)

            for i in range(self.n_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    obs_dict[i] = info[i]["reset_obs"]
                    self.train_envs.buf_obs[i] = info[i]["reset_obs"]
                    if self.use_global_state:
                        state = info[i]["reset_state"]
                        self.train_envs.buf_state[i] = info[i]["reset_state"]
                    if self.use_actions_mask:
                        avail_actions[i] = info[i]["reset_avail_actions"]
                        self.train_envs.buf_avail_actions[i] = info[i]["reset_avail_actions"]
                    self.current_episode[i] += 1
                    if self.use_wandb:
                        episode_info = {
                            f"Train-Results/Episode-Steps/env-%d" % i: info[i]["episode_step"],
                            f"Train-Results/Episode-Rewards/env-%d" % i: info[i]["episode_score"]
                        }
                    else:
                        episode_info = {
                            f"Train-Results/Episode-Steps": {"env-%d" % i: info[i]["episode_step"]},
                            f"Train-Results/Episode-Rewards": {
                                "env-%d" % i: np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"]))}
                        }
                    self.log_infos(episode_info, self.current_step)
                    train_info.update(episode_info)
                    self.callback.on_train_episode_info(envs=self.train_envs, policy=self.policy, env_id=i,
                                                        infos=info, use_wandb=self.use_wandb,
                                                        current_step=self.current_step,
                                                        current_episode=self.current_episode,
                                                        train_steps=train_steps)

            self.current_step += self.n_envs
            self._update_explore_factor()
            self.callback.on_train_step_end(self.current_step, envs=self.train_envs, policy=self.policy,
                                            train_steps=train_steps, train_info=train_info)
        return train_info

    def run_episodes(self,
                     n_episodes: int = 1,
                     run_envs: Optional[DummyVecMultiAgentEnv | SubprocVecMultiAgentEnv] = None,
                     test_mode: bool = False,
                     close_envs: bool = True) -> list:
        """Run vectorized multi-agent episodes for rollout collection or evaluation.

        This method steps a vectorized multi-agent environment using the current policy until `n_episodes` episodes
        have completed. When `test_mode` is False, collected transitions are stored into the replay buffer (and episode
        boundaries are tracked for RNN-aware buffers). When `test_mode` is True, exploration is disabled and
        episode scores are returned; optional RGB-array frames can be recorded and logged as a video.

        Args:
            n_episodes (int): Number of completed episodes to run across all parallel environments.
            run_envs (Optional[DummyVecMultiAgentEnv | SubprocVecMultiAgentEnv]): Vectorized environments to run.
                If None, `self.train_envs` is used.
            test_mode (bool): Whether to run in evaluation mode. When True, exploration is disabled and the
                replay buffer is not written.
            close_envs (bool): Whether to close `run_envs` before returning when `test_mode` is True.
                Set this to False if the caller manages the environment lifecycle externally.

        Returns:
            list: Episode scores (mean reward across agents) for each completed episode.
        """
        envs = self.train_envs if run_envs is None else run_envs
        num_envs = envs.num_envs
        videos, episode_videos, images = [[] for _ in range(num_envs)], [], None
        _current_episode, _current_step, scores, best_score = 0, 0, [], -np.inf
        obs_dict, info = envs.reset()
        state = envs.buf_state.copy() if self.use_global_state else None
        avail_actions = envs.buf_avail_actions if self.use_actions_mask else None
        if test_mode:
            if self.config.render_mode == "rgb_array" and self.render:
                images = envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)
        else:
            if self.use_rnn:
                self.memory.clear_episodes()
        rnn_hidden = self.init_rnn_hidden(num_envs)

        while _current_episode < n_episodes:
            policy_out = self.action(obs_dict=obs_dict,
                                     avail_actions_dict=avail_actions,
                                     rnn_hidden=rnn_hidden,
                                     test_mode=test_mode)
            rnn_hidden, actions_dict = policy_out['hidden_state'], policy_out['actions']
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = envs.step(actions_dict)
            next_state = envs.buf_state.copy() if self.use_global_state else None
            next_avail_actions = envs.buf_avail_actions if self.use_actions_mask else None
            if test_mode:
                if self.config.render_mode == "rgb_array" and self.render:
                    images = envs.render(self.config.render_mode)
                    for idx, img in enumerate(images):
                        videos[idx].append(img)
            else:
                self.store_experience(obs_dict, avail_actions, actions_dict, next_obs_dict, next_avail_actions,
                                      rewards_dict, terminated_dict, info,
                                      **{'state': state, 'next_state': next_state})

            self.callback.on_test_step(envs=envs, policy=self.policy, images=images, test_mode=test_mode,
                                       obs=obs_dict, policy_out=policy_out, acts=actions_dict,
                                       next_obs=next_obs_dict, rewards=rewards_dict,
                                       terminals=terminated_dict, truncations=truncated, infos=info,
                                       state=state, next_state=next_state,
                                       current_train_step=self.current_step, n_episodes=n_episodes,
                                       current_step=_current_step, current_episode=_current_episode)

            obs_dict = deepcopy(next_obs_dict)
            if self.use_global_state:
                state = deepcopy(next_state)
            if self.use_actions_mask:
                avail_actions = deepcopy(next_avail_actions)

            for i in range(num_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    _current_episode += 1
                    obs_dict[i] = info[i]["reset_obs"]
                    envs.buf_obs[i] = info[i]["reset_obs"]
                    if self.use_global_state:
                        state = info[i]["reset_state"]
                        envs.buf_state[i] = info[i]["reset_state"]
                    if self.use_actions_mask:
                        avail_actions[i] = info[i]["reset_avail_actions"]
                        envs.buf_avail_actions[i] = info[i]["reset_avail_actions"]
                    if self.use_rnn:
                        rnn_hidden = self.init_hidden_item(i_env=i, rnn_hidden=rnn_hidden)
                        if not test_mode:
                            terminal_data = {'obs': next_obs_dict[i],
                                             'episode_step': info[i]['episode_step']}
                            if self.use_global_state:
                                terminal_data['state'] = next_state[i]
                            if self.use_actions_mask:
                                terminal_data['avail_actions'] = next_avail_actions[i]
                            self.memory.finish_path(i, **terminal_data)
                    episode_score = float(np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"])))
                    scores.append(episode_score)
                    if test_mode:
                        if best_score < episode_score:
                            best_score = episode_score
                            episode_videos = videos[i].copy()
                        if self.config.test_mode:
                            print("Episode: %d, Score: %.2f" % (_current_episode, episode_score))
                    else:
                        self.current_episode[i] += 1
                        if self.use_wandb:
                            episode_info = {
                                "Train-Results/Episode-Steps/env-%d" % i: info[i]["episode_step"],
                                "Train-Results/Episode-Rewards/env-%d" % i: info[i]["episode_score"]
                            }
                        else:
                            episode_info = {
                                "Train-Results/Episode-Steps": {"env-%d" % i: info[i]["episode_step"]},
                                "Train-Results/Episode-Rewards": {
                                    "env-%d" % i: np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"]))}
                            }
                        self.current_step += info[i]["episode_step"]
                        self.log_infos(episode_info, self.current_step)
                        self._update_explore_factor()
                        self.callback.on_train_episode_info(envs=envs, policy=self.policy, env_id=i,
                                                            infos=info, use_wandb=self.use_wandb,
                                                            current_step=self.current_step,
                                                            current_episode=self.current_episode,
                                                            n_episodes=n_episodes)
            _current_step += num_envs

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

            self.callback.on_test_end(envs=envs, policy=self.policy,
                                      current_train_step=self.current_step,
                                      current_step=_current_step, current_episode=_current_episode,
                                      scores=scores, best_score=best_score)

            if close_envs:
                envs.close()
        return scores

    def train_epochs(self, n_epochs: int = 1) -> dict:
        """Update policies for multiple epochs using mini-batches sampled from the replay buffer.

        This method performs `n_epochs` optimization passes. At each epoch, it samples a mini-batch from the
        replay buffer and calls the learner's update function. When RNN-based policies are enabled, the RNN-specific
        update method is used.

        Args:
            n_epochs (int): Number of optimization epochs to perform.

        Returns:
            dict: A dictionary of training metrics returned by the learner from the last update call
                (e.g., Q loss, policy loss, entropy), augmented with the current exploration factors
                (`epsilon-greedy` and `noise_scale`).
        """
        info_train = {}
        for i_epoch in range(n_epochs):
            sample = self.memory.sample()
            info_train = self.learner.update_rnn(sample) if self.use_rnn else self.learner.update(sample)
        info_train["epsilon-greedy"] = self.e_greedy
        info_train["noise_scale"] = self.noise_scale
        return info_train

    def test(self,
             test_episodes: int,
             test_envs: Optional[DummyVecMultiAgentEnv | SubprocVecMultiAgentEnv] = None,
             close_envs: bool = True) -> list:
        """Evaluate the current multi-agent policy for a number of episodes.

        This method runs evaluation episodes in `test_envs` by delegating to `run_episodes(test_mode=True)` and returns
        the per-episode scores. During evaluation, exploration is disabled and optional RGB-array frames can be recorded
        and logged as a video when rendering is enabled.

        Args:
            test_episodes (int): Number of completed episodes to evaluate across all parallel environments.
            test_envs (Optional[DummyVecMultiAgentEnv | SubprocVecMultiAgentEnv]): Vectorized multi-agent environments
                used for evaluation. If None, `self.train_envs` is used.
            close_envs (bool): Whether to close `test_envs` before returning. Set this to False if `test_envs` is
                managed externally and will be reused after evaluation.

        Returns:
            list: Episode scores (mean reward across agents) for each completed evaluation episode.
        """
        scores = self.run_episodes(
            n_episodes=test_episodes,
            run_envs=test_envs,
            test_mode=True,
            close_envs=close_envs
        )
        return scores
