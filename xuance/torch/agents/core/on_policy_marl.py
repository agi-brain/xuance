import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from operator import itemgetter
from gymnasium.spaces import Space
from typing import Tuple, List, Optional, Dict
from xuance.common import MARL_OnPolicyBuffer, MARL_OnPolicyBuffer_RNN, MultiAgentBaseCallback
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.torch import Module
from xuance.torch.agents.base import MARLAgents
from xuance.torch.rl_models.modules import RNN_State


class OnPolicyMARLAgents(MARLAgents):
    """Base class for multi-agent on-policy reinforcement learning algorithms.

    This class implements the common logic shared by multi-agent on-policy algorithms (e.g., MAPPO, MATrPO, and other
    actor-critic variants) in XuanCe. It extends the generic `MARLAgents` abstraction with on-policy–specific
    components such as trajectory buffers, rollout collection, advantage/return estimation (GAE),
    and multi-epoch policy/value updates.

    The agent group can be used in both training and evaluation-only scenarios. When initialized without environments
    (`envs=None`), the agent group relies on explicitly provided `state_space`, `observation_space`, and `action_space`
    to build networks, which is useful for inference or standalone evaluation.

    Args:
        config (Namespace): Configuration object containing hyperparameters, algorithm settings, and runtime options.
        envs (Optional[DummyVecMultiAgentEnv | SubprocVecMultiAgentEnv]): Vectorized multi-agent environments used for
            training. If None, the agent group will not initialize training environments and must be provided with
            `state_space` (when `use_global_state=True`), `observation_space`, and `action_space`.
        num_agents (Optional[int]): Number of agents in the environment. If None, this value will be inferred from
            `envs` when available.
        agent_keys (Optional[List[str]]): Keys/names that identify each agent in the environment.
            If None, inferred from `envs` when available.
        state_space (Optional[gymnasium.spaces.Space]): Global state space used by centralized critics or global-state
            policies when enabled. Typically obtained from `envs.state_space` (or an equivalent field).
        observation_space (Optional[gymnasium.spaces.Space]): Per-agent observation space specification used to
            construct networks when `envs` is None. Typically obtained from `envs.observation_space`.
        action_space (Optional[gymnasium.spaces.Space]): Per-agent action space specification used to construct networks
            when `envs` is None. Typically obtained from `envs.action_space`.
        callback (Optional[MultiAgentBaseCallback]): Optional callback object for injecting custom logic during training
            or evaluation, such as logging, early stopping, checkpointing, or visualization.

    Notes:
        - On-policy multi-agent agents collect fresh rollouts from the current policy and update the policy
            using trajectories stored in a buffer.
        - Training and evaluation environments are conceptually separated; evaluation environments may be created and
            managed externally.
        - In evaluation mode, exploration schedules specific to training are disabled by default
            (e.g., actions are sampled without epsilon-greedy or additive noise used by off-policy agents).
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
        super(OnPolicyMARLAgents, self).__init__(
            config, envs, num_agents, agent_keys, state_space, observation_space, action_space, callback
        )
        self.on_policy = True
        self.continuous_control: bool = False
        self.n_epochs = config.n_epochs
        self.n_minibatch = config.n_minibatch
        self.buffer_size = self.config.buffer_size
        self.batch_size = self.buffer_size // self.n_minibatch
        self.memory: Optional[MARL_OnPolicyBuffer | MARL_OnPolicyBuffer_RNN] = None

    def _build_memory(self) -> MARL_OnPolicyBuffer:
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
                            buffer_size=self.config.buffer_size,
                            use_gae=self.config.use_gae,
                            use_advnorm=self.config.use_advnorm,
                            gamma=self.config.gamma,
                            gae_lam=self.config.gae_lambda,
                            avail_actions_shape=avail_actions_shape,
                            use_actions_mask=self.use_actions_mask,
                            max_episode_steps=self.episode_length)
        Buffer = MARL_OnPolicyBuffer_RNN if self.use_rnn else MARL_OnPolicyBuffer
        return Buffer(**input_buffer)

    def _build_model(self) -> Module:
        raise NotImplementedError

    def store_experience(
            self,
            obs_list: List[dict],
            avail_actions: Optional[List[dict]],
            actions_list: List[dict],
            log_pi_a: dict,
            rewards_list: List[dict],
            values_dict: dict,
            terminals_list: List[dict],
            info: List[dict],
            **kwargs
    ) -> None:
        """Store a batch of multi-agent transitions into the on-policy buffer.

        This method converts per-environment dictionaries (one dict per vector environment) into per-agent batched
        arrays and writes them into the on-policy trajectory buffer. It also stores auxiliary fields such as agent masks
        and (optionally) global state and action masks. For RNN-based policies, episode-step indices are recorded to
        support episode-aware bookkeeping.

        Args:
            obs_list (List[dict]): Observations for each parallel environment.
                Each element is a dict keyed by `self.agent_keys`.
            avail_actions (Optional[List[dict]]): Available-action masks for each parallel environment when
                `use_actions_mask=True`. Each element is a dict keyed by `self.agent_keys`.
                Can be None when action masking is disabled.
            actions_list (List[dict]): Actions executed by each agent for each parallel environment.
                Each element is a dict keyed by `self.agent_keys`.
            log_pi_a (dict): Log-probabilities of the actions under the current policy
                (typically computed during rollout collection).
            rewards_list (List[dict]): Rewards for each agent for each parallel environment.
                Each element is a dict keyed by `self.agent_keys`.
            values_dict (dict): Value estimates produced by the critic for each agent
                (used for advantage/return computation).
            terminals_list (List[dict]): Termination flags for each agent for each parallel environment.
                Each element is a dict keyed by `self.agent_keys`.
            info (List[dict]): Environment info for each parallel environment at the current step.
                Must contain `agent_mask` for each agent key.
            **kwargs: Optional extra fields. When `use_global_state=True`, this method expects `state` to be provided.
        """
        experience_data = {
            'obs': {k: np.array([data[k] for data in obs_list]) for k in self.agent_keys},
            'actions': {k: np.array([data[k] for data in actions_list]) for k in self.agent_keys},
            'log_pi_old': log_pi_a,
            'rewards': {k: np.array([data[k] for data in rewards_list]) for k in self.agent_keys},
            'values': values_dict,
            'terminals': {k: np.array([data[k] for data in terminals_list]) for k in self.agent_keys},
            'agent_mask': {k: np.array([data['agent_mask'][k] for data in info]) for k in self.agent_keys},
        }
        if self.use_rnn:
            experience_data['episode_steps'] = np.array([data['episode_step'] - 1 for data in info])
        if self.use_global_state:
            experience_data['state'] = np.array(kwargs['state'])
        if self.use_actions_mask:
            experience_data['avail_actions'] = {k: np.array([data[k] for data in avail_actions])
                                                for k in self.agent_keys}
        self.memory.store(**experience_data)

    def init_rnn_states(self, n_envs) -> Tuple[Optional[dict], Optional[dict]]:
        """Initialize RNN hidden states for vectorized multi-agent execution.

        This method creates initial hidden states for the RNN-based actor and critic representations when `self.use_rnn`
        is enabled. The batch size depends on whether parameter sharing is used:
            - If `use_parameter_sharing=True`, the batch dimension is `n_envs * n_agents`
                (one hidden state per agent per environment).
            - Otherwise, the batch dimension is `n_envs` (one hidden state per environment per model key).

        Args:
            n_envs (int): Number of parallel environments.

        Returns:
            Tuple[Optional[dict], Optional[dict]]: A tuple of `(rnn_states_actor, rnn_states_critic)`.
                Each element is a dict keyed by `self.model_keys` when `self.use_rnn` is True; otherwise both are None.
        """
        return self.model.init_actor_rnn_states(n_envs), self.model.init_critic_rnn_states(n_envs)

    def init_rnn_states_item(
            self,
            i_env: int,
            rnn_states_actor: Optional[Dict[str, RNN_State]] = None,
            rnn_states_critic: Optional[Dict[str, RNN_State]] = None
    ) -> Tuple[Dict[str, RNN_State], Dict[str, RNN_State] | None]:
        """Reset RNN hidden states for a specific environment index.

        This method re-initializes the RNN hidden states corresponding to the `i_env`-th vectorized environment.
        When parameter sharing is enabled, the hidden state batch is arranged as `(n_envs * n_agents, ...)`, so
        this method resets the contiguous slice for all agents in that environment.
        Otherwise, it resets the single hidden-state entry for `i_env` for each model key.

        Args:
            i_env (int): Index of the vectorized environment to reset.
            rnn_states_actor (Optional[dict]): Current actor RNN hidden states keyed by `self.model_keys`.
                This object is updated in-place.
            rnn_states_critic (Optional[dict]): Current critic RNN hidden states keyed by `self.model_keys`.
                This object is updated in-place. Can be None when critic hidden states are not tracked.

        Returns:
            Tuple[Optional[dict], Optional[dict]]: Updated `(rnn_states_actor, rnn_states_critic)` with
                the `i_env` entries reset.
        """
        if rnn_states_critic is None:
            return self.model.init_actor_rnn_states_item(i_env, rnn_states_actor), None
        return (self.model.init_actor_rnn_states_item(i_env, rnn_states_actor),
                self.model.init_critic_rnn_states_item(i_env, rnn_states_critic))

    def get_actions(
            self,
            obs_list: List[dict],
            state: Optional[np.ndarray] = None,
            avail_actions_list: Optional[List[dict]] = None,
            rnn_states_actor: Optional[dict] = None,
            rnn_states_critic: Optional[dict] = None,
            test_mode: Optional[bool] = False,
            deterministic: bool = False,
            **kwargs
    ) -> dict:
        """Compute actions (and optional value/log-prob outputs) for multi-agent execution.

        This method performs a forward pass through the current multi-agent actor-critic policy to produce actions for
        each agent in each parallel environment. When RNN-based representations are enabled, the method consumes and
        returns recurrent hidden states for both the actor and the critic. During training (`test_mode=False`),
        this method also computes critic values and action log-probabilities needed for on-policy updates.
        During evaluation (`test_mode=True`), critic values and log-probabilities are not computed to reduce overhead.

        Args:
            obs_list (List[dict]): Observations for each parallel environment.
                Each element is a dict keyed by `self.agent_keys`.
            state (Optional[np.ndarray]): Global state array used by centralized critics when `use_global_state=True`.
                The expected shape depends on the environment wrapper.
            avail_actions_list (Optional[List[dict]]): Available-action masks for each parallel environment when
                `use_actions_mask=True`. Each element is a dict keyed by `self.agent_keys`.
                Can be None when action masking is disabled.
            rnn_states_actor (Optional[dict]): Current actor RNN hidden states keyed by `self.model_keys`.
                Required when `self.use_rnn` is True.
            rnn_states_critic (Optional[dict]): Current critic RNN hidden states keyed by `self.model_keys`.
                Required when `self.use_rnn` is True and values are requested.
            test_mode (bool): Whether to run in evaluation mode. When True, only actions are produced and
                training-specific outputs (values/log_pi) are omitted.
            deterministic (bool): True for deterministic policy and False for stochastic policy.

        Returns:
            dict: A dictionary containing:
                - rnn_states_actor (Optional[dict]): Updated actor RNN hidden states when `self.use_rnn` is True;
                    otherwise the value returned by the policy (typically None).
                - rnn_states_critic (Optional[dict]): Updated critic RNN hidden states when computed;
                    otherwise an empty dict.
                - actions (List[dict]): Actions for each parallel environment. Each element is a dict keyed by
                    `self.agent_keys`.
                - log_pi (dict): Log-probabilities of sampled actions for each agent when `test_mode=False`;
                    otherwise an empty dict.
                - values (dict): Critic value estimates for each agent when `test_mode=False`; otherwise an empty dict.
        """
        batch_size = len(obs_list)
        rnn_states_critic_new, values_out, log_pi_a_dict, values_dict = {}, {}, {}, {}

        obs_input, agent_indices, avail_actions_input = self._build_inputs(obs_list, avail_actions_list)
        with torch.no_grad():
            model_output = self.model(observations=obs_input,
                                      agent_indices=agent_indices,
                                      avail_actions=avail_actions_input,
                                      rnn_states=rnn_states_actor,
                                      deterministic=deterministic)
            rnn_states_actor_new = model_output.actor_rnn_states
            actions = model_output.actions

        if not test_mode:
            for group, agent_keys in self.groups.items():
                # shape: batch_size * N_agents
                log_pi_a = model_output.distributions[group].log_prob(actions.packed(group)).reshape(batch_size, -1)
                for i, agent in enumerate(agent_keys):
                    log_pi_a_dict[agent] = log_pi_a[:, i].cpu().numpy()

            with torch.no_grad():
                values_model_output = self.model.get_values(state=state if self.use_global_state else None,
                                                            observations=obs_input,
                                                            agent_indices=agent_indices,
                                                            rnn_states=rnn_states_critic)
                rnn_states_critic_new = values_model_output.critic_rnn_states
                values = values_model_output.values
                values.grouped_tensor = {k: v.cpu().numpy() for k, v in values.grouped_tensor.items()}
                values_dict = {k: v.reshape(batch_size) for k, v in values.agent_wise.items()}

        if self.continuous_control:
            actions.grouped_tensor = {
                k: actions.grouped_tensor[k].reshape(batch_size, n, -1).cpu().numpy() for k, n in
                self.n_group_agents.items()
            }
            actions_list = [{
                k: actions.agent_wise[k][e].reshape([-1]) for k in self.agent_keys
            } for e in range(batch_size)]
        else:
            actions.grouped_tensor = {
                k: actions.grouped_tensor[k].reshape(batch_size, n).cpu().numpy() for k, n in
                self.n_group_agents.items()
            }
            actions_list = [{
                k: actions.agent_wise[k][e].reshape([]) for k in self.agent_keys
            } for e in range(batch_size)]

        return {"rnn_states_actor": rnn_states_actor_new, "rnn_states_critic": rnn_states_critic_new,
                "actions": actions_list, "log_pi": log_pi_a_dict, "values": values_dict}

    def values_next(
            self,
            i_env: int,
            obs_dict: dict,
            state: Optional[np.ndarray] = None,
            rnn_states_critic: Dict[str, RNN_State] = None
    ) -> Tuple[Dict[str, RNN_State], Dict[str, np.ndarray]]:
        """Compute bootstrapped critic values for an environment that reached a boundary.

        This method evaluates the critic on the terminal/next observations of a specific
        vectorized environment (`i_env`) and returns per-agent value estimates used for bootstrapping
        when finalizing trajectories (e.g., for GAE/return computation).

        Args:
            i_env (int): Index of the vectorized environment that is finishing an episode or trajectory segment.
            obs_dict (dict): Per-agent observations for the selected environment.
                This dict is keyed by `self.agent_keys`.
            state (Optional[np.ndarray]): Global state for the selected environment when `use_global_state=True`.
                If provided, it should correspond to the same `i_env` instance.
            rnn_states_critic (Optional[dict]): Current critic RNN hidden states keyed by `self.model_keys`.
                Required when `self.use_rnn` is True.

        Returns:
            Tuple[Optional[dict], dict]: A tuple of `(rnn_states_critic_new, values_dict)`:
                - rnn_states_critic_new (Optional[dict]): Updated critic hidden states for the selected environment
                    when `self.use_rnn` is True; otherwise the value returned by the critic (typically None).
                - values_dict (dict): Per-agent critic value estimates keyed by `self.agent_keys`.
        """
        if self.use_rnn:
            rnn_states_critic_i = {}
            for group, n_agents in self.n_group_agents.items():
                hidden_item_index = np.arange(i_env * n_agents, (i_env + 1) * n_agents)
                rnn_states_critic_i[group] = self.model.critics[
                    group].representation.obs_representation.get_rnn_states_item(
                    hidden_item_index, rnn_states_critic[group])
        else:
            rnn_states_critic_i = None

        obs_input, agent_indices, _ = self._build_inputs([obs_dict])
        with torch.no_grad():
            values_model_output = self.model.get_values(state=state if self.use_global_state else None,
                                                        observations=obs_input,
                                                        agent_indices=agent_indices,
                                                        rnn_states=rnn_states_critic_i)
            rnn_states_critic_new_i = values_model_output.critic_rnn_states
            values = values_model_output.values
            values.grouped_tensor = {k: v.cpu().numpy() for k, v in values.grouped_tensor.items()}
            values_dict = {k: v.reshape([]) for k, v in values.agent_wise.items()}

        return rnn_states_critic_new_i, values_dict

    def train(self, train_steps: int) -> dict:
        """Run the main multi-agent on-policy training loop.

        This method interacts with the training environments to collect fresh rollouts from the current policy, stores
        transitions in the on-policy trajectory buffer, and triggers policy/value updates when the buffer is full.
        Training advances in vectorized increments (one iteration corresponds to stepping all parallel environments once).

        Args:
            train_steps (int): Number of rollout collection iterations to run. Each iteration steps all parallel
                environments once, so the total number of environment steps is approximately `train_steps * self.n_envs`.

        Returns:
            dict: A dictionary containing aggregated training information and logged metrics collected during
                training (e.g., policy loss, value loss, entropy, KL divergence, and episode statistics).

        Notes:
            - This method assumes that training environments (`self.train_envs`) and the trajectory buffer `self.memory`
                have already been initialized.
            - When the buffer becomes full, the agent finalizes trajectories by computing bootstrapped terminal values
                via `values_next` and calling `finish_path`, then performs `n_epochs` optimization passes over
                mini-batches using `train_epochs`.
            - Episode termination and reset logic are handled per environment,
                and episode-level statistics are reported via callbacks.
        """
        train_info = {}
        if self.use_rnn:
            with tqdm(total=train_steps) as process_bar:
                step_start, step_last = deepcopy(self.current_step), deepcopy(self.current_step)
                n_steps_all = train_steps * self.n_envs
                while step_last - step_start < n_steps_all:
                    self.run_episodes(n_episodes=self.n_envs, test_mode=False, close_envs=False)
                    update_info = self.train_epochs(n_epochs=self.n_epochs)
                    self.log_infos(update_info, self.current_step)
                    train_info.update(update_info)

                    self.callback.on_train_epochs_end(self.current_step, policy=self.model, memory=self.memory,
                                                      current_episode=self.current_episode, train_steps=train_steps,
                                                      update_info=update_info)

                    process_bar.update((self.current_step - step_last) // self.n_envs)
                    step_last = deepcopy(self.current_step)
                process_bar.update(train_steps - process_bar.last_print_n)
                self.callback.on_train_step_end(self.current_step, envs=self.train_envs, policy=self.model,
                                                train_steps=train_steps, train_info=train_info)
            return train_info

        obs_list = self.train_envs.buf_obs
        avail_actions = self.train_envs.buf_avail_actions if self.use_actions_mask else None
        state = self.train_envs.buf_state if self.use_global_state else None
        for _ in tqdm(range(train_steps)):
            policy_out = self.get_actions(obs_list=obs_list, state=state, avail_actions_list=avail_actions,
                                          test_mode=False)
            actions_list, log_pi_a_list = policy_out['actions'], policy_out['log_pi']
            values_dict = policy_out['values']
            next_obs_list, rewards_list, terminated_list, truncated, info = self.train_envs.step(actions_list)
            next_state = self.train_envs.buf_state if self.use_global_state else None
            next_avail_actions = self.train_envs.buf_avail_actions if self.use_actions_mask else None

            self.callback.on_train_step(self.current_step, envs=self.train_envs, policy=self.model,
                                        obs=obs_list, policy_out=policy_out, acts=actions_list, next_obs=next_obs_list,
                                        rewards=rewards_list, state=state, next_state=next_state,
                                        avail_actions=avail_actions, next_avail_actions=next_avail_actions,
                                        terminals=terminated_list, truncations=truncated, infos=info,
                                        train_steps=train_steps, values_dict=values_dict)

            self.store_experience(obs_list, avail_actions, actions_list, log_pi_a_list, rewards_list, values_dict,
                                  terminated_list, info, **{'state': state})
            if self.memory.full:
                for i in range(self.n_envs):
                    if all(terminated_list[i].values()):
                        value_next = {key: 0.0 for key in self.agent_keys}
                    else:
                        next_state_i = next_state[i] if self.use_global_state else None
                        _, value_next = self.values_next(i_env=i, obs_dict=next_obs_list[i], state=next_state_i)
                    self.memory.finish_path(i_env=i, agent_grouping=self.agent_grouping, value_next=value_next,
                                            value_normalizer=self.learner.value_normalizer)
            update_info = self.train_epochs(n_epochs=self.n_epochs)
            self.log_infos(update_info, self.current_step)
            train_info.update(update_info)
            obs_list, avail_actions = deepcopy(next_obs_list), deepcopy(next_avail_actions)
            state = deepcopy(next_state) if self.use_global_state else None

            for i in range(self.n_envs):
                if all(terminated_list[i].values()) or truncated[i]:
                    if all(terminated_list[i].values()):
                        value_next = {key: 0.0 for key in self.agent_keys}
                    else:
                        state_i = state[i] if self.use_global_state else None
                        _, value_next = self.values_next(i_env=i, obs_dict=obs_list[i], state=state_i)
                    self.memory.finish_path(i_env=i, agent_grouping=self.agent_grouping, value_next=value_next,
                                            value_normalizer=self.learner.value_normalizer)
                    obs_list[i] = info[i]["reset_obs"]
                    self.train_envs.buf_obs[i] = info[i]["reset_obs"]
                    if self.use_actions_mask:
                        avail_actions[i] = info[i]["reset_avail_actions"]
                        self.train_envs.buf_avail_actions[i] = info[i]["reset_avail_actions"]
                    if self.use_global_state:
                        state[i] = info[i]["reset_state"]
                        self.train_envs.buf_state[i] = info[i]["reset_state"]
                    self.current_episode[i] += 1
                    if self.use_wandb:
                        episode_info = {
                            f"Train-Results/Episode-Steps/rank_{self.rank}/env-%d" % i: info[i]["episode_step"],
                            f"Train-Results/Episode-Rewards/rank_{self.rank}/env-%d" % i: info[i]["episode_score"]
                        }
                    else:
                        episode_info = {
                            f"Train-Results/Episode-Steps/rank_{self.rank}": {"env-%d" % i: info[i]["episode_step"]},
                            f"Train-Results/Episode-Rewards/rank_{self.rank}": {
                                "env-%d" % i: np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"]))}
                        }
                    self.log_infos(episode_info, self.current_step)
                    train_info.update(episode_info)
                    self.callback.on_train_episode_info(envs=self.train_envs, policy=self.model, env_id=i,
                                                        infos=info, rank=self.rank, use_wandb=self.use_wandb,
                                                        current_step=self.current_step,
                                                        current_episode=self.current_episode,
                                                        train_steps=train_steps)

            self.current_step += self.n_envs
            self.callback.on_train_step_end(self.current_step, envs=self.train_envs, policy=self.model,
                                            train_steps=train_steps, train_info=train_info)
        return train_info

    def run_episodes(self,
                     n_episodes: int = 1,
                     deterministic_policy: bool = False,
                     run_envs: Optional[DummyVecMultiAgentEnv | SubprocVecMultiAgentEnv] = None,
                     test_mode: bool = False,
                     close_envs: bool = True) -> list:
        """Run vectorized multi-agent episodes for rollout collection or evaluation.

        This method steps a vectorized multi-agent environment using the current actor-critic policy until `n_episodes`
        episodes have completed. When `test_mode` is False, collected transitions are stored into the on-policy
        trajectory buffer and episode boundaries are tracked for bootstrapping and advantage computation (GAE).
        When `test_mode` is True, training-time outputs (values/log-probabilities) are skipped, exploration schedules
        are disabled by default, and episode scores are returned; optional RGB-array frames can be recorded and logged as a video.

        Args:
            n_episodes (int): Number of completed episodes to run across all parallel environments.
            deterministic_policy (bool): True for evaluating the deterministic policy,
                and False for evaluating the stochastic policy.
            run_envs (Optional[DummyVecMultiAgentEnv | SubprocVecMultiAgentEnv]): Vectorized environments to run.
                If None, `self.train_envs` is used.
            test_mode (bool): Whether to run in evaluation mode. When True, the trajectory buffer is not written and
                only episode scores are collected.
            close_envs (bool): Whether to close `run_envs` before returning when `test_mode` is True.
                Set this to False if the caller manages the environment lifecycle externally.

        Returns:
            list: Episode scores (mean reward across agents) for each completed episode.
        """
        envs = self.train_envs if run_envs is None else run_envs
        num_envs = envs.num_envs
        videos, episode_videos, images = [[] for _ in range(num_envs)], [], None
        _current_episode, _current_step, scores, best_score = 0, 0, [], -np.inf
        obs_list, info = envs.reset()
        avail_actions = envs.buf_avail_actions if self.use_actions_mask else None
        state = envs.buf_state if self.use_global_state else None
        if test_mode:
            if self.config.render_mode == "rgb_array" and self.render:
                images = envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)
        else:
            if self.use_rnn:
                self.memory.clear_episodes()
        rnn_states_actor, rnn_states_critic = self.init_rnn_states(num_envs)

        while _current_episode < n_episodes:
            policy_out = self.get_actions(obs_list=obs_list, state=state, avail_actions_list=avail_actions,
                                          rnn_states_actor=rnn_states_actor, rnn_states_critic=rnn_states_critic,
                                          test_mode=test_mode, deterministic=deterministic_policy)
            rnn_states_actor, rnn_states_critic = policy_out['rnn_states_actor'], policy_out['rnn_states_critic']
            actions_list, log_pi_a_list = policy_out['actions'], policy_out['log_pi']
            values_dict = policy_out['values']
            next_obs_list, rewards_list, terminated_list, truncated, info = envs.step(actions_list)
            next_state = envs.buf_state if self.use_global_state else None
            next_avail_actions = envs.buf_avail_actions if self.use_actions_mask else None
            if test_mode:
                if self.config.render_mode == "rgb_array" and self.render:
                    images = envs.render(self.config.render_mode)
                    for idx, img in enumerate(images):
                        videos[idx].append(img)
            else:
                self.store_experience(obs_list, avail_actions, actions_list, log_pi_a_list, rewards_list, values_dict,
                                      terminated_list, info, **{'state': state})

            self.callback.on_test_step(envs=envs, policy=self.model, images=images, test_mode=test_mode,
                                       obs=obs_list, policy_out=policy_out, acts=actions_list,
                                       next_obs=next_obs_list, rewards=rewards_list,
                                       terminals=terminated_list, truncations=truncated, infos=info,
                                       state=state, next_state=next_state,
                                       current_train_step=self.current_step, n_episodes=n_episodes,
                                       current_step=_current_step, current_episode=_current_episode)

            obs_list, avail_actions = deepcopy(next_obs_list), deepcopy(next_avail_actions)
            state = deepcopy(next_state) if self.use_global_state else None

            for i in range(num_envs):
                if all(terminated_list[i].values()) or truncated[i]:
                    _current_episode += 1
                    episode_score = float(np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"])))
                    scores.append(episode_score)
                    if test_mode:
                        if self.use_rnn:
                            rnn_states_actor, _ = self.init_rnn_states_item(i, rnn_states_actor)
                        if best_score < episode_score:
                            best_score = episode_score
                            episode_videos = videos[i].copy()
                    else:
                        if all(terminated_list[i].values()):
                            value_next = {key: 0.0 for key in self.agent_keys}
                        else:
                            _, value_next = self.values_next(i_env=i, obs_dict=obs_list[i],
                                                             state=None if state is None else state[i],
                                                             rnn_states_critic=rnn_states_critic)
                        self.memory.finish_path(i_env=i, i_step=info[i]['episode_step'],
                                                agent_grouping=self.agent_grouping, value_next=value_next,
                                                value_normalizer=self.learner.value_normalizer)
                        if self.use_rnn:
                            rnn_states_actor, rnn_states_critic = self.init_rnn_states_item(i, rnn_states_actor,
                                                                                            rnn_states_critic)
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
                        self.callback.on_train_episode_info(envs=envs, policy=self.model, env_id=i,
                                                            infos=info, rank=self.rank, use_wandb=self.use_wandb,
                                                            current_step=self.current_step,
                                                            current_episode=self.current_episode,
                                                            n_episodes=n_episodes)

                    obs_list[i] = info[i]["reset_obs"]
                    envs.buf_obs[i] = info[i]["reset_obs"]
                    if self.use_actions_mask:
                        avail_actions[i] = info[i]["reset_avail_actions"]
                        envs.buf_avail_actions[i] = info[i]["reset_avail_actions"]
            _current_step += num_envs

        if test_mode:
            if self.config.render_mode == "rgb_array" and self.render:
                # time, height, width, channel -> time, channel, height, width
                videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
                self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)

            test_info = {
                "Test-Results/Episode-Rewards/Mean-Score": np.mean(scores),
                "Test-Results/Episode-Rewards/Std-Score": np.std(scores),
            }
            self.log_infos(test_info, self.current_step)

            self.callback.on_test_end(envs=envs, policy=self.model,
                                      current_train_step=self.current_step,
                                      current_step=_current_step, current_episode=_current_episode,
                                      scores=scores, best_score=best_score)

            if close_envs:
                envs.close()
        return scores

    def train_epochs(self, n_epochs: int = 1) -> dict:
        """Update policies for multiple epochs using mini-batches from the trajectory buffer.

        This method performs `n_epochs` optimization passes over the rollout data stored in `self.memory`.
        For each epoch, it shuffles transition indices and iterates over mini-batches to compute gradient updates via
        the learner. When RNN-based policies are enabled, the RNN-specific update method is used.

        Args:
            n_epochs (int): Number of optimization epochs to perform over the current trajectory buffer.

        Returns:
            dict: A dictionary of training metrics returned by the learner from the last mini-batch update (e.g., policy
                loss, value loss, entropy, KL divergence). Implementations may include additional diagnostics depending
                on the algorithm.
        """
        info_train = {}
        if self.memory.full:
            indexes = np.arange(self.buffer_size)
            for _ in range(n_epochs):
                np.random.shuffle(indexes)
                for start in range(0, self.buffer_size, self.batch_size):
                    end = start + self.batch_size
                    sample_idx = indexes[start:end]
                    sample = self.memory.sample(sample_idx)
                    info_train = self.learner.update(sample)
            self.callback.on_train_epochs_end(self.current_step, policy=self.model, memory=self.memory,
                                              current_episode=self.current_episode, n_epochs=n_epochs,
                                              buffer_size=self.buffer_size, update_info=info_train)
            self.memory.clear()
        return info_train

    def test(self,
             test_episodes: int,
             deterministic_policy: bool = True,
             test_envs: Optional[DummyVecMultiAgentEnv | SubprocVecMultiAgentEnv] = None,
             close_envs: bool = True) -> list:
        """Evaluate the current multi-agent policy for a number of episodes.

        This method runs evaluation episodes in `test_envs` by delegating to `run_episodes(test_mode=True)` and returns
        the per-episode scores. During evaluation, training-time outputs are skipped, exploration schedules are disabled
        by default, and optional RGB-array frames can be recorded and logged as a video when rendering is enabled.

        Args:
            test_episodes (int): Number of completed episodes to evaluate across all parallel environments.
            deterministic_policy (bool): True for evaluating the deterministic policy,
                and False for evaluating the stochastic policy.
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
            deterministic_policy=deterministic_policy,
            close_envs=close_envs
        )
        return scores
