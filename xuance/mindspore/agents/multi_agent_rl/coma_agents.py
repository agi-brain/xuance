import numpy as np
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from operator import itemgetter
from gymnasium.spaces import Space
from xuance.common import List, Optional, MultiAgentBaseCallback
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.mindspore import ms, ops, nn, Module, Tensor
from xuance.mindspore.utils import NormalizeFunctions, InitializeFunctions, ActivationFunctions
from xuance.mindspore.policies import REGISTRY_Policy
from xuance.mindspore.agents import OnPolicyMARLAgents


class COMA_Agents(OnPolicyMARLAgents):
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
        super(COMA_Agents, self).__init__(
            config, envs, num_agents, agent_keys, state_space, observation_space, action_space, callback
        )
        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.egreedy = self.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / config.decay_step_greedy

        self.use_global_state = True
        self.continuous_control = False
        self.state_space = envs.state_space

        self.policy = self._build_policy()  # build policy
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, self.policy, self.callback)
        self.learner.egreedy = self.egreedy
        self.softmax = nn.Softmax(axis=-1)

    def _build_policy(self) -> Module:
        """
        Build representation(s) and policy(ies) for agent(s)

        Returns:
            policy (torch.nn.Module): A dict of policies.
        """
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = InitializeFunctions[self.config.initialize] if hasattr(self.config, "initialize") else None
        activation = ActivationFunctions[self.config.activation]
        agent = self.config.agent

        # build representations
        A_representation = self._build_representation(self.config.representation, self.observation_space, self.config)
        C_representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policies
        if self.config.policy == "Categorical_COMA_Policy":
            policy = REGISTRY_Policy["Categorical_COMA_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                representation_actor=A_representation, representation_critic=C_representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None,
                dim_global_state=self.state_space.shape[0])
        else:
            raise AttributeError(f"COMA currently does not support the policy named {self.config.policy}.")

        return policy

    def store_experience(self, obs_dict, avail_actions, actions_dict, log_pi_a, rewards_dict, values_dict,
                         terminals_dict, info, **kwargs):
        """Store a batch of multi-agent transitions into the on-policy buffer.

        This method converts per-environment dictionaries (one dict per vector environment) into per-agent batched
        arrays and writes them into the on-policy trajectory buffer. It also stores auxiliary fields such as agent masks
        and (optionally) global state and action masks. For RNN-based policies, episode-step indices are recorded to
        support episode-aware bookkeeping.

        Args:
            obs_dict (List[dict]): Observations for each parallel environment.
                Each element is a dict keyed by `self.agent_keys`.
            avail_actions (Optional[List[dict]]): Available-action masks for each parallel environment when
                `use_actions_mask=True`. Each element is a dict keyed by `self.agent_keys`.
                Can be None when action masking is disabled.
            actions_dict (List[dict]): Actions executed by each agent for each parallel environment.
                Each element is a dict keyed by `self.agent_keys`.
            log_pi_a (dict): Log-probabilities of the actions under the current policy
                (typically computed during rollout collection).
            rewards_dict (List[dict]): Rewards for each agent for each parallel environment.
                Each element is a dict keyed by `self.agent_keys`.
            values_dict (dict): Value estimates produced by the critic for each agent
                (used for advantage/return computation).
            terminals_dict (List[dict]): Termination flags for each agent for each parallel environment.
                Each element is a dict keyed by `self.agent_keys`.
            info (List[dict]): Environment info for each parallel environment at the current step.
                Must contain `agent_mask` for each agent key.
            **kwargs: Optional extra fields. When `use_global_state=True`, this method expects `state` to be provided.
        """
        experience_data = {
            'obs': {k: np.array([data[k] for data in obs_dict]) for k in self.agent_keys},
            'actions': {k: np.array([data[k] for data in actions_dict]) for k in self.agent_keys},
            # 'log_pi_old': log_pi_a,
            'rewards': {k: np.array([np.array(list(data.values())).mean() for data in rewards_dict])
                        for k in self.agent_keys},
            'values': values_dict,
            'terminals': {k: np.array([data[k] for data in terminals_dict]) for k in self.agent_keys},
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

    def action(self,
               obs_dict: List[dict],
               state: Optional[np.ndarray] = None,
               avail_actions_dict: Optional[List[dict]] = None,
               rnn_hidden_actor: Optional[dict] = None,
               rnn_hidden_critic: Optional[dict] = None,
               test_mode: Optional[bool] = False,
               **kwargs):
        """Compute actions (and optional value/log-prob outputs) for multi-agent execution.

        This method performs a forward pass through the current multi-agent actor-critic policy to produce actions for
        each agent in each parallel environment. When RNN-based representations are enabled, the method consumes and
        returns recurrent hidden states for both the actor and the critic. During training (`test_mode=False`),
        this method also computes critic values and action log-probabilities needed for on-policy updates.
        During evaluation (`test_mode=True`), critic values and log-probabilities are not computed to reduce overhead.

        Args:
            obs_dict (List[dict]): Observations for each parallel environment.
                Each element is a dict keyed by `self.agent_keys`.
            state (Optional[np.ndarray]): Global state array used by centralized critics when `use_global_state=True`.
                The expected shape depends on the environment wrapper.
            avail_actions_dict (Optional[List[dict]]): Available-action masks for each parallel environment when
                `use_actions_mask=True`. Each element is a dict keyed by `self.agent_keys`.
                Can be None when action masking is disabled.
            rnn_hidden_actor (Optional[dict]): Current actor RNN hidden states keyed by `self.model_keys`.
                Required when `self.use_rnn` is True.
            rnn_hidden_critic (Optional[dict]): Current critic RNN hidden states keyed by `self.model_keys`.
                Required when `self.use_rnn` is True and values are requested.
            test_mode (bool): Whether to run in evaluation mode. When True, only actions are produced and
                training-specific outputs (values/log_pi) are omitted.

        Returns:
            dict: A dictionary containing:
                - rnn_hidden_actor (Optional[dict]): Updated actor RNN hidden states when `self.use_rnn` is True;
                    otherwise the value returned by the policy (typically None).
                - rnn_hidden_critic (Optional[dict]): Updated critic RNN hidden states when computed;
                    otherwise an empty dict.
                - actions (List[dict]): Actions for each parallel environment. Each element is a dict keyed by
                    `self.agent_keys`.
                - log_pi (dict): Log-probabilities of sampled actions for each agent when `test_mode=False`;
                    otherwise an empty dict.
                - values (dict): Critic value estimates for each agent when `test_mode=False`; otherwise an empty dict.
        """
        n_env = len(obs_dict)
        rnn_hidden_critic_new, log_pi_a_dict, values_dict, actions_out = {}, {}, {}, None

        obs_input, agents_id, avail_actions_input = self._build_inputs(obs_dict, avail_actions_dict)
        rnn_hidden_actor_new, pi_logits = self.policy(observation=obs_input,
                                                      agent_ids=agents_id,
                                                      avail_actions=avail_actions_input,
                                                      rnn_hidden=rnn_hidden_actor,
                                                      epsilon=self.egreedy,
                                                      test_mode=test_mode)

        if self.use_parameter_sharing:
            key = self.agent_keys[0]
            if test_mode:
                actions_sample = ops.argmax(pi_logits[key], dim=-1, keepdim=True).astype(ms.int32)
            else:
                pi_probs = self.softmax(pi_logits[key])
                pi_probs = (1 - self.egreedy) * pi_probs + self.egreedy * 1 / self.action_space[key].n
                self.policy.actor[key].dist.set_param(probs=pi_probs)
                pi_dists = self.policy.actor[key].dist
                actions_sample = pi_dists.stochastic_sample()
            actions_out = actions_sample.reshape(n_env, self.n_agents)
            actions_dict = [{k: actions_out[e, i].asnumpy() for i, k in enumerate(self.agent_keys)}
                            for e in range(n_env)]
            actions_onehot = {key: ops.one_hot(actions_out, depth=int(self.action_space[key].n)).astype(ms.float32)}
        else:
            agents_id = ops.eye(self.n_agents).unsqueeze(0).expand(n_env, -1, -1)
            bs = n_env * self.n_agents
            agents_id = agents_id.reshape(bs, 1, -1) if self.use_rnn else agents_id.reshape(bs, -1)
            if test_mode:
                actions_sample = {k: pi_logits[k].max(dim=-1)[1] for k in self.agent_keys}
            else:
                actions_sample = {}
                for k in self.agent_keys:
                    pi_probs = self.softmax(pi_logits[k])
                    pi_probs = (1 - self.egreedy) * pi_probs + self.egreedy * 1 / self.action_space[k].n
                    self.policy.actor[k].dist.set_param(probs=pi_probs)
                    pi_dists = self.policy.actor[k].dist
                    actions_sample[k] = pi_dists.stochastic_sample()
            actions_out = ops.stack(itemgetter(*self.agent_keys)(actions_sample), axis=-1)
            actions_dict = [{k: actions_sample[k].asnumpy()[e].reshape([]) for k in self.agent_keys}
                            for e in range(n_env)]
            actions_onehot = {k: ops.one_hot(actions_sample[k], depth=int(self.action_space[k].n)).astype(ms.float32)
                              for k in self.agent_keys}

        if not test_mode:  # calculate target values
            if self.use_rnn:
                state = Tensor(np.array(state)).reshape(n_env, 1, -1)
                if self.use_parameter_sharing:
                    actions_onehot = {k: actions_onehot[k].unsqueeze(1) for k in self.model_keys}
                else:
                    actions_onehot = {k: actions_onehot[k] for k in self.model_keys}
            else:
                state = Tensor(np.array(state)).reshape(n_env, -1)

            rnn_hidden_critic_new, values_out = self.policy.get_values(state=state,
                                                                       observation=obs_input,
                                                                       actions=actions_onehot,
                                                                       agent_ids=agents_id,
                                                                       rnn_hidden=rnn_hidden_critic,
                                                                       target=True)
            if self.use_rnn:
                values_out = values_out.reshape(n_env, self.n_agents, -1)
                actions_out = actions_out.reshape(n_env, self.n_agents)
            values_out = values_out.gather(-1, actions_out.unsqueeze(-1)).reshape(n_env, self.n_agents)
            values_out = values_out.asnumpy()
            values_dict = {k: values_out[:, i] for i, k in enumerate(self.agent_keys)}
        return {"rnn_hidden_actor": rnn_hidden_actor_new, "rnn_hidden_critic": rnn_hidden_critic_new,
                "actions": actions_dict, "log_pi": log_pi_a_dict, "values": values_dict}

    def values_next(self,
                    i_env: int,
                    obs_dict: dict,
                    state: Optional[np.ndarray] = None,
                    actions_n: Optional[np.ndarray] = None,
                    rnn_hidden_critic: Optional[dict] = None):
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
            rnn_hidden_critic (Optional[dict]): Current critic RNN hidden states keyed by `self.model_keys`.
                Required when `self.use_rnn` is True.

        Returns:
            Tuple[Optional[dict], dict]: A tuple of `(rnn_hidden_critic_new, values_dict)`:
                - rnn_hidden_critic_new (Optional[dict]): Updated critic hidden states for the selected environment
                    when `self.use_rnn` is True; otherwise the value returned by the critic (typically None).
                - values_dict (dict): Per-agent critic value estimates keyed by `self.agent_keys`.
        """
        n_env = 1
        bs = n_env * self.n_agents
        rnn_hidden_critic_i = None

        agents_id = ops.repeat_elements(ops.eye(self.n_agents, dtype=ms.float32).unsqueeze(0), rep=n_env, axis=0)
        if self.use_rnn:
            state = state.reshape(n_env, 1, -1)
            agents_id = agents_id.reshape(bs, 1, -1)
        else:
            state = state.reshape(n_env, -1)
            agents_id.reshape(bs, -1)

        if self.use_parameter_sharing:
            key = self.agent_keys[0]
            actions_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(actions_n)))
            if self.use_rnn:
                hidden_item_index = np.arange(i_env * self.n_agents, (i_env + 1) * self.n_agents)
                rnn_hidden_critic_i = {key: self.policy.critic_representation[key].get_hidden_item(
                    hidden_item_index, *rnn_hidden_critic[key])}
                obs_array = np.array(itemgetter(*self.agent_keys)(obs_dict))
                obs_input = {key: Tensor(obs_array.reshape([bs, 1, -1]))}
                actions_tensor = actions_tensor.reshape(n_env, 1, self.n_agents)
            else:
                obs_input = {key: Tensor(np.array([itemgetter(*self.agent_keys)(obs_dict)]))}
                actions_tensor = actions_tensor.reshape(n_env, self.n_agents)
            actions_onehot = {key: ops.one_hot(actions_tensor.long(),
                                               depth=int(self.action_space[key].n)).astype(ms.float32)}
        else:
            if self.use_rnn:
                rnn_hidden_critic_i = {k: self.policy.critic_representation[k].get_hidden_item(
                    [i_env, ], *rnn_hidden_critic[k]) for k in self.agent_keys}
                obs_input = {k: Tensor(obs_dict[k][None, None, :]) for k in self.agent_keys}
            else:
                obs_input = {k: Tensor(obs_dict[k][None, :]) for k in self.agent_keys}
            actions_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(actions_n))).reshape(n_env, self.n_agents)
            actions_onehot = {k: ops.one_hot(actions_tensor[:, i].long(),
                                             depth=int(self.action_space[k].n)).astype(ms.float32)
                              for i, k in enumerate(self.agent_keys)}

        rnn_hidden_critic_new, values_out = self.policy.get_values(state=Tensor(state),
                                                                   observation=obs_input,
                                                                   actions=actions_onehot,
                                                                   agent_ids=agents_id,
                                                                   rnn_hidden=rnn_hidden_critic_i,
                                                                   target=True)
        if self.use_rnn:
            values_out = values_out.reshape(n_env, self.n_agents, -1)
            actions_tensor = actions_tensor.reshape(n_env, self.n_agents)
        values_out = values_out.gather(-1, actions_tensor.unsqueeze(-1).long())
        values_out = values_out.asnumpy().reshape(self.n_agents)
        values_dict = {k: values_out[i] for i, k in enumerate(self.agent_keys)}
        return rnn_hidden_critic_new, values_dict

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
                    self.run_episodes(None, n_episodes=self.n_envs, test_mode=False)
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
        state = self.train_envs.buf_state if self.use_global_state else None
        for _ in tqdm(range(train_steps)):
            policy_out = self.action(obs_dict=obs_dict, state=state, avail_actions_dict=avail_actions, test_mode=False)
            actions_dict, log_pi_a_dict = policy_out['actions'], policy_out['log_pi']
            values_dict = policy_out['values']
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = self.train_envs.step(actions_dict)
            next_state = self.train_envs.buf_state.copy() if self.use_global_state else None
            next_avail_actions = self.train_envs.buf_avail_actions if self.use_actions_mask else None

            self.callback.on_train_step(self.current_step, envs=self.train_envs, policy=self.policy,
                                        obs=obs_dict, policy_out=policy_out, acts=actions_dict, next_obs=next_obs_dict,
                                        rewards=rewards_dict, state=state, next_state=next_state,
                                        avail_actions=avail_actions, next_avail_actions=next_avail_actions,
                                        terminals=terminated_dict, truncations=truncated, infos=info,
                                        train_steps=train_steps, values_dict=values_dict)

            self.store_experience(obs_dict, avail_actions, actions_dict, log_pi_a_dict, rewards_dict, values_dict,
                                  terminated_dict, info, **{'state': state})
            if self.memory.full:
                for i in range(self.n_envs):
                    if all(terminated_dict[i].values()):
                        value_next = {key: 0.0 for key in self.agent_keys}
                    else:
                        state_i = state[i] if self.use_global_state else None
                        _, value_next = self.values_next(i_env=i, obs_dict=next_obs_dict[i],
                                                         state=state_i, actions_n=actions_dict[i])
                    self.memory.finish_path(i_env=i, value_next=value_next,
                                            value_normalizer=self.learner.value_normalizer)
            update_info = self.train_epochs(n_epochs=self.n_epochs)
            self.log_infos(update_info, self.current_step)
            train_info.update(update_info)
            obs_dict, avail_actions = deepcopy(next_obs_dict), deepcopy(next_avail_actions)
            state = self.train_envs.buf_state if self.use_global_state else None

            for i in range(self.n_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    if all(terminated_dict[i].values()):
                        value_next = {key: 0.0 for key in self.agent_keys}
                    else:
                        state_i = state[i] if self.use_global_state else None
                        _, value_next = self.values_next(i_env=i, obs_dict=obs_dict[i],
                                                         state=state_i, actions_n=actions_dict[i])
                    self.memory.finish_path(i_env=i, value_next=value_next,
                                            value_normalizer=self.learner.value_normalizer)
                    obs_dict[i] = info[i]["reset_obs"]
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
            self.callback.on_train_step_end(self.current_step, envs=self.train_envs, policy=self.policy,
                                            train_steps=train_steps, train_info=train_info)
        return train_info

    def run_episodes(self,
                     n_episodes: int = 1,
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
        current_episode, current_step, scores, best_score = 0, 0, [0.0 for _ in range(num_envs)], -np.inf
        obs_dict, info = envs.reset()
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
        rnn_hidden_actor, rnn_hidden_critic = self.init_rnn_hidden(num_envs)

        while current_episode < n_episodes:
            policy_out = self.action(obs_dict=obs_dict, state=state, avail_actions_dict=avail_actions,
                                     rnn_hidden_actor=rnn_hidden_actor, rnn_hidden_critic=rnn_hidden_critic,
                                     test_mode=test_mode)
            rnn_hidden_actor, rnn_hidden_critic = policy_out['rnn_hidden_actor'], policy_out['rnn_hidden_critic']
            actions_dict, log_pi_a_dict = policy_out['actions'], policy_out['log_pi']
            values_dict = policy_out['values']
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = envs.step(actions_dict)
            next_state = envs.buf_state if self.use_global_state else None
            next_avail_actions = envs.buf_avail_actions if self.use_actions_mask else None
            if test_mode:
                if self.config.render_mode == "rgb_array" and self.render:
                    images = envs.render(self.config.render_mode)
                    for idx, img in enumerate(images):
                        videos[idx].append(img)
            else:
                self.store_experience(obs_dict, avail_actions, actions_dict, log_pi_a_dict, rewards_dict, values_dict,
                                      terminated_dict, info, **{'state': state})

            self.callback.on_test_step(envs=envs, policy=self.policy, images=images, test_mode=test_mode,
                                       obs=obs_dict, policy_out=policy_out, acts=actions_dict,
                                       next_obs=next_obs_dict, rewards=rewards_dict,
                                       terminals=terminated_dict, truncations=truncated, infos=info,
                                       state=state, next_state=next_state,
                                       current_train_step=self.current_step, n_episodes=n_episodes,
                                       current_step=current_step, current_episode=current_episode)

            obs_dict, avail_actions = deepcopy(next_obs_dict), deepcopy(next_avail_actions)
            state = envs.buf_state if self.use_global_state else None

            for i in range(num_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    current_episode += 1
                    episode_score = float(np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"])))
                    scores.append(episode_score)
                    if test_mode:
                        if self.use_rnn:
                            rnn_hidden_actor, _ = self.init_hidden_item(i, rnn_hidden_actor)
                        if best_score < episode_score:
                            best_score = episode_score
                            episode_videos = videos[i].copy()
                        if self.config.test_mode:
                            print("Episode: %d, Score: %.2f" % (current_episode, episode_score))
                    else:
                        if all(terminated_dict[i].values()):
                            value_next = {key: 0.0 for key in self.agent_keys}
                        else:
                            _, value_next = self.values_next(i_env=i, obs_dict=obs_dict[i],
                                                             state=state[i], actions_n=actions_dict[i],
                                                             rnn_hidden_critic=rnn_hidden_critic)
                        self.memory.finish_path(i_env=i, i_step=info[i]['episode_step'], value_next=value_next,
                                                value_normalizer=self.learner.value_normalizer)
                        if self.use_rnn:
                            rnn_hidden_actor, rnn_hidden_critic = self.init_hidden_item(i, rnn_hidden_actor,
                                                                                        rnn_hidden_critic)
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
                        self.callback.on_train_episode_info(envs=self.train_envs, policy=self.policy, env_id=i,
                                                            infos=info, use_wandb=self.use_wandb,
                                                            current_step=self.current_step,
                                                            current_episode=self.current_episode,
                                                            n_episodes=n_episodes)
                    obs_dict[i] = info[i]["reset_obs"]
                    envs.buf_obs[i] = info[i]["reset_obs"]
                    if self.use_actions_mask:
                        avail_actions[i] = info[i]["reset_avail_actions"]
                        envs.buf_avail_actions[i] = info[i]["reset_avail_actions"]
            current_step += num_envs

        if test_mode:
            if self.config.render_mode == "rgb_array" and self.render:
                # time, height, width, channel -> time, channel, height, width
                videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
                self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)

            if self.config.test_mode:
                print("Best Score: %.2f" % best_score)

            test_info = {
                "Test-Results/Episode-Rewards/Mean-Score": np.mean(scores),
                "Test-Results/Episode-Rewards/Std-Score": np.std(scores),
            }
            self.log_infos(test_info, self.current_step)

            self.callback.on_test_end(envs=envs, policy=self.policy,
                                      current_train_step=self.current_step,
                                      current_step=current_step, current_episode=current_episode,
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
        if self.egreedy >= self.end_greedy:
            self.egreedy = self.start_greedy - self.delta_egreedy * self.current_step
        info_train = {}
        if self.memory.full:
            indexes = np.arange(self.buffer_size)
            for _ in range(n_epochs):
                np.random.shuffle(indexes)
                for start in range(0, self.buffer_size, self.batch_size):
                    end = start + self.batch_size
                    sample_idx = indexes[start:end]
                    sample = self.memory.sample(sample_idx)
                    if self.use_rnn:
                        info_train = self.learner.update_rnn(sample, self.egreedy)
                    else:
                        info_train = self.learner.update(sample, self.egreedy)
            self.callback.on_train_epochs_end(self.current_step, policy=self.policy, memory=self.memory,
                                              current_episode=self.current_episode, n_epochs=n_epochs,
                                              buffer_size=self.buffer_size, update_info=info_train)
            self.memory.clear()
        info_train["epsilon-greedy"] = self.egreedy
        return info_train
