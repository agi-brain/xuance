import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from operator import itemgetter
from gymnasium.spaces import Space
from typing import List, Optional, Dict, Tuple
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.common import MeanField_OnPolicyBuffer, MeanField_OnPolicyBuffer_RNN, MultiAgentBaseCallback
from xuance.torch import Module, ModuleDict
from xuance.torch.agents import OnPolicyMARLAgents
from xuance.torch.rl_models import CategoricalActor as Actor
from xuance.torch.rl_models import MeanFieldStateValueCritic as Critic
from xuance.torch.rl_models.modules import RNN_State
from xuance.torch.rl_models.architectures import MeanFiledActorCritic


class MFAC_Agents(OnPolicyMARLAgents):
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
        super(MFAC_Agents, self).__init__(
            config, envs, num_agents, agent_keys, state_space, observation_space, action_space, callback
        )

        self.n_actions_list = [a_space.n for a_space in self.action_space.values()]
        self.n_actions_max = max(self.n_actions_list)
        self.actions_mean = [{k: np.zeros(self.n_actions_max) for k in self.agent_keys} for _ in range(self.n_envs)]

        self.model = self._build_model()  # build the MARL model
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.agent_grouping, self.model, self.callback)

    def _build_memory(self) -> MeanField_OnPolicyBuffer | MeanField_OnPolicyBuffer_RNN:
        """Build replay buffer for models training
        """
        if self.use_actions_mask:
            avail_actions_shape = {key: (self.action_space[key].n,) for key in self.agent_keys}
        else:
            avail_actions_shape = None
        input_dict = dict(agent_keys=self.agent_keys,
                          state_space=self.state_space if self.use_global_state else None,
                          obs_space=self.observation_space,
                          act_space=self.action_space,
                          n_envs=self.n_envs,
                          buffer_size=self.buffer_size,
                          batch_size=self.batch_size,
                          use_gae=self.config.use_gae,
                          use_advnorm=self.config.use_advnorm,
                          gamma=self.config.gamma,
                          gae_lam=self.config.gae_lambda,
                          avail_actions_shape=avail_actions_shape,
                          use_actions_mask=self.use_actions_mask,
                          max_episode_steps=self.episode_length,
                          n_actions_max=self.n_actions_max)
        Buffer = MeanField_OnPolicyBuffer_RNN if self.use_rnn else MeanField_OnPolicyBuffer
        return Buffer(**input_dict)

    def _build_model(self) -> Module:
        """
        Build the MARL model.

        Returns:
            model (torch.nn.Module): The MARL model.
        """
        actor_networks = ModuleDict()
        critic_networks = ModuleDict()
        for group_key, group_agents in self.groups.items():
            reference_agent = group_agents[0]
            # build agent feature encoder as actor representations
            actor_feature_encoder = self._build_agent_feature_encoder(
                representation_choice=self.config.representation,
                group_agents=group_agents,
                input_space=self.observation_space[reference_agent]
            )
            # build inner-group shared actor-network
            actor_networks[group_key] = Actor(
                representation=actor_feature_encoder,
                action_space=self.action_space[reference_agent],
                actor_hidden_size=self.config.actor_hidden_size,
                normalizer=self.normalize_fn,
                initializer=self.initializer,
                activation=self.activation,
                device=self.device
            )
            # build mean actions embedding network
            mean_actions_encoder = self._build_agent_feature_encoder(
                representation_choice="Basic_MLP",
                group_agents=group_agents,
                input_space=(self.n_actions_max,)
            )
            # build critic feature encoder as critic representations
            critic_feature_encoder = self._build_agent_feature_encoder(
                representation_choice=self.config.representation,
                group_agents=group_agents,
                input_space=self.observation_space[reference_agent]
            )
            # build inner-group shared critic-network
            critic_networks[group_key] = Critic(
                representation=critic_feature_encoder,
                mean_actions_encoder=mean_actions_encoder,
                critic_hidden_size=self.config.critic_hidden_size,
                normalizer=self.normalize_fn,
                initializer=self.initializer,
                activation=self.activation,
                device=self.device
            )

        model = MeanFiledActorCritic(
            grouping=self.agent_grouping,
            actors=actor_networks,
            critics=critic_networks,
            use_rnn=self.use_rnn,
            device=self.device,
            use_distributed_training=self.distributed_training,
            temperature=self.config.temperature,
            n_actions_max=self.n_actions_max
        )

        return model

    def _build_inputs_mean_mask(self,
                                agent_mask: Optional[dict] = None,
                                act_mean_dict=None):
        batch_size = len(act_mean_dict)
        mean_actions_input = {}
        agent_mask_array = np.array([itemgetter(*self.agent_keys)(data) for data in agent_mask])
        # get mean actions as input
        for group, group_agents in self.groups.items():
            bs = batch_size * len(group_agents)
            mean_actions_array = np.array([itemgetter(*self.agent_keys)(data) for data in act_mean_dict])
            if self.use_rnn:
                mean_actions_input[group] = mean_actions_array.reshape([bs, 1, -1])
            else:
                mean_actions_input[group] = mean_actions_array.reshape([bs, -1])

        return mean_actions_input, agent_mask_array

    def store_experience(self, obs_dict, avail_actions, actions_dict, log_pi_a, rewards_dict, values_dict,
                         terminals_dict, info, **kwargs):
        """
        Store experience data into replay buffer.

        Parameters:
            obs_dict (List[dict]): Observations for each agent in self.agent_keys.
            avail_actions (List[dict]): Actions mask values for each agent in self.agent_keys.
            actions_dict (List[dict]): Actions for each agent in self.agent_keys.
            log_pi_a (dict): The log of pi.
            rewards_dict (List[dict]): Rewards for each agent in self.agent_keys.
            values_dict (dict): Critic values for each agent in self.agent_keys.
            terminals_dict (List[dict]): Terminated values for each agent in self.agent_keys.
            info (List[dict]): Other information for the environment at current step.
            **kwargs: Other inputs.
        """
        experience_data = {
            'obs': {k: np.array([data[k] for data in obs_dict]) for k in self.agent_keys},
            'actions': {k: np.array([data[k] for data in actions_dict]) for k in self.agent_keys},
            'log_pi_old': log_pi_a,
            'rewards': {k: np.array([data[k] for data in rewards_dict]) for k in self.agent_keys},
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
        experience_data['actions_mean'] = {k: np.array([data[k] for data in kwargs['actions_mean']])
                                           for k in self.agent_keys}
        self.memory.store(**experience_data)

    def get_actions(self,
                    obs_dict: List[dict],
                    agent_mask: Optional[List[dict]] = None,
                    act_mean_dict: Optional[List[dict]] = None,
                    state: Optional[np.ndarray] = None,
                    avail_actions_dict: Optional[List[dict]] = None,
                    rnn_states_actor: Optional[Dict[str, RNN_State]] = None,
                    rnn_states_critic: Optional[Dict[str, RNN_State]] = None,
                    test_mode: Optional[bool] = False,
                    deterministic: bool = False,
                    **kwargs):
        """
        Returns actions for agents.

        Parameters:
            obs_dict (dict): Observations for each agent in self.agent_keys.
            agent_mask (Optional[List[dict]]): Mask the agents that are alive.
            state (Optional[np.ndarray]): The global state.
            act_mean_dict (Optional[List[dict]]): Mean actions of each agent's neighbors.
            avail_actions_dict (Optional[List[dict]]): Actions mask values, default is None.
            rnn_states_actor (Optional[dict]): The RNN hidden states of actor representation.
            rnn_states_critic (Optional[dict]): The RNN hidden states of critic representation.
            test_mode (Optional[bool]): True for testing without noises.
            deterministic (bool): True for deterministic policy and False for stochastic policy.

        Returns:
            rnn_states_actor_new (dict): The new RNN hidden states of actor representation (if self.use_rnn=True).
            rnn_states_critic_new (dict): The new RNN hidden states of critic representation (if self.use_rnn=True).
            actions_dict (dict): The output actions.
            log_pi_a (dict): The log of pi.
            values_dict (dict): The evaluated critic values (when test_mode is False).
        """
        n_env = len(obs_dict)
        rnn_states_critic_new, values_out, log_pi_a_dict, values_dict = {}, {}, {}, {}

        mean_actions_input, agent_mask_array = self._build_inputs_mean_mask(agent_mask, act_mean_dict)
        obs_input, agent_indices, avail_actions_input = self._build_inputs(obs_dict, avail_actions_dict)
        agent_mask_tensor = torch.tensor(agent_mask_array, dtype=torch.float32, device=self.device)
        model_output = self.model(observations=obs_input,
                                  agent_indices=agent_indices,
                                  avail_actions=avail_actions_input,
                                  rnn_states=rnn_states_actor,
                                  deterministic=deterministic)
        rnn_states_actor_new = model_output.actor_rnn_states
        actions = model_output.actions

        actions_out = {k: actions[k].reshape(n_env).cpu().detach().numpy() for k in self.agent_keys}
        actions_dict = [{k: actions_out[k][i] for k in self.agent_keys} for i in range(n_env)]
        actions_mean_masked = self.model.get_mean_actions(actions=actions, agent_mask_tensor=agent_mask_tensor,
                                                          batch_size=n_env)
        actions_mean_masked = {k: v.cpu().detach().numpy() for k, v in actions_mean_masked.items()}
        actions_mean_dict = [{k: v[e] for k, v in actions_mean_masked.items()} for e in range(n_env)]

        if not test_mode:
            for group, agent_keys in self.groups.items():
                n_agents = self.n_group_agents[group]
                actions_group = torch.stack([model_output.actions[k] for k in agent_keys], dim=1)
                if self.use_rnn:
                    actions_group = actions_group.reshape([n_env * n_agents, -1])
                else:
                    actions_group = actions_group.reshape([n_env * n_agents])
                log_pi_a = model_output.distributions[group].log_prob(actions_group).reshape(n_env, n_agents)
                for i, agent in enumerate(agent_keys):
                    log_pi_a_dict[agent] = log_pi_a[:, i].detach().cpu().numpy()

            values_model_output = self.model.get_values(observations=obs_input,
                                                        mean_actions=mean_actions_input,
                                                        agent_indices=agent_indices,
                                                        rnn_states=rnn_states_critic)
            rnn_states_critic_new = values_model_output.critic_rnn_states
            values_dict = {k: v.detach().cpu().numpy().reshape(n_env) for k, v in values_model_output.values.items()}

        return {"rnn_states_actor": rnn_states_actor_new, "rnn_states_critic": rnn_states_critic_new,
                "actions": actions_dict, "actions_mean": actions_mean_dict,
                "log_pi": log_pi_a_dict, "values": values_dict}

    def values_next(
            self,
            i_env: int,
            obs_dict: dict,
            state: Optional[np.ndarray] = None,
            agent_mask: dict = None,
            act_mean_dict: dict = None,
            rnn_states_critic: Dict[str, RNN_State] = None
    ) -> Tuple[Dict[str, RNN_State], Dict[str, np.ndarray]]:
        """
        Returns critic values of one environment that finished an episode.

        Parameters:
            i_env (int): The index of environment.
            obs_dict (dict): Observations for each agent in self.agent_keys.
            state (Optional[np.ndarray]): The global state.
            agent_mask (dict): Mask the agents that are alive.
            act_mean_dict (dict): The mean actions of each agent's neighbors.
            rnn_states_critic (Optional[dict]): The RNN hidden states of critic representation.

        Returns:
            rnn_states_critic_new (dict): The new RNN hidden states of critic representation (if self.use_rnn=True).
            values_dict: The critic values.
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

        mean_actions_input, _ = self._build_inputs_mean_mask([agent_mask], [act_mean_dict])
        obs_input, agent_indices, _ = self._build_inputs([obs_dict])

        values_model_output = self.model.get_values(state=state if self.use_global_state else None,
                                                    observations=obs_input,
                                                    agent_indices=agent_indices,
                                                    mean_actions=mean_actions_input,
                                                    rnn_states=rnn_states_critic_i)
        rnn_states_critic_new_i = values_model_output.critic_rnn_states
        values_dict = {k: v.detach().cpu().numpy().reshape([]) for k, v in values_model_output.values.items()}

        return rnn_states_critic_new_i, values_dict

    def train(self, train_steps: int) -> dict:
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

                    self.callback.on_train_epochs_end(self.current_step, model=self.model, memory=self.memory,
                                                      current_episode=self.current_episode, train_steps=train_steps,
                                                      update_info=update_info)

                    process_bar.update((self.current_step - step_last) // self.n_envs)
                    step_last = deepcopy(self.current_step)
                process_bar.update(train_steps - process_bar.last_print_n)
                self.callback.on_train_step_end(self.current_step, envs=self.train_envs, model=self.model,
                                                n_steps=train_steps, train_info=train_info)
            return train_info

        obs_dict = self.train_envs.buf_obs
        agent_mask_dict = [data['agent_mask'] for data in self.train_envs.buf_info]
        actions_mean_dict = self.actions_mean
        avail_actions = self.train_envs.buf_avail_actions if self.use_actions_mask else None
        state = self.train_envs.buf_state if self.use_global_state else None
        for _ in tqdm(range(train_steps)):
            policy_out = self.get_actions(obs_dict=obs_dict, state=state,
                                          agent_mask=agent_mask_dict, act_mean_dict=actions_mean_dict,
                                          avail_actions_dict=avail_actions, test_mode=False)
            actions_dict, log_pi_a_dict = policy_out['actions'], policy_out['log_pi']
            actions_mean_next_dict = policy_out['actions_mean']
            values_dict = policy_out['values']
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = self.train_envs.step(actions_dict)
            next_state = self.train_envs.buf_state if self.use_global_state else None
            next_avail_actions = self.train_envs.buf_avail_actions if self.use_actions_mask else None

            self.callback.on_train_step(self.current_step, envs=self.train_envs, model=self.model,
                                        obs=obs_dict, policy_out=policy_out, acts=actions_dict, next_obs=next_obs_dict,
                                        rewards=rewards_dict, state=state, next_state=next_state,
                                        avail_actions=avail_actions, next_avail_actions=next_avail_actions,
                                        actions_mean_dict=actions_mean_dict,
                                        terminals=terminated_dict, truncations=truncated, infos=info,
                                        train_steps=train_steps, values_dict=values_dict)

            self.store_experience(obs_dict, avail_actions, actions_dict, log_pi_a_dict, rewards_dict, values_dict,
                                  terminated_dict, info,
                                  **{'state': state, 'actions_mean': actions_mean_dict})
            if self.memory.full:
                for i in range(self.n_envs):
                    if all(terminated_dict[i].values()):
                        value_next = {key: 0.0 for key in self.agent_keys}
                    else:
                        next_state_i = next_state[i] if self.use_global_state else None
                        _, value_next = self.values_next(i_env=i, obs_dict=next_obs_dict[i], state=next_state_i,
                                                         act_mean_dict=actions_mean_dict[i],
                                                         agent_mask=agent_mask_dict[i])
                    self.memory.finish_path(i_env=i, agent_grouping=self.agent_grouping, value_next=value_next,
                                            value_normalizer=self.learner.value_normalizer)
            update_info = self.train_epochs(n_epochs=self.n_epochs)
            self.log_infos(update_info, self.current_step)
            train_info.update(update_info)
            obs_dict, avail_actions = deepcopy(next_obs_dict), deepcopy(next_avail_actions)
            state = deepcopy(next_state) if self.use_global_state else None
            agent_mask_dict = [data['agent_mask'] for data in info]
            actions_mean_dict = deepcopy(actions_mean_next_dict)

            for i in range(self.n_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    if all(terminated_dict[i].values()):
                        value_next = {key: 0.0 for key in self.agent_keys}
                    else:
                        state_i = state[i] if self.use_global_state else None
                        _, value_next = self.values_next(i_env=i, obs_dict=obs_dict[i], state=state_i,
                                                         act_mean_dict=actions_mean_dict[i],
                                                         agent_mask=agent_mask_dict[i])
                    self.memory.finish_path(i_env=i, agent_grouping=self.agent_grouping, value_next=value_next,
                                            value_normalizer=self.learner.value_normalizer)
                    obs_dict[i] = info[i]["reset_obs"]
                    self.train_envs.buf_obs[i] = info[i]["reset_obs"]
                    if self.use_actions_mask:
                        avail_actions[i] = info[i]["reset_avail_actions"]
                        self.train_envs.buf_avail_actions[i] = info[i]["reset_avail_actions"]
                    if self.use_global_state:
                        state[i] = info[i]["reset_state"]
                        self.train_envs.buf_state[i] = info[i]["reset_state"]
                    self.train_envs.buf_info[i]["agent_mask"] = {k: True for k in self.agent_keys}
                    agent_mask_dict[i] = {k: True for k in self.agent_keys}
                    actions_mean_dict[i] = {k: np.zeros(self.n_actions_max) for k in self.agent_keys}
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
                    self.callback.on_train_episode_info(envs=self.train_envs, model=self.model, env_id=i,
                                                        infos=info, rank=self.rank, use_wandb=self.use_wandb,
                                                        current_step=self.current_step,
                                                        current_episode=self.current_episode,
                                                        train_steps=train_steps)

            self.current_step += self.n_envs
            self.actions_mean = deepcopy(actions_mean_dict)
            self.callback.on_train_step_end(self.current_step, envs=self.train_envs, model=self.model,
                                            train_steps=train_steps, train_info=train_info)
        return train_info

    def run_episodes(self,
                     n_episodes: int = 1,
                     deterministic_policy: bool = False,
                     run_envs: Optional[DummyVecMultiAgentEnv | SubprocVecMultiAgentEnv] = None,
                     test_mode: bool = False,
                     close_envs: bool = True) -> list:
        envs = self.train_envs if run_envs is None else run_envs
        num_envs = envs.num_envs
        videos, episode_videos, images = [[] for _ in range(num_envs)], [], None
        current_episode, current_step, scores, best_score = 0, 0, [], -np.inf
        obs_dict, info = envs.reset()
        agent_mask_dict = [data['agent_mask'] for data in info]
        actions_mean_dict = [{k: np.zeros(self.n_actions_max) for k in self.agent_keys} for _ in range(num_envs)]
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

        while current_episode < n_episodes:
            policy_out = self.get_actions(obs_dict=obs_dict, state=state, avail_actions_dict=avail_actions,
                                          agent_mask=agent_mask_dict, act_mean_dict=actions_mean_dict,
                                          rnn_states_actor=rnn_states_actor, rnn_states_critic=rnn_states_critic,
                                          test_mode=test_mode, deterministic=deterministic_policy)
            rnn_states_actor, rnn_states_critic = policy_out['rnn_states_actor'], policy_out['rnn_states_critic']
            actions_dict, log_pi_a_dict = policy_out['actions'], policy_out['log_pi']
            actions_mean_next_dict = policy_out['actions_mean']
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
                                      terminated_dict, info, **{'state': state, 'actions_mean': actions_mean_dict})

            self.callback.on_test_step(envs=envs, model=self.model, images=images, test_mode=test_mode,
                                       obs=obs_dict, policy_out=policy_out, acts=actions_dict,
                                       actions_mean_dict=actions_mean_dict,
                                       next_obs=next_obs_dict, rewards=rewards_dict,
                                       terminals=terminated_dict, truncations=truncated, infos=info,
                                       state=state, next_state=next_state,
                                       current_train_step=self.current_step, n_episodes=n_episodes,
                                       current_step=current_step, current_episode=current_episode)

            obs_dict, avail_actions = deepcopy(next_obs_dict), deepcopy(next_avail_actions)
            agent_mask_dict = [data['agent_mask'] for data in info]
            actions_mean_dict = deepcopy(actions_mean_next_dict)
            state = deepcopy(next_state) if self.use_global_state else None

            for i in range(num_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    current_episode += 1
                    episode_score = float(np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"])))
                    scores.append(episode_score)
                    if test_mode:
                        if self.use_rnn:
                            rnn_states_actor, _ = self.init_rnn_states_item(i, rnn_states_actor)
                        if best_score < episode_score:
                            best_score = episode_score
                            episode_videos = videos[i].copy()
                    else:
                        if all(terminated_dict[i].values()):
                            value_next = {key: 0.0 for key in self.agent_keys}
                        else:
                            _, value_next = self.values_next(i_env=i, obs_dict=obs_dict[i],
                                                             state=None if state is None else state[i],
                                                             act_mean_dict=actions_mean_dict[i],
                                                             agent_mask=agent_mask_dict[i],
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
                        self.callback.on_train_episode_info(envs=self.train_envs, model=self.model, env_id=i,
                                                            infos=info, rank=self.rank, use_wandb=self.use_wandb,
                                                            current_step=self.current_step,
                                                            current_episode=self.current_episode,
                                                            n_episodes=n_episodes)

                    obs_dict[i] = info[i]["reset_obs"]
                    envs.buf_obs[i] = info[i]["reset_obs"]
                    agent_mask_dict[i] = {k: True for k in self.agent_keys}
                    actions_mean_dict[i] = {k: np.zeros(self.n_actions_max) for k in self.agent_keys}
                    if self.use_actions_mask:
                        avail_actions[i] = info[i]["reset_avail_actions"]
                        envs.buf_avail_actions[i] = info[i]["reset_avail_actions"]
            current_step += num_envs

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

            self.callback.on_test_end(envs=envs, model=self.model,
                                      current_train_step=self.current_step,
                                      current_step=current_step, current_episode=current_episode,
                                      scores=scores, best_score=best_score)

            if close_envs:
                envs.close()
        return scores
