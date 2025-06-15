import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from operator import itemgetter
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.common import List, Optional, Union, MeanField_OnPolicyBuffer, MeanField_OnPolicyBuffer_RNN
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.agents import OnPolicyMARLAgents, BaseCallback


class MFAC_Agents(OnPolicyMARLAgents):
    """The implementation of Mean Field Actor-Critic (MFAC) agents.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """

    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv],
                 callback: Optional[BaseCallback] = None):
        super(MFAC_Agents, self).__init__(config, envs, callback)

        self.n_actions_list = [a_space.n for a_space in self.action_space.values()]
        self.n_actions_max = max(self.n_actions_list)
        self.actions_mean = [{k: np.zeros(self.n_actions_max) for k in self.agent_keys} for _ in range(self.n_envs)]

        self.policy = self._build_policy()  # build policy
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, self.policy, self.callback)

    def _build_memory(self):
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
        agent = self.config.agent

        # build representations
        A_representation = self._build_representation(self.config.representation, self.observation_space, self.config)
        C_representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policies
        if self.config.policy == "Categorical_MFAC_Policy":
            policy = REGISTRY_Policy["Categorical_MFAC_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                representation_actor=A_representation, representation_critic=C_representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                device=device, use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None,
                temperature=self.config.temperature,
                action_embedding_hidden_size=self.config.action_embedding_hidden_size)
            self.continuous_control = False
        elif self.config.policy == "Gaussian_MFAC_Policy":
            policy = REGISTRY_Policy["Gaussian_MFAC_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                representation_actor=A_representation, representation_critic=C_representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                activation_action=ActivationFunctions[self.config.activation_action],
                device=device, use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None,
                temperature=self.config.temperature,
                action_embedding_hidden_size=self.config.action_embedding_hidden_size)
            self.continuous_control = True
        else:
            raise AttributeError(f"{agent} currently does not support the policy named {self.config.policy}.")
        return policy

    def _build_inputs_mean_mask(self,
                                agent_mask: Optional[dict] = None,
                                act_mean_dict=None):
        batch_size = len(act_mean_dict)
        agent_mask_array = np.array([itemgetter(*self.agent_keys)(data) for data in agent_mask])
        # get mean actions as input
        if self.use_parameter_sharing:
            key = self.agent_keys[0]
            mean_actions_array = np.array([itemgetter(*self.agent_keys)(data) for data in act_mean_dict])
            if self.use_rnn:
                mean_actions_input = {key: mean_actions_array.reshape([batch_size * self.n_agents, 1, -1])}
            else:
                mean_actions_input = {key: mean_actions_array.reshape([batch_size * self.n_agents, -1])}
        else:
            if self.use_rnn:
                mean_actions_input = {k: np.stack([data[k] for data in act_mean_dict]).reshape([batch_size, 1, -1])
                                      for k in self.agent_keys}
            else:
                mean_actions_input = {k: np.stack([data[k] for data in act_mean_dict]).reshape(batch_size, -1)
                                      for k in self.agent_keys}
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

    def action(self,
               obs_dict: List[dict],
               agent_mask: Optional[List[dict]] = None,
               act_mean_dict: Optional[List[dict]] = None,
               state: Optional[np.ndarray] = None,
               avail_actions_dict: Optional[List[dict]] = None,
               rnn_hidden_actor: Optional[dict] = None,
               rnn_hidden_critic: Optional[dict] = None,
               test_mode: Optional[bool] = False,
               **kwargs):
        """
        Returns actions for agents.

        Parameters:
            obs_dict (dict): Observations for each agent in self.agent_keys.
            agent_mask (Optional[List[dict]]): Mask the agents that are alive.
            state (Optional[np.ndarray]): The global state.
            act_mean_dict (Optional[List[dict]]): Mean actions of each agent's neighbors.
            avail_actions_dict (Optional[List[dict]]): Actions mask values, default is None.
            rnn_hidden_actor (Optional[dict]): The RNN hidden states of actor representation.
            rnn_hidden_critic (Optional[dict]): The RNN hidden states of critic representation.
            test_mode (Optional[bool]): True for testing without noises.

        Returns:
            rnn_hidden_actor_new (dict): The new RNN hidden states of actor representation (if self.use_rnn=True).
            rnn_hidden_critic_new (dict): The new RNN hidden states of critic representation (if self.use_rnn=True).
            actions_dict (dict): The output actions.
            log_pi_a (dict): The log of pi.
            values_dict (dict): The evaluated critic values (when test_mode is False).
        """
        n_env = len(obs_dict)
        rnn_hidden_critic_new, values_out, log_pi_a_dict, values_dict = {}, {}, {}, {}

        mean_actions_input, agent_mask_array = self._build_inputs_mean_mask(agent_mask, act_mean_dict)
        obs_input, agents_id, avail_actions_input = self._build_inputs(obs_dict, avail_actions_dict)
        agent_mask_tensor = torch.tensor(agent_mask_array, dtype=torch.float32, device=self.device)
        rnn_hidden_actor_new, pi_dists = self.policy(observation=obs_input,
                                                     agent_ids=agents_id,
                                                     avail_actions=avail_actions_input,
                                                     rnn_hidden=rnn_hidden_actor)

        if not test_mode:
            rnn_hidden_critic_new, values_out = self.policy.get_values(observation=obs_input,
                                                                       actions_mean=mean_actions_input,
                                                                       agent_ids=agents_id,
                                                                       rnn_hidden=rnn_hidden_critic)

        if self.use_parameter_sharing:
            key = self.agent_keys[0]
            actions_sample = pi_dists[key].sample()
            actions_mean_masked = self.policy.get_mean_actions(actions={key: actions_sample},
                                                               agent_mask_tensor=agent_mask_tensor,
                                                               batch_size=n_env)
            if self.continuous_control:
                actions_out = actions_sample.reshape(n_env, self.n_agents, -1)
            else:
                actions_out = actions_sample.reshape(n_env, self.n_agents)
            actions_dict = [{k: actions_out[e, i].cpu().detach().numpy() for i, k in enumerate(self.agent_keys)}
                            for e in range(n_env)]
            if not test_mode:
                log_pi_a = pi_dists[key].log_prob(actions_sample).cpu().detach().numpy().reshape(n_env, self.n_agents)
                log_pi_a_dict = {k: log_pi_a[:, i] for i, k in enumerate(self.agent_keys)}
                values_out[key] = values_out[key].reshape(n_env, self.n_agents)
                values_dict = {k: values_out[key][:, i].cpu().detach().numpy() for i, k in enumerate(self.agent_keys)}
        else:
            actions_sample = {k: pi_dists[k].sample() for k in self.agent_keys}
            if self.continuous_control:
                actions_dict = [{k: actions_sample[k].cpu().detach().numpy()[e].reshape([-1]) for k in self.agent_keys}
                                for e in range(n_env)]
            else:
                actions_dict = [{k: actions_sample[k].cpu().detach().numpy()[e].reshape([]) for k in self.agent_keys}
                                for e in range(n_env)]
            actions_mean_masked = self.policy.get_mean_actions(actions=actions_sample,
                                                               agent_mask_tensor=agent_mask_tensor,
                                                               batch_size=n_env)
            if not test_mode:
                log_pi_a = {k: pi_dists[k].log_prob(actions_sample[k]).cpu().detach().numpy() for k in self.agent_keys}
                log_pi_a_dict = {k: log_pi_a[k].reshape([n_env]) for i, k in enumerate(self.agent_keys)}
                values_dict = {k: values_out[k].cpu().detach().numpy().reshape([n_env]) for k in self.agent_keys}

        actions_mean_masked = actions_mean_masked.cpu().detach().numpy()
        actions_mean_dict = [{k: actions_mean_masked[e, i] for i, k in enumerate(self.agent_keys)}
                             for e in range(n_env)]

        return {"rnn_hidden_actor": rnn_hidden_actor_new, "rnn_hidden_critic": rnn_hidden_critic_new,
                "actions": actions_dict, "actions_mean": actions_mean_dict,
                "log_pi": log_pi_a_dict, "values": values_dict}

    def values_next(self,
                    i_env: int,
                    obs_dict: dict,
                    state: Optional[np.ndarray] = None,
                    agent_mask: dict = None,
                    act_mean_dict: dict = None,
                    rnn_hidden_critic: Optional[dict] = None):
        """
        Returns critic values of one environment that finished an episode.

        Parameters:
            i_env (int): The index of environment.
            obs_dict (dict): Observations for each agent in self.agent_keys.
            state (Optional[np.ndarray]): The global state.
            agent_mask (dict): Mask the agents that are alive.
            act_mean_dict (dict): The mean actions of each agent's neighbors.
            rnn_hidden_critic (Optional[dict]): The RNN hidden states of critic representation.

        Returns:
            rnn_hidden_critic_new (dict): The new RNN hidden states of critic representation (if self.use_rnn=True).
            values_dict: The critic values.
        """
        n_env = 1
        rnn_hidden_critic_i = None
        if self.use_parameter_sharing:
            key = self.agent_keys[0]
            if self.use_rnn:
                hidden_item_index = np.arange(i_env * self.n_agents, (i_env + 1) * self.n_agents)
                rnn_hidden_critic_i = {key: self.policy.critic_representation[key].get_hidden_item(
                    hidden_item_index, *rnn_hidden_critic[key])}
                batch_size = n_env * self.n_agents
                obs_array = np.array(itemgetter(*self.agent_keys)(obs_dict))
                obs_input = {key: obs_array.reshape([batch_size, 1, -1])}
                action_mean_array = np.array(itemgetter(*self.agent_keys)(act_mean_dict))
                action_mean_input = {key: action_mean_array.reshape([batch_size, 1, -1])}
                agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(n_env, -1, -1).reshape(batch_size, 1, -1).to(
                    self.device)
            else:
                obs_input = {key: np.array([itemgetter(*self.agent_keys)(obs_dict)])}
                action_mean_input = {key: np.array([itemgetter(*self.agent_keys)(act_mean_dict)])}
                agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(n_env, -1, -1).to(self.device)

            rnn_hidden_critic_new, values_out = self.policy.get_values(observation=obs_input,
                                                                       actions_mean=action_mean_input,
                                                                       agent_ids=agents_id,
                                                                       rnn_hidden=rnn_hidden_critic_i)
            values_out = values_out[key].reshape(self.n_agents)
            values_dict = {k: values_out[i].cpu().detach().numpy() for i, k in enumerate(self.agent_keys)}

        else:
            if self.use_rnn:
                rnn_hidden_critic_i = {k: self.policy.critic_representation[k].get_hidden_item(
                    [i_env, ], *rnn_hidden_critic[k]) for k in self.agent_keys}
            obs_input = {k: obs_dict[k][None, :] for k in self.agent_keys} if self.use_rnn else obs_dict
            action_mean_input = {k: act_mean_dict[k][None, :]
                                 for k in self.agent_keys} if self.use_rnn else act_mean_dict

            rnn_hidden_critic_new, values_out = self.policy.get_values(observation=obs_input,
                                                                       actions_mean=action_mean_input,
                                                                       rnn_hidden=rnn_hidden_critic_i)
            values_dict = {k: values_out[k].cpu().detach().numpy().reshape([]) for k in self.agent_keys}

        return rnn_hidden_critic_new, values_dict

    def train(self, n_steps):
        """
        Train the model for numerous steps.

        Parameters:
            n_steps (int): The number of steps to train the model.
        """
        train_info = {}
        if self.use_rnn:
            with tqdm(total=n_steps) as process_bar:
                step_start, step_last = deepcopy(self.current_step), deepcopy(self.current_step)
                n_steps_all = n_steps * self.n_envs
                while step_last - step_start < n_steps_all:
                    self.run_episodes(None, n_episodes=self.n_envs, test_mode=False)
                    update_info = self.train_epochs(n_epochs=self.n_epochs)
                    self.log_infos(update_info, self.current_step)
                    train_info.update(update_info)

                    self.callback.on_train_epochs_end(self.current_step, policy=self.policy, memory=self.memory,
                                                      current_episode=self.current_episode, n_steps=n_steps,
                                                      update_info=update_info)

                    process_bar.update((self.current_step - step_last) // self.n_envs)
                    step_last = deepcopy(self.current_step)
                process_bar.update(n_steps - process_bar.last_print_n)
                self.callback.on_train_step_end(self.current_step, envs=self.envs, policy=self.policy,
                                                n_steps=n_steps, train_info=train_info)
            return train_info

        obs_dict = self.envs.buf_obs
        agent_mask_dict = [data['agent_mask'] for data in self.envs.buf_info]
        actions_mean_dict = self.actions_mean
        avail_actions = self.envs.buf_avail_actions if self.use_actions_mask else None
        state = self.envs.buf_state if self.use_global_state else None
        for _ in tqdm(range(n_steps)):
            policy_out = self.action(obs_dict=obs_dict, state=state,
                                     agent_mask=agent_mask_dict, act_mean_dict=actions_mean_dict,
                                     avail_actions_dict=avail_actions, test_mode=False)
            actions_dict, log_pi_a_dict = policy_out['actions'], policy_out['log_pi']
            actions_mean_next_dict = policy_out['actions_mean']
            values_dict = policy_out['values']
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = self.envs.step(actions_dict)
            next_state = self.envs.buf_state if self.use_global_state else None
            next_avail_actions = self.envs.buf_avail_actions if self.use_actions_mask else None

            self.callback.on_train_step(self.current_step, envs=self.envs, policy=self.policy,
                                        obs=obs_dict, policy_out=policy_out, acts=actions_dict, next_obs=next_obs_dict,
                                        rewards=rewards_dict, state=state, next_state=next_state,
                                        avail_actions=avail_actions, next_avail_actions=next_avail_actions,
                                        actions_mean_dict=actions_mean_dict,
                                        terminals=terminated_dict, truncations=truncated, infos=info,
                                        n_steps=n_steps, values_dict=values_dict)

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
                    self.memory.finish_path(i_env=i, value_next=value_next,
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
                    self.memory.finish_path(i_env=i, value_next=value_next,
                                            value_normalizer=self.learner.value_normalizer)
                    obs_dict[i] = info[i]["reset_obs"]
                    self.envs.buf_obs[i] = info[i]["reset_obs"]
                    if self.use_actions_mask:
                        avail_actions[i] = info[i]["reset_avail_actions"]
                        self.envs.buf_avail_actions[i] = info[i]["reset_avail_actions"]
                    if self.use_global_state:
                        state[i] = info[i]["reset_state"]
                        self.envs.buf_state[i] = info[i]["reset_state"]
                    self.envs.buf_info[i]["agent_mask"] = {k: True for k in self.agent_keys}
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
                    self.callback.on_train_episode_info(envs=self.envs, policy=self.policy, env_id=i,
                                                        infos=info, rank=self.rank, use_wandb=self.use_wandb,
                                                        current_step=self.current_step,
                                                        current_episode=self.current_episode,
                                                        n_steps=n_steps)

            self.current_step += self.n_envs
            self.actions_mean = deepcopy(actions_mean_dict)
            self.callback.on_train_step_end(self.current_step, envs=self.envs, policy=self.policy,
                                            n_steps=n_steps, train_info=train_info)
        return train_info

    def run_episodes(self, env_fn=None, n_episodes: int = 1, test_mode: bool = False):
        """
        Run some episodes when use RNN.

        Parameters:
            env_fn: The function that can make some testing environments.
            n_episodes (int): Number of episodes.
            test_mode (bool): Whether to test the model.

        Returns:
            Scores: The episode scores.
        """
        envs = self.envs if env_fn is None else env_fn()
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
        rnn_hidden_actor, rnn_hidden_critic = self.init_rnn_hidden(num_envs)

        while current_episode < n_episodes:
            policy_out = self.action(obs_dict=obs_dict, state=state, avail_actions_dict=avail_actions,
                                     agent_mask=agent_mask_dict, act_mean_dict=actions_mean_dict,
                                     rnn_hidden_actor=rnn_hidden_actor, rnn_hidden_critic=rnn_hidden_critic,
                                     test_mode=test_mode)
            rnn_hidden_actor, rnn_hidden_critic = policy_out['rnn_hidden_actor'], policy_out['rnn_hidden_critic']
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

            self.callback.on_test_step(envs=envs, policy=self.policy, images=images, test_mode=test_mode,
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
                                                             state=None if state is None else state[i],
                                                             act_mean_dict=actions_mean_dict[i],
                                                             agent_mask=agent_mask_dict[i],
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
                        self.callback.on_train_episode_info(envs=self.envs, policy=self.policy, env_id=i,
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

            if env_fn is not None:
                envs.close()
        return scores
