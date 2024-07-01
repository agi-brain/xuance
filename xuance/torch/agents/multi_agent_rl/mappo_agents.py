import torch
import torch.nn as nn
import numpy as np
from argparse import Namespace
from operator import itemgetter
from typing import Optional, List
from xuance.environment import DummyVecMultiAgentEnv
from xuance.torch import ModuleDict
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.representations import REGISTRY_Representation
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.learners import MAPPO_Clip_Learner
from xuance.torch.agents.multi_agent_rl.ippo_agents import IPPO_Agents


class MAPPO_Agents(IPPO_Agents):
    """The implementation of MAPPO agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMultiAgentEnv):
        super(MAPPO_Agents, self).__init__(config, envs)

    def _build_critic_representation(self, representation_key: str, config: Namespace):
        """
        Build representation for critics in MAPPO.

        Parameters:
            representation_key (str): The selection of representation, e.g., "Basic_MLP", "Basic_RNN", etc.
            config: The configurations for creating the representation module.
        
        Returns:
            representation (Module): The representation Module. 
        """
        normalize_fn = NormalizeFunctions[config.normalize] if hasattr(config, "normalize") else None
        initializer = nn.init.orthogonal_
        activation = ActivationFunctions[config.activation]
        device = self.device
        agent = config.agent
        # build representations
        representation = ModuleDict()
        dim_obs_all = sum([self.observation_space[k].shape[-1] for k in self.agent_keys])
        if self.use_global_state:
            dim_obs_all = self.state_space.shape[-1]
        input_shape = (dim_obs_all,)
        for key in self.model_keys:
            if representation_key == "Basic_Identical":
                representation[key] = REGISTRY_Representation["Basic_Identical"](input_shape=input_shape,
                                                                                 device=self.device)
            elif representation_key == "Basic_MLP":
                representation[key] = REGISTRY_Representation["Basic_MLP"](
                    input_shape=input_shape, hidden_sizes=self.config.representation_hidden_size,
                    normalize=normalize_fn, initialize=initializer, activation=activation, device=device)
            elif representation_key == "Basic_RNN":
                representation[key] = REGISTRY_Representation["Basic_RNN"](
                    input_shape=input_shape,
                    hidden_sizes={'fc_hidden_sizes': self.config.fc_hidden_sizes,
                                  'recurrent_hidden_size': self.config.recurrent_hidden_size},
                    normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
                    N_recurrent_layers=self.config.N_recurrent_layers,
                    dropout=self.config.dropout, rnn=self.config.rnn)
            else:
                raise AttributeError(f"{agent} currently does not support {representation_key} representation.")
        return representation

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
        # build representations
        representation_actor = self._build_representation(self.config.representation, self.config)
        representation_critic = self._build_critic_representation(self.config.representation, self.config)
        # build policies
        if self.config.policy == "Categorical_MAAC_Policy":
            policy = REGISTRY_Policy["Categorical_MAAC_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                representation_actor=representation_actor, representation_critic=representation_critic,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                device=device, use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
            self.continuous_control = False
        elif self.config.policy == "Gaussian_MAAC_Policy":
            policy = REGISTRY_Policy["Gaussian_MAAC_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                representation_actor=representation_actor, representation_critic=representation_critic,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                activation_action=ActivationFunctions[self.config.activation_action],
                device=device, use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
            self.continuous_control = True
        else:
            raise AttributeError(f"MAPPO currently does not support the policy named {self.config.policy}.")
        return policy

    def _build_learner(self, config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler):
        return MAPPO_Clip_Learner(config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler)

    def _build_critic_inputs(self, batch_size: int, obs_batch: dict,
                             state: Optional[np.ndarray]):
        """
        Build inputs for critic representations before calculating actions.

        Parameters:
            batch_size (int): The size of the obs batch.
            obs_batch (dict): Observations for each agent in self.agent_keys.
            state (Optional[np.ndarray]): The global state.

        Returns:
            critic_input: The represented observations.
        """
        if self.use_global_state:
            critic_input = state
        else:
            if self.use_parameter_sharing:
                key = self.model_keys[0]
                bs = batch_size * self.n_agents
                joint_obs = obs_batch[key].reshape([batch_size, self.n_agents, -1]).reshape([batch_size, 1, -1])
                joint_obs = np.repeat(joint_obs, repeats=self.n_agents, axis=1)
            else:
                bs = batch_size
                joint_obs = np.stack(itemgetter(*self.agent_keys)(obs_batch), axis=1)
            joint_obs = joint_obs.reshape([bs, 1, -1]) if self.use_rnn else joint_obs.reshape([bs, -1])
            critic_input = {k: joint_obs for k in self.model_keys}
        return critic_input

    def action(self,
               obs_dict: List[dict],
               state: Optional[np.ndarray] = None,
               avail_actions_dict: Optional[List[dict]] = None,
               rnn_hidden_actor: Optional[dict] = None,
               rnn_hidden_critic: Optional[dict] = None,
               test_mode: Optional[bool] = False):
        """
        Returns actions for agents.

        Parameters:
            obs_dict (dict): Observations for each agent in self.agent_keys.
            state (Optional[np.ndarray]): The global state.
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

        obs_input, agents_id, avail_actions_input = self._build_inputs(obs_dict, avail_actions_dict)
        rnn_hidden_actor_new, pi_dists = self.policy(observation=obs_input,
                                                     agent_ids=agents_id,
                                                     avail_actions=avail_actions_input,
                                                     rnn_hidden=rnn_hidden_actor)
        if not test_mode:
            critic_input = self._build_critic_inputs(batch_size=n_env, obs_batch=obs_input, state=state)
            rnn_hidden_critic_new, values_out = self.policy.get_values(observation=critic_input,
                                                                       agent_ids=agents_id,
                                                                       rnn_hidden=rnn_hidden_critic)

        if self.use_parameter_sharing:
            key = self.agent_keys[0]
            actions_sample = pi_dists[key].stochastic_sample()
            if self.continuous_control:
                actions_out = actions_sample.reshape(n_env, self.n_agents, -1)
            else:
                actions_out = actions_sample.reshape(n_env, self.n_agents)
            actions_dict = [{k: actions_out[e, i].cpu().detach().numpy() for i, k in enumerate(self.agent_keys)}
                            for e in range(n_env)]
            if not test_mode:
                log_pi_a = pi_dists[key].log_prob(actions_sample).cpu().detach().numpy()
                log_pi_a = log_pi_a.reshape(n_env, self.n_agents)
                log_pi_a_dict = {k: log_pi_a[:, i] for i, k in enumerate(self.agent_keys)}
                values_out[key] = values_out[key].reshape(n_env, self.n_agents)
                values_dict = {k: values_out[key][:, i].cpu().detach().numpy() for i, k in enumerate(self.agent_keys)}
        else:
            actions_sample = {k: pi_dists[k].stochastic_sample() for k in self.agent_keys}
            if self.continuous_control:
                actions_dict = [{k: actions_sample[k].cpu().detach().numpy()[e].reshape([-1]) for k in self.agent_keys}
                                for e in range(n_env)]
            else:
                actions_dict = [{k: actions_sample[k].cpu().detach().numpy()[e].reshape([]) for k in self.agent_keys}
                                for e in range(n_env)]
            if not test_mode:
                log_pi_a = {k: pi_dists[k].log_prob(actions_sample[k]).cpu().detach().numpy() for k in self.agent_keys}
                log_pi_a_dict = {k: log_pi_a[k].reshape([n_env]) for i, k in enumerate(self.agent_keys)}
                values_dict = {k: values_out[k].cpu().detach().numpy().reshape([n_env]) for k in self.agent_keys}

        return rnn_hidden_actor_new, rnn_hidden_critic_new, actions_dict, log_pi_a_dict, values_dict

    def values_next(self,
                    i_env: int,
                    obs_dict: dict,
                    state: Optional[np.ndarray] = None,
                    rnn_hidden_critic: Optional[dict] = None):
        """
        Returns critic values of one environment that finished an episode.

        Parameters:
            i_env (int): The index of environment.
            obs_dict (dict): Observations for each agent in self.agent_keys.
            state (Optional[np.ndarray]): The global state.
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
                if self.use_global_state:
                    critic_input = np.repeat(state.reshape([n_env, 1, -1]),
                                             self.n_agents, axis=1).reshape([batch_size, 1, -1])
                else:
                    obs_array = np.array(itemgetter(*self.agent_keys)(obs_dict))
                    critic_input = np.repeat(obs_array.reshape([n_env, 1, -1]),
                                             self.n_agents, axis=1).reshape([batch_size, 1, -1])
                agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(n_env, -1, -1).reshape(batch_size, 1, -1).to(
                    self.device)
            else:
                if self.use_global_state:
                    critic_input = np.repeat(state.reshape([n_env, 1, -1]), self.n_agents, axis=1)
                else:
                    obs_array = np.array([itemgetter(*self.agent_keys)(obs_dict)]).reshape([n_env, 1, -1])
                    critic_input = np.repeat(obs_array, self.n_agents, axis=1)
                agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(n_env, -1, -1).to(self.device)

            rnn_hidden_critic_new, values_out = self.policy.get_values(observation={key: critic_input},
                                                                       agent_ids=agents_id,
                                                                       rnn_hidden=rnn_hidden_critic_i)
            values_out = values_out[key].reshape(self.n_agents)
            values_dict = {k: values_out[i].cpu().detach().numpy() for i, k in enumerate(self.agent_keys)}

        else:
            if self.use_rnn:
                rnn_hidden_critic_i = {k: self.policy.critic_representation[k].get_hidden_item(
                    [i_env, ], *rnn_hidden_critic[k]) for k in self.agent_keys}
                joint_obs = np.stack(itemgetter(*self.agent_keys)(obs_dict), axis=0).reshape([n_env, 1, -1])
                critic_input = {k: joint_obs for k in self.agent_keys}
            else:
                critic_input_array = np.concatenate([obs_dict[k].reshape(n_env, 1, -1) for k in self.agent_keys],
                                                    axis=1).reshape(n_env, -1)
                if self.use_global_state:
                    critic_input_array = np.concatenate([critic_input_array, state], axis=-1)
                critic_input = {k: critic_input_array for k in self.agent_keys}

            rnn_hidden_critic_new, values_out = self.policy.get_values(observation=critic_input,
                                                                       rnn_hidden=rnn_hidden_critic_i)
            values_dict = {k: values_out[k].cpu().detach().numpy().reshape([]) for k in self.agent_keys}

        return rnn_hidden_critic_new, values_dict
