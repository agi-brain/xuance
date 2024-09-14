import numpy as np
from argparse import Namespace
from operator import itemgetter
from xuance.common import Optional, List
from xuance.environment import DummyVecMultiAgentEnv
from xuance.tensorflow import Module
from xuance.tensorflow.utils import NormalizeFunctions, ActivationFunctions, InitializeFunctions
from xuance.tensorflow.policies import REGISTRY_Policy
from xuance.tensorflow.agents.multi_agent_rl.ippo_agents import IPPO_Agents


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

    def _build_policy(self) -> Module:
        """
        Build representation(s) and policy(ies) for agent(s)

        Returns:
            policy (torch.nn.Module): A dict of policies.
        """
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = InitializeFunctions[self.config.initialize] if hasattr(self.config, "initialize") else None
        activation = ActivationFunctions[self.config.activation]
        # build representations
        A_representation = self._build_representation(self.config.representation, self.observation_space, self.config)
        if self.use_global_state:
            dim_obs_all = sum(self.state_space.shape)
        else:
            dim_obs_all = sum([sum(self.observation_space[k].shape) for k in self.agent_keys])
        space_critic_in = {k: (dim_obs_all,) for k in self.agent_keys}
        C_representation = self._build_representation(self.config.representation, space_critic_in, self.config)
        # build policies
        if self.config.policy == "Categorical_MAAC_Policy":
            policy = REGISTRY_Policy["Categorical_MAAC_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                representation_actor=A_representation, representation_critic=C_representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
            self.continuous_control = False
        elif self.config.policy == "Gaussian_MAAC_Policy":
            policy = REGISTRY_Policy["Gaussian_MAAC_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                representation_actor=A_representation, representation_critic=C_representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                activation_action=ActivationFunctions[self.config.activation_action],
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
            self.continuous_control = True
        else:
            raise AttributeError(f"MAPPO currently does not support the policy named {self.config.policy}.")
        return policy

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
            if self.use_parameter_sharing:
                key = self.model_keys[0]
                bs = batch_size * self.n_agents
                state_n = np.stack([state for _ in range(self.n_agents)], axis=1).reshape([bs, -1])
                critic_input = {key: state_n}
            else:
                critic_input = {k: state for k in self.model_keys}
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
        rnn_hidden_actor_new, _ = self.policy(observation=obs_input,
                                              agent_ids=agents_id,
                                              avail_actions=avail_actions_input,
                                              rnn_hidden=rnn_hidden_actor)
        pi_dists = {key: self.policy.actor[key].dist for key in self.model_keys}
        if not test_mode:
            critic_input = self._build_critic_inputs(batch_size=n_env, obs_batch=obs_input, state=state)
            rnn_hidden_critic_new, values_out = self.policy.get_values(observation=critic_input,
                                                                       agent_ids=agents_id,
                                                                       rnn_hidden=rnn_hidden_critic)

        if self.use_parameter_sharing:
            key = self.agent_keys[0]
            actions_sample = pi_dists[key].stochastic_sample().numpy()
            if self.continuous_control:
                actions_out = actions_sample.reshape(n_env, self.n_agents, -1)
            else:
                actions_out = actions_sample.reshape(n_env, self.n_agents)
            actions_dict = [{k: actions_out[e, i] for i, k in enumerate(self.agent_keys)} for e in range(n_env)]
            if not test_mode:
                log_pi_a = pi_dists[key].log_prob(actions_sample).numpy()
                log_pi_a = log_pi_a.reshape(n_env, self.n_agents)
                log_pi_a_dict = {k: log_pi_a[:, i] for i, k in enumerate(self.agent_keys)}
                values_out[key] = values_out[key].numpy().reshape(n_env, self.n_agents)
                values_dict = {k: values_out[key][:, i] for i, k in enumerate(self.agent_keys)}
        else:
            actions_sample = {k: pi_dists[k].stochastic_sample().numpy() for k in self.agent_keys}
            if self.continuous_control:
                actions_dict = [{k: actions_sample[k][e].reshape([-1]) for k in self.agent_keys} for e in range(n_env)]
            else:
                actions_dict = [{k: actions_sample[k][e].reshape([]) for k in self.agent_keys} for e in range(n_env)]
            if not test_mode:
                log_pi_a = {k: pi_dists[k].log_prob(actions_sample[k]).numpy() for k in self.agent_keys}
                log_pi_a_dict = {k: log_pi_a[k].reshape([n_env]) for i, k in enumerate(self.agent_keys)}
                values_dict = {k: values_out[k].numpy().reshape([n_env]) for k in self.agent_keys}

        return {"rnn_hidden_actor": rnn_hidden_actor_new, "rnn_hidden_critic": rnn_hidden_critic_new,
                "actions": actions_dict, "log_pi": log_pi_a_dict, "values": values_dict}

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
            batch_size = n_env * self.n_agents
            if self.use_rnn:
                hidden_item_index = np.arange(i_env * self.n_agents, (i_env + 1) * self.n_agents)
                rnn_hidden_critic_i = {key: self.policy.critic_representation[key].get_hidden_item(
                    hidden_item_index, *rnn_hidden_critic[key])}
                if self.use_global_state:
                    critic_input = np.repeat(state.reshape([n_env, 1, -1]),
                                             self.n_agents, axis=1).reshape([batch_size, 1, -1])
                else:
                    obs_array = np.array(itemgetter(*self.agent_keys)(obs_dict))
                    critic_input = np.repeat(obs_array.reshape([n_env, 1, -1]),
                                             self.n_agents, axis=1).reshape([batch_size, 1, -1])
                agents_id = np.eye(self.n_agents, dtype=np.float32)[None].repeat(n_env, 0).reshape(batch_size, -1)
            else:
                if self.use_global_state:
                    critic_input = np.repeat(state.reshape([batch_size, -1]), self.n_agents, axis=1)
                else:
                    obs_array = np.array([itemgetter(*self.agent_keys)(obs_dict)]).reshape([batch_size, -1])
                    critic_input = np.repeat(obs_array, self.n_agents, axis=1).reshape(batch_size, -1)
                agents_id = np.eye(self.n_agents, dtype=np.float32)[None].repeat(n_env, 0).reshape(batch_size, -1)

            rnn_hidden_critic_new, values_out = self.policy.get_values(observation={key: critic_input},
                                                                       agent_ids=agents_id,
                                                                       rnn_hidden=rnn_hidden_critic_i)
            values_out = values_out[key].numpy().reshape(self.n_agents)
            values_dict = {k: values_out[i] for i, k in enumerate(self.agent_keys)}

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
                    critic_input_array = state.reshape([n_env, -1])
                critic_input = {k: critic_input_array for k in self.agent_keys}

            rnn_hidden_critic_new, values_out = self.policy.get_values(observation=critic_input,
                                                                       rnn_hidden=rnn_hidden_critic_i)
            values_dict = {k: values_out[k].numpy().reshape([]) for k in self.agent_keys}

        return rnn_hidden_critic_new, values_dict
