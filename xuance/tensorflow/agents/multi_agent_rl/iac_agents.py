import numpy as np
from argparse import Namespace
from operator import itemgetter
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.common import List, Optional, Union
from xuance.tensorflow import Module, tf
from xuance.tensorflow.utils import NormalizeFunctions, InitializeFunctions, ActivationFunctions
from xuance.tensorflow.policies import REGISTRY_Policy
from xuance.tensorflow.agents import OnPolicyMARLAgents, BaseCallback


class IAC_Agents(OnPolicyMARLAgents):
    """The implementation of IAC agents.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """

    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv],
                 callback: Optional[BaseCallback] = None):
        super(IAC_Agents, self).__init__(config, envs, callback)
        self.policy = self._build_policy()  # build policy
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, self.policy, self.callback)

    def _build_policy(self) -> Module:
        """
        Build representation(s) and policy(ies) for agent(s)

        Returns:
            policy (Module): A dict of policies.
        """
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = InitializeFunctions[self.config.initialize] if hasattr(self.config, "initialize") else None
        activation = ActivationFunctions[self.config.activation]
        agent = self.config.agent

        # build representations
        A_representation = self._build_representation(self.config.representation, self.observation_space, self.config)
        C_representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policies
        if self.config.policy == "Categorical_MAAC_Policy":
            policy = REGISTRY_Policy["Categorical_MAAC_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                representation_actor=A_representation, representation_critic=C_representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
        elif self.config.policy == "Gaussian_MAAC_Policy":
            policy = REGISTRY_Policy["Gaussian_MAAC_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                representation_actor=A_representation, representation_critic=C_representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                activation_action=ActivationFunctions[self.config.activation_action],
                use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
        else:
            raise AttributeError(f"{agent} currently does not support the policy named {self.config.policy}.")
        return policy

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
        rnn_hidden_critic_new, log_pi_a_dict, values_dict = {}, {}, {}

        obs_input, agents_id, avail_actions_input = self._build_inputs(obs_dict, avail_actions_dict)
        rnn_hidden_actor_new, pi_logits = self.policy(observation=obs_input,
                                                      agent_ids=agents_id,
                                                      avail_actions=avail_actions_input,
                                                      rnn_hidden=rnn_hidden_actor)
        if not test_mode:
            rnn_hidden_critic_new, values_dict = self.policy.get_values(observation=obs_input,
                                                                        agent_ids=agents_id,
                                                                        rnn_hidden=rnn_hidden_critic)
            if self.use_parameter_sharing:
                values_dict = {k: values_dict[self.model_keys[0]].numpy().reshape(n_env, -1)[:, i]
                               for i, k in enumerate(self.agent_keys)}
            else:
                values_dict = {k: v.numpy().reshape(n_env) for k, v in values_dict.items()}

        if self.use_parameter_sharing:
            key = self.agent_keys[0]
            pi_dists = self.policy.actor[key].distribution(logits=pi_logits[key])
            actions_sample = pi_dists.stochastic_sample()
            if self.continuous_control:
                actions_out = actions_sample.numpy().reshape(n_env, self.n_agents, -1)
            else:
                actions_out = actions_sample.numpy().reshape(n_env, self.n_agents)
            actions_dict = [{k: actions_out[e, i] for i, k in enumerate(self.agent_keys)} for e in range(n_env)]
        else:
            pi_dists = {k: self.policy.actor[k].distribution(logits=pi_logits[k]) for k in self.agent_keys}
            actions_sample = {k: pi_dists[k].stochastic_sample() for k in self.agent_keys}
            if self.continuous_control:
                actions_dict = [{k: actions_sample[k].numpy()[e].reshape([-1]) for k in self.agent_keys}
                                for e in range(n_env)]
            else:
                actions_dict = [{k: actions_sample[k].numpy()[e].reshape([]) for k in self.agent_keys}
                                for e in range(n_env)]

        return {"rnn_hidden_actor": rnn_hidden_actor_new, "rnn_hidden_critic": rnn_hidden_critic_new,
                "actions": actions_dict, "log_pi": None, "values": values_dict}

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

        obs_input, agents_id, avail_actions_input = self._build_inputs([obs_dict])
        rnn_hidden_critic_new, values_dict = self.policy.get_values(observation=obs_input,
                                                                    agent_ids=agents_id,
                                                                    rnn_hidden=rnn_hidden_critic)
        if self.use_parameter_sharing:
            values_dict = {k: values_dict[self.model_keys[0]].numpy().reshape(-1)[i]
                           for i, k in enumerate(self.agent_keys)}
        else:
            values_dict = {k: v.numpy().reshape([]) for k, v in values_dict.items()}
        return rnn_hidden_critic_new, values_dict
