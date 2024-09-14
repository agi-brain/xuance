from argparse import Namespace
from xuance.common import Optional, List
from xuance.environment import DummyVecMultiAgentEnv
from xuance.tensorflow import Module
from xuance.tensorflow.utils import NormalizeFunctions, ActivationFunctions, InitializeFunctions
from xuance.tensorflow.policies import REGISTRY_Policy
from xuance.tensorflow.agents import OffPolicyMARLAgents


class IDDPG_Agents(OffPolicyMARLAgents):
    """The implementation of Independent DDPG agents.

    Args:
        config: The Namespace variable that provides hyper-parameters and other settings.
        envs: The vectorized environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMultiAgentEnv):
        super(IDDPG_Agents, self).__init__(config, envs)

        self.start_noise, self.end_noise = config.start_noise, config.end_noise
        self.noise_scale = config.start_noise
        self.delta_noise = (self.start_noise - self.end_noise) / config.running_steps

        # build policy, optimizers, schedulers
        self.policy = self._build_policy()  # build policy
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, self.policy)

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
        C_representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policies
        if self.config.policy == "Independent_DDPG_Policy":
            policy = REGISTRY_Policy["Independent_DDPG_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                actor_representation=A_representation, critic_representation=C_representation,
                actor_hidden_size=self.config.actor_hidden_size,
                critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                activation_action=ActivationFunctions[self.config.activation_action],
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
        else:
            raise AttributeError(f"IDDPG currently does not support the policy named {self.config.policy}.")

        return policy

    def init_rnn_hidden(self, n_envs):
        """
        Returns initialized hidden states of RNN if use RNN-based representations.

        Parameters:
            n_envs (int): The number of parallel environments.

        Returns:
            rnn_hidden_states: The hidden states for RNN.
        """
        rnn_hidden_states = None
        if self.use_rnn:
            batch = n_envs * self.n_agents if self.use_parameter_sharing else n_envs
            rnn_hidden_states = {k: self.policy.actor_representation[k].init_hidden(batch) for k in self.model_keys}
        return rnn_hidden_states

    def init_hidden_item(self, i_env: int,
                         rnn_hidden: Optional[dict] = None):
        """
        Returns initialized hidden states of RNN for i-th environment.

        Parameters:
            i_env (int): The index of environment that to be selected.
            rnn_hidden (Optional[dict]): The RNN hidden states of actor representation.
        """
        assert self.use_rnn is True, "This method cannot be called when self.use_rnn is False."
        if self.use_parameter_sharing:
            batch_index = list(range(i_env * self.n_agents, (i_env + 1) * self.n_agents))
        else:
            batch_index = [i_env, ]
        for key in self.model_keys:
            rnn_hidden[key] = self.policy.actor_representation[key].init_hidden_item(batch_index, *rnn_hidden[key])
        return rnn_hidden

    def action(self,
               obs_dict: List[dict],
               avail_actions_dict: Optional[List[dict]] = None,
               rnn_hidden: Optional[dict] = None,
               test_mode: Optional[bool] = False):
        """
        Returns actions for agents.

        Parameters:
            obs_dict (List[dict]): Observations for each agent in self.agent_keys.
            avail_actions_dict (Optional[List[dict]]): Actions mask values, default is None.
            rnn_hidden (Optional[dict]): The hidden variables of the RNN.
            test_mode (Optional[bool]): True for testing without noises.

        Returns:
            rnn_hidden_state (dict): The new hidden states for RNN (if self.use_rnn=True).
            actions_dict (dict): The output actions.
        """
        batch_size = len(obs_dict)

        obs_input, agents_id, _ = self._build_inputs(obs_dict)
        hidden_state, actions = self.policy(observation=obs_input, agent_ids=agents_id, rnn_hidden=rnn_hidden)

        if self.use_parameter_sharing:
            key = self.agent_keys[0]
            actions[key] = actions[key].numpy().reshape(batch_size, self.n_agents, -1)
            if not test_mode:
                actions = self.exploration(batch_size, actions)
            actions_dict = [{k: actions[key][e, i] for i, k in enumerate(self.agent_keys)} for e in range(batch_size)]
        else:
            for key in self.agent_keys:
                actions[key] = actions[key].numpy().reshape(batch_size, -1)
                if not test_mode:
                    actions = self.exploration(batch_size, actions)
            actions_dict = [{k: actions[k][i] for k in self.agent_keys} for i in range(batch_size)]

        return {"hidden_state": hidden_state, "actions": actions_dict}
