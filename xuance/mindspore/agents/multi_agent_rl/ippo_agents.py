import numpy as np
from argparse import Namespace
from xuance.common import Optional
from xuance.environment import DummyVecMultiAgentEnv
from xuance.mindspore import Module
from xuance.mindspore.utils import NormalizeFunctions, ActivationFunctions, InitializeFunctions
from xuance.mindspore.policies import REGISTRY_Policy
from xuance.mindspore.agents import OnPolicyMARLAgents


class IPPO_Agents(OnPolicyMARLAgents):
    """The implementation of Independent PPO agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMultiAgentEnv):
        super(IPPO_Agents, self).__init__(config, envs)

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
            raise AttributeError(f"{agent} currently does not support the policy named {self.config.policy}.")
        return policy

    def init_rnn_hidden(self, n_envs):
        """
        Returns initialized hidden states of RNN if use RNN-based representations.

        Parameters:
            n_envs (int): The number of parallel environments.
        """
        rnn_hidden_actor, rnn_hidden_critic = None, None
        if self.use_rnn:
            batch = n_envs * self.n_agents if self.use_parameter_sharing else n_envs
            rnn_hidden_actor = {k: self.policy.actor_representation[k].init_hidden(batch) for k in self.model_keys}
            rnn_hidden_critic = {k: self.policy.critic_representation[k].init_hidden(batch) for k in self.model_keys}
        return rnn_hidden_actor, rnn_hidden_critic

    def init_hidden_item(self,
                         i_env: int,
                         rnn_hidden_actor: Optional[dict] = None,
                         rnn_hidden_critic: Optional[dict] = None):
        """
        Returns initialized hidden states of RNN for i-th environment.

        Parameters:
            i_env (int): The index of environment that to be selected.
            rnn_hidden_actor (Optional[dict]): The RNN hidden states of actor representation.
            rnn_hidden_critic (Optional[dict]): The RNN hidden states of critic representation.
        """
        assert self.use_rnn is True, "This method cannot be called when self.use_rnn is False."
        if self.use_parameter_sharing:
            b_index = np.arange(i_env * self.n_agents, (i_env + 1) * self.n_agents)
        else:
            b_index = [i_env, ]
        for k in self.model_keys:
            rnn_hidden_actor[k] = self.policy.actor_representation[k].init_hidden_item(b_index, *rnn_hidden_actor[k])
        if rnn_hidden_critic is None:
            return rnn_hidden_actor, None
        for k in self.model_keys:
            rnn_hidden_critic[k] = self.policy.critic_representation[k].init_hidden_item(b_index, *rnn_hidden_critic[k])
        return rnn_hidden_actor, rnn_hidden_critic
