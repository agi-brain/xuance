from argparse import Namespace
from xuance.common import Union, Optional, MultiAgentBaseCallback
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.tensorflow import Module
from xuance.tensorflow.utils import NormalizeFunctions, ActivationFunctions, InitializeFunctions
from xuance.tensorflow.policies import REGISTRY_Policy
from xuance.tensorflow.agents import OffPolicyMARLAgents


class IQL_Agents(OffPolicyMARLAgents):
    """The implementation of Independent Q-Learning agents.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """

    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv],
                 callback: Optional[MultiAgentBaseCallback] = None):
        super(IQL_Agents, self).__init__(config, envs, callback)

        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / config.decay_step_greedy
        self.e_greedy = self.start_greedy

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

        # build representations
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policies
        if self.config.policy == "Basic_Q_network_marl":
            policy = REGISTRY_Policy["Basic_Q_network_marl"](
                action_space=self.action_space, n_agents=self.n_agents,
                representation=representation,
                hidden_size=self.config.q_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
        else:
            raise AttributeError(f"IQL currently does not support the policy named {self.config.policy}.")

        return policy
