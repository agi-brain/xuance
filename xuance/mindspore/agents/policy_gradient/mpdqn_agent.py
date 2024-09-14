from argparse import Namespace
from xuance.environment.single_agent_env import Gym_Env
from xuance.mindspore import Module
from xuance.mindspore.utils import NormalizeFunctions, ActivationFunctions, InitializeFunctions
from xuance.mindspore.policies import REGISTRY_Policy
from xuance.mindspore.agents.policy_gradient.pdqn_agent import PDQN_Agent


class MPDQN_Agent(PDQN_Agent):
    """The implementation of MPDQN agent.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """
    def __init__(self,
                 config: Namespace,
                 envs: Gym_Env):
        super(MPDQN_Agent, self).__init__(config, envs)

    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = InitializeFunctions[self.config.initialize] if hasattr(self.config, "initialize") else None
        activation = ActivationFunctions[self.config.activation]

        # build representation.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policy.
        if self.config.policy == "MPDQN_Policy":
            policy = REGISTRY_Policy["MPDQN_Policy"](
                observation_space=self.observation_space, action_space=self.action_space,
                representation=representation,
                conactor_hidden_size=self.config.conactor_hidden_size,
                qnetwork_hidden_size=self.config.qnetwork_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                activation_action=ActivationFunctions[self.config.activation_action])
        else:
            raise AttributeError(
                f"{self.config.agent} currently does not support the policy named {self.config.policy}.")

        return policy

