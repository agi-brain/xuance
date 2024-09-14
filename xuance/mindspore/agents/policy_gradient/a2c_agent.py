from argparse import Namespace
from xuance.environment import DummyVecEnv
from xuance.mindspore import Module
from xuance.mindspore.utils import NormalizeFunctions, ActivationFunctions, InitializeFunctions
from xuance.mindspore.policies import REGISTRY_Policy
from xuance.mindspore.agents import OnPolicyAgent


class A2C_Agent(OnPolicyAgent):
    """The implementation of A2C agent.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv):
        super(A2C_Agent, self).__init__(config, envs)
        self.memory = self._build_memory()  # build memory
        self.policy = self._build_policy()  # build policy
        self.learner = self._build_learner(self.config, self.policy)  # build learner

    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = InitializeFunctions[self.config.initialize] if hasattr(self.config, "initialize") else None
        activation = ActivationFunctions[self.config.activation]

        # build representation.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policy.
        if self.config.policy == "Categorical_AC":
            policy = REGISTRY_Policy["Categorical_AC"](
                action_space=self.action_space, representation=representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation)
        elif self.config.policy == "Gaussian_AC":
            policy = REGISTRY_Policy["Gaussian_AC"](
                action_space=self.action_space, representation=representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                activation_action=ActivationFunctions[self.config.activation_action])
        else:
            raise AttributeError(f"A2C currently does not support the policy named {self.config.policy}.")

        return policy
