import torch
from argparse import Namespace
from xuance.common import Union, Optional
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch import Module
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.agents import OnPolicyAgent, BaseCallback


class NPG_Agent(OnPolicyAgent):
    """The implementation of NPG agent.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """
    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecEnv, SubprocVecEnv],
                 callback: Optional[BaseCallback] = None):
        super(NPG_Agent, self).__init__(config, envs, callback)
        self.memory = self._build_memory()  # build memory
        self.policy = self._build_policy()  # build policy
        self.learner = self._build_learner(self.config, self.policy, self.callback)  # build learner

    def _build_policy(self) -> Module:
        """Builds the policy network based on the configuration.

        This method constructs the policy network using the specified configuration parameters,
        including normalization, initialization, activation functions, and device placement.
        It supports different types of policies, such as Categorical Actor-Critic (Categorical_AC)
        and Gaussian Actor-Critic (Gaussian_AC).

        Returns:
            Module: The constructed policy network.

        Raises:
            AttributeError: If the specified policy type is not supported.
        """
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device

        # build representation.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policy.
        if self.config.policy == "Categorical_AC":
            policy = REGISTRY_Policy["Categorical_AC"](
                action_space=self.action_space, representation=representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
                use_distributed_training=self.distributed_training)
        elif self.config.policy == "Gaussian_AC":
            policy = REGISTRY_Policy["Gaussian_AC"](
                action_space=self.action_space, representation=representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
                use_distributed_training=self.distributed_training,
                activation_action=ActivationFunctions[self.config.activation_action])
        else:
            raise AttributeError(f"NPG currently does not support the policy named {self.config.policy}.")

        return policy
