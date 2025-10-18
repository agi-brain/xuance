from argparse import Namespace
from xuance.common import Union, Optional, BaseCallback
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.tensorflow import Module
from xuance.tensorflow.utils import NormalizeFunctions, ActivationFunctions, InitializeFunctions
from xuance.tensorflow.policies import REGISTRY_Policy
from xuance.tensorflow.agents.policy_gradient.ddpg_agent import DDPG_Agent


class TD3_Agent(DDPG_Agent):
    """The implementation of TD3 agent.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """
    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecEnv, SubprocVecEnv],
                 callback: Optional[BaseCallback] = None):
        super(TD3_Agent, self).__init__(config, envs, callback)

    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = InitializeFunctions[self.config.initialize] if hasattr(self.config, "initialize") else None
        activation = ActivationFunctions[self.config.activation]

        # build representations.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policy
        if self.config.policy == "TD3_Policy":
            policy = REGISTRY_Policy["TD3_Policy"](
                action_space=self.action_space, representation=representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer,
                activation=activation, activation_action=ActivationFunctions[self.config.activation_action],
                use_distributed_training=self.distributed_training)
        else:
            raise AttributeError(f"TD3 currently does not support the policy named {self.config.policy}.")

        return policy
