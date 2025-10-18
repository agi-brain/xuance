import torch
from argparse import Namespace
from xuance.environment.single_agent_env import Gym_Env
from xuance.common import Optional, BaseCallback
from xuance.torch import Module
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.agents.policy_gradient.pdqn_agent import PDQN_Agent


class MPDQN_Agent(PDQN_Agent):
    """The implementation of MPDQN agent.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """
    def __init__(self,
                 config: Namespace,
                 envs: Gym_Env,
                 callback: Optional[BaseCallback] = None):
        super(MPDQN_Agent, self).__init__(config, envs, callback)

    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device

        # build representation.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policy.
        if self.config.policy == "MPDQN_Policy":
            policy = REGISTRY_Policy["MPDQN_Policy"](
                observation_space=self.observation_space, action_space=self.action_space,
                representation=representation,
                conactor_hidden_size=self.config.conactor_hidden_size,
                qnetwork_hidden_size=self.config.qnetwork_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
                activation_action=ActivationFunctions[self.config.activation_action],
                use_distributed_training=self.distributed_training)
        else:
            raise AttributeError(
                f"{self.config.agent} currently does not support the policy named {self.config.policy}.")

        return policy

