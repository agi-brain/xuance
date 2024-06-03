import torch
from argparse import Namespace
from xuance.environment import DummyVecEnv
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.representations import REGISTRY_Representation
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.learners import DuelDQN_Learner
from xuance.torch.agents.qlearning_family.dqn_agent import DQN_Agent
from xuance.common import space2shape


class DuelDQN_Agent(DQN_Agent):
    """The implementation of DuelDQN agent.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv):
        super(DuelDQN_Agent, self).__init__(config, envs)

    def _build_policy(self):
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device

        # build representation.
        if self.config.representation == "Basic_Identical":
            representation = REGISTRY_Representation["Basic_Identical"](input_shape=space2shape(self.observation_space),
                                                                        device=self.device)
        elif self.config.representation == "Basic_MLP":
            representation = REGISTRY_Representation["Basic_MLP"](
                input_shape=space2shape(self.observation_space),
                hidden_sizes=self.config.representation_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device)
        elif self.config.representation == "Basic_CNN":
            representation = REGISTRY_Representation["Basic_CNN"](
                input_shape=space2shape(self.observation_space),
                kernels=self.config.kernels, strides=self.config.strides, filters=self.config.filters,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device)
        else:
            raise AttributeError(f"{self.config.agent} currently does not support {self.config.representation} representation.")

        # build policy.
        if self.config.policy == "Duel_Q_network":
            policy = REGISTRY_Policy["Duel_Q_network"](
                action_space=self.action_space, representation=representation, hidden_size=self.config.q_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device)
        else:
            raise AttributeError(f"{self.config.agent} currently does not support the policy named {self.config.policy}.")

        return policy

    def _build_learner(self, *args):
        return DuelDQN_Learner(*args)
