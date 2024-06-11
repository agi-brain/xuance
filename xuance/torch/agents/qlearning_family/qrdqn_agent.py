import torch
from argparse import Namespace
from xuance.environment import DummyVecEnv
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.learners import QRDQN_Learner
from xuance.torch.agents.qlearning_family.dqn_agent import DQN_Agent


class QRDQN_Agent(DQN_Agent):
    """The implementation of QRDQN agent.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv):
        super(QRDQN_Agent, self).__init__(config, envs)

    def _build_policy(self):
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device

        # build representation.
        representation = self._build_representation(self.config.representation, self.config)

        # build policy.
        if self.config.policy == "QR_Q_network":
            policy = REGISTRY_Policy["QR_Q_network"](
                action_space=self.action_space, quantile_num=self.config.quantile_num,
                representation=representation, hidden_size=self.config.q_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device)
        else:
            raise AttributeError(f"{self.config.agent} currently does not support the policy named {self.config.policy}.")

        return policy

    def _build_learner(self, *args):
        return QRDQN_Learner(*args)
