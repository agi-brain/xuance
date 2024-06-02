import torch
from argparse import Namespace
from xuance.environment import DummyVecEnv
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.representations import REGISTRY_Representation
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.learners import TD3_Learner
from xuance.torch.agents.policy_gradient.ddpg_agent import DDPG_Agent
from xuance.common import space2shape


class TD3_Agent(DDPG_Agent):
    """The implementation of TD3 agent.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv):
        super(TD3_Agent, self).__init__(config, envs)

    def _build_policy(self):
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device

        # build representations.
        if self.config.representation == "Basic_Identical":
            representation = REGISTRY_Representation["Basic_Identical"](input_shape=space2shape(self.observation_space),
                                                                        device=self.device)
        elif self.config.representation == "Basic_MLP":
            representation = REGISTRY_Representation["Basic_MLP"](
                input_shape=space2shape(self.observation_space),
                hidden_sizes=self.config.representation_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device)
        else:
            raise AttributeError(f"TD3 currently does not support {self.config.representation} representation.")

        # build policy
        if self.config.policy == "TD3_Policy":
            policy = REGISTRY_Policy["TD3_Policy"](
                action_space=self.action_space, representation=representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, device=device,
                activation=activation, activation_action=ActivationFunctions[self.config.activation_action])
        else:
            raise AttributeError(f"TD3 currently does not support the policy named {self.config.policy}.")

        return policy

    def _build_learner(self, *args):
        return TD3_Learner(*args)
