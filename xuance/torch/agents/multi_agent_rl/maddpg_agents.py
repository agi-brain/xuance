import torch
from argparse import Namespace
from xuance.environment import DummyVecMultiAgentEnv
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.learners import MADDPG_Learner
from xuance.torch.agents.multi_agent_rl.iddpg_agents import IDDPG_Agents


class MADDPG_Agents(IDDPG_Agents):
    """The implementation of MADDPG agents.

    Args:
        config: The Namespace variable that provides hyper-parameters and other settings.
        envs: The vectorized environments.
    """
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMultiAgentEnv):
        super(MADDPG_Agents, self).__init__(config, envs)

    def _build_policy(self):
        """
        Build representation(s) and policy(ies) for agent(s)

        Returns:
            policy (torch.nn.Module): A dict of policies.
        """
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device

        # build representations
        representation = self._build_representation(self.config.representation, self.config)

        # build policies
        if self.config.policy == "MADDPG_Policy":
            policy = REGISTRY_Policy["MADDPG_Policy"](
                action_space=self.action_space, n_agents=self.n_agents, representation=representation,
                actor_hidden_size=self.config.actor_hidden_size,
                critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
                activation_action=ActivationFunctions[self.config.activation_action],
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys)
        else:
            raise AttributeError(f"MADDPG currently does not support the policy named {self.config.policy}.")

        return policy

    def _build_learner(self, config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler):
        return MADDPG_Learner(config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler)

