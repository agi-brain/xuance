import torch
from argparse import Namespace
from xuance.environment import DummyVecMutliAgentEnv
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.representations import REGISTRY_Representation
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.learners import MATD3_Learner
from xuance.torch.agents.multi_agent_rl.iddpg_agents import IDDPG_Agents


class MATD3_Agents(IDDPG_Agents):
    """The implementation of MATD3 agents.

    Args:
        config: The Namespace variable that provides hyper-parameters and other settings.
        envs: The vectorized environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMutliAgentEnv):
        super(MATD3_Agents, self).__init__(config, envs)

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
        representation = {key: None for key in self.model_keys}
        for key in self.model_keys:
            input_shape = self.observation_space[key].shape
            if self.config.representation == "Basic_Identical":
                representation[key] = REGISTRY_Representation["Basic_Identical"](input_shape=input_shape,
                                                                                 device=self.device)
            elif self.config.representation == "Basic_MLP":
                representation[key] = REGISTRY_Representation["Basic_MLP"](
                    input_shape=input_shape, hidden_sizes=self.config.representation_hidden_size,
                    normalize=normalize_fn, initialize=initializer, activation=activation, device=device)
            else:
                raise AttributeError(f"MATD3 currently does not support {self.config.representation} representation.")

        # build policies
        if self.config.policy == "MATD3_Policy":
            policy = REGISTRY_Policy["MATD3_Policy"](
                action_space=self.action_space, n_agents=self.n_agents, representation=representation,
                actor_hidden_size=self.config.actor_hidden_size,
                critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
                activation_action=ActivationFunctions[self.config.activation_action],
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys)
        else:
            raise AttributeError(f"MATD3 currently does not support the policy named {self.config.policy}.")

        return policy

    def _build_learner(self, config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler):
        return MATD3_Learner(config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler)
