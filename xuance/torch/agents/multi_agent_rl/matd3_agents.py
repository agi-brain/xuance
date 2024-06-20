import torch
from argparse import Namespace
from torch import nn
from xuance.environment import DummyVecMultiAgentEnv
from xuance.torch import ModuleDict
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
                 envs: DummyVecMultiAgentEnv):
        super(MATD3_Agents, self).__init__(config, envs)

    def _build_critic_representation(self, representation_key: str, config: Namespace):
        """
        Build representation for the critic of policies.

        Parameters:
            representation_key (str): The selection of representation, e.g., "Basic_MLP", "Basic_RNN", etc.
            config: The configurations for creating the representation module.

        Returns:
            representation (Module): The representation Module.
        """
        normalize_fn = NormalizeFunctions[config.normalize] if hasattr(config, "normalize") else None
        initializer = nn.init.orthogonal_
        activation = ActivationFunctions[config.activation]
        device = self.device
        agent = config.agent

        # build representations
        representation = ModuleDict()
        input_shape = (sum([self.observation_space[k].shape[0] for k in self.agent_keys]), )
        for key in self.model_keys:
            if representation_key == "Basic_Identical":
                representation[key] = REGISTRY_Representation["Basic_Identical"](input_shape=input_shape,
                                                                                 device=self.device)
            elif representation_key == "Basic_MLP":
                representation[key] = REGISTRY_Representation["Basic_MLP"](
                    input_shape=input_shape, hidden_sizes=self.config.representation_hidden_size,
                    normalize=normalize_fn, initialize=initializer, activation=activation, device=device)
            elif representation_key == "Basic_RNN":
                representation[key] = REGISTRY_Representation["Basic_RNN"](
                    input_shape=input_shape,
                    hidden_sizes={'fc_hidden_sizes': self.config.fc_hidden_sizes,
                                  'recurrent_hidden_size': self.config.recurrent_hidden_size},
                    normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
                    N_recurrent_layers=self.config.N_recurrent_layers,
                    dropout=self.config.dropout, rnn=self.config.rnn)
            else:
                raise AttributeError(f"{agent} currently does not support {representation_key} representation.")
        return representation

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
        actor_representation = self._build_representation(self.config.representation, self.config)
        critic_representation = self._build_critic_representation(self.config.representation, self.config)

        # build policies
        if self.config.policy == "MATD3_Policy":
            policy = REGISTRY_Policy["MATD3_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                actor_representation=actor_representation, critic_representation=critic_representation,
                actor_hidden_size=self.config.actor_hidden_size,
                critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
                activation_action=ActivationFunctions[self.config.activation_action],
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
        else:
            raise AttributeError(f"MATD3 currently does not support the policy named {self.config.policy}.")

        return policy

    def _build_learner(self, config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler):
        return MATD3_Learner(config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler)
