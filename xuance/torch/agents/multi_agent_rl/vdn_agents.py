import torch
from argparse import Namespace
from xuance.common import Union, Optional, MultiAgentBaseCallback
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.torch import Module
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy, VDN_mixer
from xuance.torch.agents import OffPolicyMARLAgents


class VDN_Agents(OffPolicyMARLAgents):
    """The implementation of Value Decomposition Networks (VDN) agents.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """

    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv],
                 callback: Optional[MultiAgentBaseCallback] = None):
        super(VDN_Agents, self).__init__(config, envs, callback)

        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.e_greedy = self.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / (config.decay_step_greedy / self.n_envs)

        self.policy = self._build_policy()  # build policy
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, self.policy, self.callback)

    def _build_policy(self) -> Module:
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
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policies
        mixer = VDN_mixer()
        if self.config.policy == "Mixing_Q_network":
            policy = REGISTRY_Policy["Mixing_Q_network"](
                action_space=self.action_space, n_agents=self.n_agents, representation=representation,
                mixer=mixer, hidden_size=self.config.q_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                device=device, use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
        else:
            raise AttributeError(f"VDN currently does not support the policy named {self.config.policy}.")

        return policy
