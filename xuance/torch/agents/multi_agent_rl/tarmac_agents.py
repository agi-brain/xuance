from argparse import Namespace
from typing import Union, Optional, Dict
import gymnasium as gym
import numpy as np
import torch
from gymnasium import Space
from torch.nn import Module, ModuleDict

from xuance.common import space2shape

from xuance.torch import REGISTRY_Policy
from xuance.torch.communications.attention_comm import TarMAC
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions

from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.torch.agents import BaseCallback
from xuance.torch.agents.multi_agent_rl.ic3net_agents import IC3Net_Agents


class TarMAC_Agents(IC3Net_Agents):
    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv],
                 callback: Optional[BaseCallback] = None):
        super(TarMAC_Agents, self).__init__(config, envs, callback)
        self.policy = self._build_policy()
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, self.policy, callback)

    def _build_communicator(self, input_space: Union[Dict[str, Space], Dict[str, tuple]], ) -> Module:
        communicator = ModuleDict()
        hidden_sizes = {'fc_hidden_sizes': self.config.fc_hidden_sizes,
                        'recurrent_hidden_size': self.config.recurrent_hidden_size}
        for key in self.model_keys:
            input_communicator = dict(
                input_shape=space2shape(input_space[key]),
                hidden_sizes=hidden_sizes,
                comm_passes=self.config.comm_passes,
                model_keys=self.model_keys,
                agent_keys=self.agent_keys,
                n_agents=self.n_agents,
                device=self.device,
                config=self.config)
            communicator[key] = TarMAC(**input_communicator)
        return communicator

    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device
        agent = self.config.agent

        # build representations
        communicator = self._build_communicator(self.observation_space)
        space_actor_in = {agent: gym.spaces.Box(-np.inf, np.inf, (self.config.recurrent_hidden_size,), dtype=np.float32)
                          for agent in self.observation_space}
        A_representation = self._build_representation(self.config.representation, space_actor_in, self.config)
        C_representation = self._build_representation(self.config.representation, space_actor_in, self.config)

        # build policies
        if self.config.policy == "TarMAC_Policy":
            policy = REGISTRY_Policy[self.config.policy](
                action_space=self.action_space, n_agents=self.n_agents,
                representation_actor=A_representation, representation_critic=C_representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                device=device, use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None,
                communicator=communicator, agent_keys=self.agent_keys,
                comm_passes=self.config.comm_passes, config=self.config)
            self.continuous_control = False
        else:
            raise AttributeError(f"{agent} currently does not support the policy named {self.config.policy}.")
        return policy