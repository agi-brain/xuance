from argparse import Namespace
from typing import Union, Optional, Dict
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces, Space
from torch.nn import Module, ModuleDict

from examples.dgn.dgn_learner import DGN_Learner
from examples.dgn.dgn_policy import DGN_Policy
from xuance.common import space2shape
from xuance.torch import REGISTRY_Policy, REGISTRY_Learners
from xuance.torch.communications.gnn_comm import DGNComm
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.common import MultiAgentBaseCallback

from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.torch.agents import CommNet_Agents


class DGN_Agents(CommNet_Agents):
    def __init__(self, config: Namespace, envs: Union[DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv],
                 callback: Optional[MultiAgentBaseCallback] = None):
        super(DGN_Agents, self).__init__(config, envs, callback)
        self.policy = self._build_policy()  # build policy
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
                model_keys=self.model_keys,
                agent_keys=self.agent_keys,
                n_agents=self.n_agents,
                device=self.device,
                config=self.config)
            communicator[key] = DGNComm(**input_communicator)
        return communicator

    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device
        agent = self.config.agent
        max_length = max(space.shape[0] for space in self.observation_space.values())
        self.observation_space = {agent: gym.spaces.Box(-np.inf, np.inf, (max_length,), dtype=np.float32)
                                  for agent in self.observation_space}
        # build representations
        communicator = self._build_communicator(self.observation_space)
        space_actor_in = {agent: gym.spaces.Box(-np.inf, np.inf, (self.config.recurrent_hidden_size,), dtype=np.float32)
                          for agent in self.observation_space}
        if self.use_global_state:
            dim_obs_all = sum(self.state_space.shape)
        else:
            dim_obs_all = sum([sum(self.observation_space[k].shape) for k in self.agent_keys])
        space_critic_in = {k: (dim_obs_all,) for k in self.agent_keys}
        A_representation = self._build_representation(self.config.representation, space_actor_in, self.config)
        C_representation = self._build_representation(self.config.representation, space_critic_in, self.config)

        # build policies
        if self.config.policy == "DGN_Policy":
            REGISTRY_Policy["DGN_Policy"] = DGN_Policy
            policy = REGISTRY_Policy[self.config.policy](
                action_space=self.action_space, n_agents=self.n_agents,
                representation_actor=A_representation, representation_critic=C_representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                device=device, use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None,
                communicator=communicator, agent_keys=self.agent_keys, comm_passes=self.config.comm_passes)
            self.continuous_control = False
        else:
            raise AttributeError(f"{agent} currently does not support the policy named {self.config.policy}.")
        return policy

    def _build_learner(self, *args):
        REGISTRY_Learners['DGN_Learner'] = DGN_Learner
        return REGISTRY_Learners[self.config.learner](*args)