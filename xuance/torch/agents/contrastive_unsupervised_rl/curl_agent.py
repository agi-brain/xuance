from typing import Optional

import torch
from argparse import Namespace

from gymnasium import Space
from torch import nn

from xuance.common import Union
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch import REGISTRY_Policy
from xuance.torch.learners import CURL_Learner
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.agents import OffPolicyAgent


class CURL_Agent(OffPolicyAgent):

    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecEnv, SubprocVecEnv]):
        super(CURL_Agent, self).__init__(config, envs)

        self._init_exploration_params(config)

        self.policy = self._build_policy()
        self.memory = self._build_memory()
        self.learner = self._build_learner(config, self.policy)

    def _init_exploration_params(self, config: Namespace):

        self.e_greedy = config.start_greedy
        self.e_greedy_decay = (config.start_greedy - config.end_greedy) / (config.decay_step_greedy / self.n_envs)

    def _build_policy(self) -> nn.Module:

        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        activation = ActivationFunctions[self.config.activation]
        initializer = torch.nn.init.orthogonal_

        representation = self._build_representation(self.config.representation, self.observation_space, self.config)
        target_representation = self._build_representation(self.config.representation, self.observation_space, self.config)
        policy = REGISTRY_Policy[self.config.policy](
            action_space=self.action_space, representation=representation, hidden_size=self.config.q_hidden_size,
            normalize=normalize_fn, initialize=initializer, activation=activation, device=self.device,
            use_distributed_training=self.distributed_training)

        target_policy = REGISTRY_Policy[self.config.policy](
            action_space=self.action_space, representation=target_representation, hidden_size=self.config.q_hidden_size,
            normalize=normalize_fn, initialize=initializer, activation=activation, device=self.device,
            use_distributed_training=self.distributed_training)

        target_representation.load_state_dict(representation.state_dict())
        for param in target_representation.parameters():
            param.requires_grad = False

        return CURL_Policy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            representation=representation,
            target_representation=target_representation,
            hidden_size=self.config.q_hidden_size,
            normalize=normalize_fn,
            activation=activation,
            device=self.device,
            policy = policy,
            target_policy = target_policy,
        )

    def _build_learner(self, config: Namespace, policy: nn.Module):
        return CURL_Learner(
            config=config,
            policy=policy,
        )

    def _learn(self, batch_size: int):
        batch = self.memory.sample(batch_size)
        samples = {
            'obs': torch.as_tensor(batch['obs'], device=self.device),
            'actions': torch.as_tensor(batch['actions'], device=self.device),
            'rewards': torch.as_tensor(batch['rewards'], device=self.device),
            'obs_next': torch.as_tensor(batch['next_obs'], device=self.device),
            'terminals': torch.as_tensor(batch['dones'], dtype=torch.float, device=self.device)
        }

        learner_info = self.learner.update(**samples)

        return {
            "curl_loss": learner_info["curl_loss"],
            "dqn_loss": learner_info["q_loss"],
            "predictQ": learner_info["predictQ"],
            "learning_rate": learner_info["learning_rate"]
        }


class CURL_Policy(nn.Module):

    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 representation: nn.Module,
                 target_representation: nn.Module,
                 hidden_size: int,
                 normalize: Optional[str] = None,
                 activation: Optional[str] = 'ReLU',
                 device: str = 'cuda:0',
                 policy: nn.Module = None,
                 target_policy: nn.Module = None):
        super(CURL_Policy, self).__init__()

        self.device = device

        self.representation = representation.to(device)
        self.target_representation = target_representation.to(device)

        self.q_net = policy

        self.target_q_net = target_policy


        self.target_q_net.load_state_dict(self.q_net.state_dict())
        for param in self.target_q_net.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        features = self.representation(x)
        return self.q_net(features)

    def target(self, x: torch.Tensor):
        with torch.no_grad():
            features = self.target_representation(x)
            return self.target_q_net(features)

    def copy_target(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())
