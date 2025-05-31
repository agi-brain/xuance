import torch
import torch.nn as nn
from argparse import Namespace
from xuance.common import Union, Optional
from xuance.torch import REGISTRY_Policy
from xuance.torch.agents import OffPolicyAgent, BaseCallback
from xuance.torch.learners.contrastive_unsupervised_rl.spr_learner import SPR_Learner
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.environment import DummyVecEnv, SubprocVecEnv


class SPR_Agent(OffPolicyAgent):

    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecEnv, SubprocVecEnv],
                 callback: Optional[BaseCallback] = None):
        super().__init__(config, envs, callback)
        self._init_exploration_params(config)
        self.policy = self._build_policy()
        self.memory = self._build_memory()
        self.learner = self._build_learner(self.config, self.policy, self.callback)

    def _init_exploration_params(self, config: Namespace):
        self.e_greedy = config.start_greedy
        self.e_greedy_decay = (config.start_greedy - config.end_greedy) / (config.decay_step_greedy / self.n_envs)

    def _build_policy(self) -> nn.Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        activation = ActivationFunctions[self.config.activation]
        initializer = torch.nn.init.orthogonal_

        representation = self._build_representation(self.config.representation, self.observation_space, self.config)
        policy = REGISTRY_Policy[self.config.policy](
            action_space=self.action_space, representation=representation, hidden_size=self.config.q_hidden_size,
            normalize=normalize_fn, initialize=initializer, activation=activation, device=self.device,
            use_distributed_training=self.distributed_training)
        return SPR_Policy(
            device=self.device,
            policy=policy,
        )

    def _build_learner(self, config: Namespace, policy: nn.Module, callback: Optional[BaseCallback] = None):
        return SPR_Learner(
            config=config,
            policy=policy,
            temperature=config.temperature,
            tau=config.tau,
            repr_lr=config.repr_lr,
            prediction_steps=config.prediction_steps,
            callback=callback,
        )

class SPR_Policy(nn.Module):

    def __init__(self,
                 device: str = 'cuda:0',
                 policy: nn.Module = None, ):
        super().__init__()
        self.device = device
        self.policy = policy
        self.representation = self.policy.representation.to(device)
        self.target_representation = self.policy.target_representation.to(device)

        self.q_net = self.policy.eval_Qhead.to(device)

        self.target_q_net = self.policy.target_Qhead.to(device)
        self.action_dim = policy.action_dim
        for param in self.target_q_net.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        features = self.representation(x)
        evalQ = self.q_net(features['state'])
        argmax_action = evalQ.argmax(dim=-1)
        return features, argmax_action, evalQ

    def target(self, x: torch.Tensor):
        with torch.no_grad():
            features = self.target_representation(x)
            evalQ = self.q_net(features['state'])
            argmax_action = evalQ.argmax(dim=-1)
            return features, argmax_action, evalQ

    def copy_target(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())


