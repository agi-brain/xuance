import torch
from torch import nn
from typing import Type, Optional, Union, Callable, Sequence
from torch import Tensor
from torch.nn import Module
from xuance.torch.rl_models.modules import mlp_block
from xuance.torch.rl_models.modules.distributions import (
    CategoricalDistribution,
    DiagGaussianDistribution,
    ActivatedDiagGaussianDistribution
)


class CategoricalActorHead(Module):
    def __init__(self,
                 feature_dim: int,
                 hidden_size: Sequence[int],
                 action_dim: int,
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        layers = []
        input_shape = (feature_dim, )
        for h in hidden_size:
            mlp, input_shape = mlp_block(input_shape[0], h, normalizer, activation, initializer, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, None, initializer, device)[0])
        self.logits = nn.Sequential(*layers)
        self.policy_distribution = CategoricalDistribution(action_dim=action_dim)

    def forward(self,
                features: Tensor,
                avail_actions: Optional[Tensor] = None,
                **kwargs):
        logits = self.logits(features)
        if avail_actions is not None:
            logits[avail_actions == 0] = -1e10
        self.policy_distribution.set_param(logits=logits)
        return self.policy_distribution


class GaussianActorHead(Module):
    def __init__(self,
                 feature_dim: int,
                 hidden_size: Sequence[int],
                 action_dim: int,
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 activation_action: Optional[Type[Module]] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        layers = []
        input_shape = (feature_dim,)
        for h in hidden_size:
            mlp, input_shape = mlp_block(input_shape[0], h, normalizer, activation, initializer, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, activation_action, initializer, device)[0])
        self.mu = nn.Sequential(*layers)
        self.log_std = nn.Parameter(-torch.ones((action_dim,), device=device))
        self.policy_distribution = DiagGaussianDistribution(action_dim)

    def forward(self,
                features: Tensor,
                avail_actions: Optional[Tensor] = None,
                **kwargs):
        self.policy_distribution.set_param(self.mu(features), self.log_std.exp())
        return self.policy_distribution


class SAC_GaussianActorHead(Module):
    def __init__(self,
                 feature_dim: int,
                 hidden_size: Sequence[int],
                 action_dim: int,
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 activation_action: Optional[Type[Module]] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        layers = []
        input_shape = (feature_dim,)
        for h in hidden_size:
            mlp, input_shape = mlp_block(input_shape[0], h, normalizer, activation, initializer, device)
            layers.extend(mlp)
        self.output = nn.Sequential(*layers)
        self.out_mu = nn.Linear(hidden_size[-1], action_dim, device=device)
        self.out_log_std = nn.Linear(hidden_size[-1], action_dim, device=device)
        self.policy_distribution = ActivatedDiagGaussianDistribution(action_dim, activation_action, device)

    def forward(self,
                features: Tensor,
                avail_actions: Optional[Tensor] = None,
                **kwargs):
        output = self.output(features)
        mu = self.out_mu(output)
        log_std = torch.clamp(self.out_log_std(output), -20, 2)
        self.policy_distribution.set_param(mu, log_std.exp())
        return self.policy_distribution


class DeterministicActorHead(Module):
    def __init__(self,
                 feature_dim: int,
                 hidden_size: Sequence[int],
                 action_dim: int,
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 activation_action: Optional[Type[Module]] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        layers = []
        input_shape = (feature_dim,)
        for h in hidden_size:
            mlp, input_shape = mlp_block(input_shape[0], h, normalizer, activation, initializer, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, activation_action, initializer, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self,
                features: Tensor,
                avail_actions: Optional[Tensor] = None,
                **kwargs):
        actions = self.model(features)
        if avail_actions is not None:
            actions[avail_actions == 0] = 0
        return actions
