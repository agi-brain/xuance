from typing import Type, Optional, Callable, Sequence

from xuance.mindspore import ms, ops, nn, Tensor, Module, Parameter
from xuance.mindspore.rl_models.modules import mlp_block
from xuance.mindspore.rl_models.modules.distributions import (
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
                 **kwargs):
        super().__init__(**kwargs)
        layers = []
        input_shape = (feature_dim,)
        for h in hidden_size:
            mlp, input_shape = mlp_block(input_shape[0], h, normalizer, activation, initializer)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, None, initializer)[0])
        self.logits = nn.SequentialCell(*layers)
        self.policy_distribution = CategoricalDistribution(action_dim=action_dim)

    def construct(self,
                  features: Tensor,
                  avail_actions: Optional[Tensor] = None,
                  **kwargs):
        logits = self.logits(features)
        if avail_actions is not None:
            logits[avail_actions == 0] = -1e10
        return logits

    def distribution(self, logits: Tensor):
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
                 **kwargs):
        super().__init__(**kwargs)
        layers = []
        input_shape = (feature_dim,)
        for h in hidden_size:
            mlp, input_shape = mlp_block(input_shape[0], h, normalizer, activation, initializer)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, activation_action, initializer)[0])
        self.mu = nn.SequentialCell(*layers)
        self.log_std = Parameter(-ops.ones((action_dim,), dtype=ms.float32))
        self.policy_distribution = DiagGaussianDistribution(action_dim)

    def construct(self,
                  features: Tensor,
                  avail_actions: Optional[Tensor] = None,
                  **kwargs):
        _mu = self.mu(features)
        _log_std = self.log_std.exp()
        return _mu, _log_std

    def distribution(self, mu: Tensor, std: Tensor):
        self.policy_distribution.set_param(mu=mu, std=std)
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
                 **kwargs):
        super().__init__(**kwargs)
        layers = []
        input_shape = (feature_dim,)
        for h in hidden_size:
            mlp, input_shape = mlp_block(input_shape[0], h, normalizer, activation, initializer)
            layers.extend(mlp)
        self.output = nn.SequentialCell(*layers)
        self.out_mu = nn.Linear(hidden_size[-1], action_dim)
        self.out_log_std = nn.Linear(hidden_size[-1], action_dim)
        self.policy_distribution = ActivatedDiagGaussianDistribution(action_dim, activation_action)

    def construct(self,
                  features: Tensor,
                  avail_actions: Optional[Tensor] = None,
                  deterministic: bool = False,
                  **kwargs):
        output = self.output(features)
        mu = self.out_mu(output)
        log_std = ops.clamp(self.out_log_std(output), -20, 2)
        return mu, log_std


class DeterministicActorHead(Module):
    def __init__(self,
                 feature_dim: int,
                 hidden_size: Sequence[int],
                 action_dim: int,
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 activation_action: Optional[Type[Module]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        layers = []
        input_shape = (feature_dim,)
        for h in hidden_size:
            mlp, input_shape = mlp_block(input_shape[0], h, normalizer, activation, initializer)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, activation_action, initializer)[0])
        self.model = nn.SequentialCell(*layers)

    def construct(self,
                  features: Tensor,
                  avail_actions: Optional[Tensor] = None,
                  **kwargs):
        actions = self.model(features)
        if avail_actions is not None:
            actions[avail_actions == 0] = 0
        return actions
