import torch
import gymnasium
from typing import Type, Sequence, Optional, Callable, Union
from gymnasium.spaces import Discrete, Box
from torch import Tensor
from torch.nn import Module
from xuance.torch.rl_models.heads import ValueHead, QValueHead
from xuance.torch.rl_models.modules import CriticOutput


class StateValueCritic(Module):
    """
    V(s)
    """

    def __init__(self,
                 representation: Module,
                 critic_hidden_size: Sequence[int],
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.representation = representation
        self.representation_info_shape = representation.output_shapes
        self.critic_head = ValueHead(
            feature_dim=self.representation_info_shape['state'][0],
            hidden_size=critic_hidden_size,
            normalizer=normalizer,
            initializer=initializer,
            activation=activation,
            device=device,
            **kwargs,
        )

    def forward(self,
                observation: Union[Tensor, dict],
                **kwargs) -> CriticOutput:
        rep_out = self.representation(observation, **kwargs)
        return CriticOutput(
            representations=rep_out,
            values=self.critic_head(rep_out.embeddings, **kwargs)
        )


class ActionValueCritic(Module):
    """
    Q(s, a)
    """

    def __init__(self,
                 representation: Module,
                 action_space: Union[Box],
                 critic_hidden_size: Sequence[int],
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(action_space, Box):
            self.action_space = action_space
            self.action_dim = action_space.shape[-1]
        else:
            raise ValueError('action_space must be Box.')
        self.representation = representation
        self.representation_info_shape = representation.output_shapes

        self.critic_head = ValueHead(
            feature_dim=self.representation_info_shape['state'][0] + self.action_dim,
            hidden_size=critic_hidden_size,
            normalizer=normalizer,
            initializer=initializer,
            activation=activation,
            device=device,
            **kwargs,
        )

    def forward(self,
                observations: Union[Tensor, dict],
                actions: Union[Tensor, dict],
                **kwargs) -> CriticOutput:
        rep_out = self.representation(observations, **kwargs)
        return CriticOutput(
            representations=rep_out,
            values=self.critic_head(torch.concat([rep_out.embeddings, actions], dim=-1), **kwargs)
        )


class DiscreteActionValueCritic(Module):
    """
    Q(s,·)
    """

    def __init__(self,
                 representation: Module,
                 action_space: Union[Discrete, Box],
                 critic_hidden_size: Sequence[int],
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(action_space, Discrete):
            self.action_space = action_space
            self.n_actions = action_space.n
        else:
            raise ValueError('action_space must be Discrete.')
        self.representation = representation
        self.representation_info_shape = representation.output_shapes

        self.critic_head = QValueHead(
            feature_dim=self.representation_info_shape['state'][0],
            hidden_size=critic_hidden_size,
            n_actions=self.n_actions,
            normalizer=normalizer,
            initializer=initializer,
            activation=activation,
            device=device,
            **kwargs,
        )

    def forward(self,
                observation: Tensor,
                **kwargs) -> CriticOutput:
        rep_out = self.representation(observation, **kwargs)
        return CriticOutput(
            representations=rep_out,
            values=self.critic_head(rep_out.embeddings, **kwargs)
        )


class HybridActionValueCritic(Module):
    """
    Q(s,a_con, ·)
    """

    def __init__(self,
                 representation: Module,
                 action_space: gymnasium.spaces.Tuple,
                 critic_hidden_size: Sequence[int],
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(action_space, gymnasium.spaces.Tuple):
            self.action_space = action_space
            self.num_disact = self.action_space.spaces[0].n
            self.conact_sizes = [self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)]
            self.conact_size = sum(self.conact_sizes)
        else:
            raise ValueError('Invalid action space.')
        self.representation = representation
        self.representation_info_shape = representation.output_shapes

        self.critic_head = QValueHead(
            feature_dim=self.representation_info_shape['state'][0] + self.conact_size,
            hidden_size=critic_hidden_size,
            n_actions=self.num_disact,
            normalizer=normalizer,
            initializer=initializer,
            activation=activation,
            device=device,
            **kwargs,
        )

    def forward(self,
                observation: Union[Tensor, dict],
                actions: Tensor,
                **kwargs) -> CriticOutput:
        rep_out = self.representation(observation, **kwargs)
        critic_input = torch.concat([rep_out.embeddings, actions], dim=1)
        return CriticOutput(
            representations=rep_out,
            values=self.critic_head(critic_input, **kwargs)
        )


class MeanFieldStateValueCritic(Module):
    """
    V(s, a_mean)
    """

    def __init__(self,
                 representation: Module,
                 mean_actions_encoder: Module,
                 critic_hidden_size: Sequence[int],
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.representation = representation
        self.mean_actions_encoder = mean_actions_encoder
        self.representation_feature_dim = representation.output_shapes['state'][0]
        self.mean_action_feature_dim = mean_actions_encoder.output_shapes['state'][0]
        self.critic_head = ValueHead(
            feature_dim=self.representation_feature_dim + self.mean_action_feature_dim,
            hidden_size=critic_hidden_size,
            normalizer=normalizer,
            initializer=initializer,
            activation=activation,
            device=device,
            **kwargs,
        )

    def forward(self,
                observation: Union[Tensor, dict],
                mean_actions: Tensor,
                **kwargs) -> CriticOutput:
        rep_out = self.representation(observation, **kwargs)
        mean_actions_rep_out = self.mean_actions_encoder(mean_actions, **kwargs)
        critic_input = torch.concat([rep_out.embeddings, mean_actions_rep_out.embeddings], dim=-1)
        return CriticOutput(
            representations=rep_out,
            values=self.critic_head(critic_input, **kwargs)
        )


class MeanFieldActionValueCritic(Module):
    """
    Q(s, a_mean, ·)
    """

    def __init__(self,
                 representation: Module,
                 mean_actions_encoder: Module,
                 action_space: gymnasium.spaces.Tuple,
                 critic_hidden_size: Sequence[int],
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(action_space, Discrete):
            self.action_space = action_space
            self.n_actions = action_space.n
        else:
            raise ValueError('action_space must be Discrete.')
        self.representation = representation
        self.mean_actions_encoder = mean_actions_encoder
        self.representation_feature_dim = representation.output_shapes['state'][0]
        self.mean_action_feature_dim = mean_actions_encoder.output_shapes['state'][0]

        self.critic_head = QValueHead(
            feature_dim=self.representation_feature_dim + self.mean_action_feature_dim,
            hidden_size=critic_hidden_size,
            n_actions=self.n_actions,
            normalizer=normalizer,
            initializer=initializer,
            activation=activation,
            device=device,
            **kwargs,
        )

    def forward(self,
                observations: Tensor,
                mean_actions: Tensor,
                **kwargs) -> CriticOutput:
        rep_out = self.representation(observations, **kwargs)
        mean_actions_rep_out = self.mean_actions_encoder(mean_actions, **kwargs)
        critic_input = torch.concat([rep_out.embeddings, mean_actions_rep_out.embeddings], dim=-1)
        return CriticOutput(
            representations=rep_out,
            values=self.critic_head(critic_input, **kwargs)
        )
