import torch
from copy import deepcopy
from typing import Type, Sequence, Optional, Callable, Union, Dict
from gymnasium.spaces import Box, Discrete
from torch import Tensor
from torch.nn import Module
from xuance.torch.rl_models.heads import ValueHead, QValueHead
from xuance.torch.rl_models.modules import TwinCriticOutput, RNN_State


class TwinActionValueCritic(Module):
    def __init__(self,
                 representation: Module,
                 action_space: Box,
                 critic_hidden_size: Sequence[int],
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs) -> None:
        super().__init__()
        if isinstance(action_space, Box):
            self.action_space = action_space
            self.action_dim = action_space.shape[-1]
        else:
            raise ValueError('action_space must be Box.')
        self.representation_1 = representation
        self.representation_2 = deepcopy(representation)

        self.representation_info_shape = representation.output_shapes
        self.feature_dim = self.representation_info_shape['state'][0] + self.action_dim
        value_head_input = dict(
            feature_dim=self.feature_dim,
            hidden_size=critic_hidden_size,
            normalizer=normalizer,
            initializer=initializer,
            activation=activation,
            device=device,
            **kwargs,
        )

        self.critic_head_1 = ValueHead(**value_head_input)
        self.critic_head_2 = ValueHead(**value_head_input)

    def forward(self,
                observation: Union[Tensor, dict],
                actions: Union[Tensor, dict],
                rnn_states_1: Dict[str, RNN_State | dict] = None,
                rnn_states_2: Dict[str, RNN_State | dict] = None,
                **kwargs) -> TwinCriticOutput:
        if rnn_states_1 is not None:
            kwargs["rnn_states"] = rnn_states_1
        rep_out_1 = self.representation_1(observation, **kwargs)
        if rnn_states_1 is not None:
            kwargs["rnn_states"] = rnn_states_2
        rep_out_2 = self.representation_2(observation, **kwargs)
        return TwinCriticOutput(
            representations_1=rep_out_1,
            representations_2=rep_out_2,
            values_1=self.critic_head_1(torch.concat([rep_out_1.embeddings, actions], dim=-1), **kwargs),
            values_2=self.critic_head_2(torch.concat([rep_out_2.embeddings, actions], dim=-1), **kwargs)
        )


class TwinDiscreteActionValueCritic(Module):

    def __init__(self,
                 representation: Module,
                 action_space: Discrete,
                 critic_hidden_size: Sequence[int],
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs) -> None:
        super().__init__()
        if isinstance(action_space, Discrete):
            self.action_space = action_space
            self.n_actions = action_space.n
        else:
            raise ValueError('action_space must be Box.')
        self.representation_1 = representation
        self.representation_2 = deepcopy(representation)

        self.representation_info_shape = representation.output_shapes
        self.feature_dim = self.representation_info_shape['state'][0]
        value_head_input = dict(
            feature_dim=self.feature_dim,
            hidden_size=critic_hidden_size,
            n_actions=self.n_actions,
            normalizer=normalizer,
            initializer=initializer,
            activation=activation,
            device=device,
            **kwargs,
        )

        self.critic_head_1 = QValueHead(**value_head_input)
        self.critic_head_2 = QValueHead(**value_head_input)

    def forward(self,
                observation: Union[Tensor, dict],
                rnn_states_1: Dict[str, RNN_State | dict] = None,
                rnn_states_2: Dict[str, RNN_State | dict] = None,
                **kwargs) -> TwinCriticOutput:
        if rnn_states_1 is not None:
            kwargs["rnn_states"] = rnn_states_1
        rep_out_1 = self.representation_1(observation, **kwargs)
        if rnn_states_1 is not None:
            kwargs["rnn_states"] = rnn_states_2
        rep_out_2 = self.representation_2(observation, **kwargs)
        return TwinCriticOutput(
            representations_1=rep_out_1,
            representations_2=rep_out_2,
            values_1=self.critic_head_1(rep_out_1.embeddings, **kwargs),
            values_2=self.critic_head_2(rep_out_2.embeddings, **kwargs)
        )
