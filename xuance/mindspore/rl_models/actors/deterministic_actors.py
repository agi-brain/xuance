from typing import Type, Sequence, Optional, Callable, Union, Tuple
from gymnasium.spaces import Box

from xuance.mindspore import Tensor, Module
from xuance.mindspore.rl_models.heads import DeterministicActorHead


class DeterministicActor(Module):
    actor_head_cls = DeterministicActorHead

    def __init__(self,
                 representation: Module,
                 actor_hidden_size: Sequence[int],
                 action_space: Box,
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 activation_action: Optional[Type[Module]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(action_space, Box):
            self.action_space = action_space
            self.action_dim = action_space.shape[0]
            self.action_low = Tensor(action_space.low)
            self.action_high = Tensor(action_space.high)
        else:
            raise ValueError('action_space must be Box')
        self.representation = representation
        self.representation_info_shape = representation.output_shapes
        self.actor_head = self.actor_head_cls(
            feature_dim=self.representation_info_shape['state'][0],
            hidden_size=actor_hidden_size,
            action_dim=self.action_dim,
            normalizer=normalizer,
            initializer=initializer,
            activation=activation,
            activation_action=activation_action,
            **kwargs
        )

    def construct(self,
                  observation: Union[Tensor, dict],
                  **kwargs) -> Tuple[Tensor, ...]:
        rep_embeddings, rnn_states = self.representation(observation, **kwargs)
        actions = self.actor_head(rep_embeddings, **kwargs)
        return actions, rnn_states
