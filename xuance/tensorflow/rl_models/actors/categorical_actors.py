from typing import Type, Sequence, Optional, Union
from gymnasium.spaces import Space, Discrete
from xuance.tensorflow import keras, Tensor, Module
from xuance.tensorflow.rl_models.heads import CategoricalActorHead
from xuance.tensorflow.rl_models.modules import StochasticActorOutput


class CategoricalActor(Module):
    def __init__(self,
                 representation: Type[Type[Module]],
                 actor_hidden_size: Sequence[int],
                 action_space: Optional[Space] = None,
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[keras.initializers.Initializer] = None,
                 activation: Optional[Type[Module]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(action_space, Discrete):
            self.action_dim = action_space.n
        else:
            raise ValueError('action_space must be Discrete')
        self.representation = representation
        self.representation_info_shape = representation.output_shapes
        self.actor_head = CategoricalActorHead(
            feature_dim=self.representation_info_shape['state'][0],
            hidden_size=actor_hidden_size,
            action_dim=self.action_dim,
            normalizer=normalizer,
            initializer=initializer,
            activation=activation,
            **kwargs
        )

    def call(self,
             observation: Union[Tensor, dict],
             avail_actions: Optional[Tensor] = None,
             **kwargs) -> StochasticActorOutput:
        rep_out = self.representation(observation, **kwargs)
        return StochasticActorOutput(
            representations=rep_out,
            distributions=self.actor_head(rep_out.embeddings,
                                          avail_actions=avail_actions,
                                          **kwargs)
        )
