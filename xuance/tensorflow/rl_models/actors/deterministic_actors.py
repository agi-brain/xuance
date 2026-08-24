from typing import Type, Sequence, Optional, Union
from gymnasium.spaces import Box
from xuance.tensorflow import tf, keras, Tensor, Module
from xuance.tensorflow.rl_models.heads import DeterministicActorHead
from xuance.tensorflow.rl_models.modules import DeterministicActorOutput


class DeterministicActor(Module):
    actor_head_cls = DeterministicActorHead

    def __init__(self,
                 representation: Module,
                 actor_hidden_size: Sequence[int],
                 action_space: Box,
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[keras.initializers.Initializer] = None,
                 activation: Optional[Type[Module]] = None,
                 activation_action: Optional[Type[Module]] = None,
                 device: str = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(action_space, Box):
            self.action_space = action_space
            self.action_dim = action_space.shape[0]
            self.action_low = tf.convert_to_tensor(action_space.low, name="action_low")
            self.action_high = tf.convert_to_tensor(action_space.high, name="action_high")
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
            device=device,
            **kwargs
        )

    def call(self,
             observation: Union[Tensor, dict],
             **kwargs) -> DeterministicActorOutput:
        rep_out = self.representation(observation, **kwargs)
        return DeterministicActorOutput(
            representations=rep_out,
            actions=self.actor_head(rep_out.embeddings, **kwargs)
        )
