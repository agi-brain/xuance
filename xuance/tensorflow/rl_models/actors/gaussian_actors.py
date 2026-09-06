from typing import Type, Sequence, Optional, Union
from gymnasium.spaces import Space, Box
from xuance.tensorflow import keras, Tensor, Module
from xuance.tensorflow.rl_models.heads import GaussianActorHead, SAC_GaussianActorHead
from xuance.tensorflow.rl_models.modules import StochasticActorOutput, RNN_State


class GaussianActor(Module):
    actor_head_cls = GaussianActorHead

    def __init__(self,
                 representation: Module,
                 actor_hidden_size: Sequence[int],
                 action_space: Optional[Space] = None,
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
            self.action_low = action_space.low
            self.action_high = action_space.high
        else:
            raise ValueError('action_space must be Box')

        self.representation = representation
        self.actor_hidden_size = actor_hidden_size
        self.normalizer = normalizer
        self.initializer = initializer
        self.activation = activation
        self.activation_action = activation_action

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

    def call(self,
             observation: Union[Tensor, dict],
             avail_actions: Optional[Tensor] = None,
             agent_indices: Optional[Tensor] = None,
             rnn_states: Optional[RNN_State] = None,
             **kwargs) -> StochasticActorOutput:
        rep_out = self.representation(observation, agent_indices=agent_indices, rnn_states=rnn_states, **kwargs)
        return StochasticActorOutput(
            representations=rep_out,
            distributions=self.actor_head(rep_out.embeddings, **kwargs)
        )

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            representation=self.representation.clone(copy_weights=True, trainable=False,
                                                     name="target_actor_representation"),
            actor_hidden_size=self.actor_hidden_size,
            action_space=self.action_space,
            normalizer=self.normalizer,
            initializer=self.initializer,
            activation=self.activation,
            activation_action=self.activation_action
        ))
        return config


class SAC_GaussianActor(GaussianActor):
    actor_head_cls = SAC_GaussianActorHead

    def call(self,
             observation: Union[Tensor, dict],
             agent_indices: Optional[Tensor] = None,
             rnn_states: Optional[RNN_State] = None,
             **kwargs) -> StochasticActorOutput:
        rep_out = self.representation(observation, agent_indices=agent_indices, rnn_states=rnn_states, **kwargs)
        return StochasticActorOutput(
            representations=rep_out,
            distributions=self.actor_head(rep_out.embeddings, **kwargs)
        )
