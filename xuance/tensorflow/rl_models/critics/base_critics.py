import gymnasium
from typing import Type, Sequence, Optional, Union
from gymnasium.spaces import Discrete, Box
from xuance.tensorflow import tf, keras, Tensor, Module
from xuance.tensorflow.rl_models.heads import ValueHead, QValueHead
from xuance.tensorflow.rl_models.modules import CriticOutput, RNN_State


class StateValueCritic(Module):
    """
    V(s)
    """

    def __init__(self,
                 representation: Module,
                 critic_hidden_size: Sequence[int],
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[keras.initializers.Initializer] = None,
                 activation: Optional[Type[Module]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.representation = representation
        self.critic_hidden_size = critic_hidden_size
        self.normalizer = normalizer
        self.initializer = initializer
        self.activation = activation

        self.representation_info_shape = representation.output_shapes
        self.critic_head = ValueHead(
            feature_dim=self.representation_info_shape['state'][0],
            hidden_size=critic_hidden_size,
            normalizer=normalizer,
            initializer=initializer,
            activation=activation,
            **kwargs,
        )

    def call(self,
             observation: Union[Tensor, dict],
             **kwargs) -> CriticOutput:
        rep_out = self.representation(observation, **kwargs)
        return CriticOutput(
            representations=rep_out,
            values=self.critic_head(rep_out.embeddings, **kwargs)
        )

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            representation=self.representation.clone(copy_weights=True, trainable=False,
                                                     name="target_critic_representation"),
            critic_hidden_size=self.critic_hidden_size,
            normalizer=self.normalizer,
            initializer=self.initializer,
            activation=self.activation,
        ))
        return config


class ActionValueCritic(Module):
    """
    Q(s, a)
    """

    def __init__(self,
                 representation: Module,
                 action_space: Union[Box],
                 critic_hidden_size: Sequence[int],
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[keras.initializers.Initializer] = None,
                 activation: Optional[Type[Module]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(action_space, Box):
            self.action_space = action_space
            self.action_dim = action_space.shape[-1]
        else:
            raise ValueError('action_space must be Box.')
        self.representation = representation
        self.critic_hidden_size = critic_hidden_size
        self.normalizer = normalizer
        self.initializer = initializer
        self.activation = activation

        self.representation_info_shape = representation.output_shapes

        self.critic_head = ValueHead(
            feature_dim=self.representation_info_shape['state'][0] + self.action_dim,
            hidden_size=critic_hidden_size,
            normalizer=normalizer,
            initializer=initializer,
            activation=activation,
            **kwargs,
        )

    def call(self,
             observations: Union[Tensor, dict],
             actions: Union[Tensor, dict],
             agent_indices: Optional[Tensor] = None,
             rnn_states: Optional[RNN_State] = None,
             **kwargs) -> CriticOutput:
        rep_out = self.representation(observations, agent_indices=agent_indices, rnn_states=rnn_states, **kwargs)
        return CriticOutput(
            representations=rep_out,
            values=self.critic_head(tf.concat([rep_out.embeddings, actions], axis=-1), **kwargs)
        )

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            representation=self.representation.clone(copy_weights=True, trainable=False,
                                                     name="target_critic_representation"),
            action_space=self.action_space,
            critic_hidden_size=self.critic_hidden_size,
            normalizer=self.normalizer,
            initializer=self.initializer,
            activation=self.activation,
        ))
        return config


class DiscreteActionValueCritic(Module):
    """
    Q(s,·)
    """

    def __init__(self,
                 representation: Module,
                 action_space: Union[Discrete, Box],
                 critic_hidden_size: Sequence[int],
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[keras.initializers.Initializer] = None,
                 activation: Optional[Type[Module]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(action_space, Discrete):
            self.action_space = action_space
            self.n_actions = action_space.n
        else:
            raise ValueError('action_space must be Discrete.')
        self.representation = representation
        self.critic_hidden_size = critic_hidden_size
        self.normalizer = normalizer
        self.initializer = initializer
        self.activation = activation

        self.representation_info_shape = representation.output_shapes

        self.critic_head = QValueHead(
            feature_dim=self.representation_info_shape['state'][0],
            hidden_size=critic_hidden_size,
            n_actions=self.n_actions,
            normalizer=normalizer,
            initializer=initializer,
            activation=activation,
            **kwargs,
        )

    def call(self,
             observation: Tensor,
             agent_indices: Optional[Tensor] = None,
             rnn_states: Optional[RNN_State] = None,
             **kwargs) -> CriticOutput:
        rep_out = self.representation(observation, agent_indices=agent_indices, rnn_states=rnn_states, **kwargs)
        return CriticOutput(
            representations=rep_out,
            values=self.critic_head(rep_out.embeddings, **kwargs)
        )

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            representation=self.representation.clone(copy_weights=True, trainable=False,
                                                     name="target_critic_representation"),
            action_space=self.action_space,
            critic_hidden_size=self.critic_hidden_size,
            normalizer=self.normalizer,
            initializer=self.initializer,
            activation=self.activation,
        ))
        return config


class HybridActionValueCritic(Module):
    """
    Q(s,a_con, ·)
    """

    def __init__(self,
                 representation: Module,
                 action_space: gymnasium.spaces.Tuple,
                 critic_hidden_size: Sequence[int],
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[keras.initializers.Initializer] = None,
                 activation: Optional[Type[Module]] = None,
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
        self.critic_hidden_size = critic_hidden_size
        self.normalizer = normalizer
        self.initializer = initializer
        self.activation = activation

        self.representation_info_shape = representation.output_shapes

        self.critic_head = QValueHead(
            feature_dim=self.representation_info_shape['state'][0] + self.conact_size,
            hidden_size=critic_hidden_size,
            n_actions=self.num_disact,
            normalizer=normalizer,
            initializer=initializer,
            activation=activation,
            **kwargs,
        )

    def call(self,
             observation: Union[Tensor, dict],
             actions: Tensor,
             **kwargs) -> CriticOutput:
        rep_out = self.representation(observation, **kwargs)
        critic_input = tf.concat([rep_out.embeddings, actions], axis=1)
        return CriticOutput(
            representations=rep_out,
            values=self.critic_head(critic_input, **kwargs)
        )

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            representation=self.representation.clone(copy_weights=True, trainable=False,
                                                     name="target_critic_representation"),
            action_space=self.action_space,
            critic_hidden_size=self.critic_hidden_size,
            normalizer=self.normalizer,
            initializer=self.initializer,
            activation=self.activation,
        ))
        return config


class MeanFieldStateValueCritic(Module):
    """
    V(s, a_mean)
    """

    def __init__(self,
                 representation: Module,
                 mean_actions_encoder: Module,
                 critic_hidden_size: Sequence[int],
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[keras.initializers.Initializer] = None,
                 activation: Optional[Type[Module]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.representation = representation
        self.mean_actions_encoder = mean_actions_encoder
        self.critic_hidden_size = critic_hidden_size
        self.normalizer = normalizer
        self.initializer = initializer
        self.activation = activation

        self.representation_feature_dim = representation.output_shapes['state'][0]
        self.mean_action_feature_dim = mean_actions_encoder.output_shapes['state'][0]
        self.critic_head = ValueHead(
            feature_dim=self.representation_feature_dim + self.mean_action_feature_dim,
            hidden_size=critic_hidden_size,
            normalizer=normalizer,
            initializer=initializer,
            activation=activation,
            **kwargs,
        )

    def call(self,
             observation: Union[Tensor, dict],
             mean_actions: Tensor,
             **kwargs) -> CriticOutput:
        rep_out = self.representation(observation, **kwargs)
        mean_actions_rep_out = self.mean_actions_encoder(mean_actions, **kwargs)
        critic_input = tf.concat([rep_out.embeddings, mean_actions_rep_out.embeddings], axis=-1)
        return CriticOutput(
            representations=rep_out,
            values=self.critic_head(critic_input, **kwargs)
        )

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            representation=self.representation.clone(copy_weights=True, trainable=False,
                                                     name="target_critic_representation"),
            mean_actions_encoder=self.mean_actions_encoder.clone(copy_weights=True, trainable=False,
                                                                 name="target_critic_mean_actions_encoder"),
            critic_hidden_size=self.critic_hidden_size,
            normalizer=self.normalizer,
            initializer=self.initializer,
            activation=self.activation,
        ))
        return config


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
                 initializer: Optional[keras.initializers.Initializer] = None,
                 activation: Optional[Type[Module]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(action_space, Discrete):
            self.action_space = action_space
            self.n_actions = action_space.n
        else:
            raise ValueError('action_space must be Discrete.')
        self.representation = representation
        self.mean_actions_encoder = mean_actions_encoder
        self.critic_hidden_size = critic_hidden_size
        self.normalizer = normalizer
        self.initializer = initializer
        self.activation = activation

        self.representation_feature_dim = representation.output_shapes['state'][0]
        self.mean_action_feature_dim = mean_actions_encoder.output_shapes['state'][0]

        self.critic_head = QValueHead(
            feature_dim=self.representation_feature_dim + self.mean_action_feature_dim,
            hidden_size=critic_hidden_size,
            n_actions=self.n_actions,
            normalizer=normalizer,
            initializer=initializer,
            activation=activation,
            **kwargs,
        )

    def call(self,
             observations: Tensor,
             mean_actions: Tensor,
             **kwargs) -> CriticOutput:
        rep_out = self.representation(observations, **kwargs)
        mean_actions_rep_out = self.mean_actions_encoder(mean_actions, **kwargs)
        critic_input = tf.concat([rep_out.embeddings, mean_actions_rep_out.embeddings], axis=-1)
        return CriticOutput(
            representations=rep_out,
            values=self.critic_head(critic_input, **kwargs)
        )

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            representation=self.representation.clone(copy_weights=True, trainable=False,
                                                     name="target_critic_representation"),
            mean_actions_encoder=self.mean_actions_encoder.clone(copy_weights=True, trainable=False,
                                                                 name="target_critic_mean_actions_encoder"),
            action_space=self.action_space,
            critic_hidden_size=self.critic_hidden_size,
            normalizer=self.normalizer,
            initializer=self.initializer,
            activation=self.activation,
        ))
        return config
