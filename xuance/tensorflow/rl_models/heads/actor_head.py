from typing import Type, Optional, Sequence
from xuance.tensorflow import tf, keras, Tensor, Module
from xuance.tensorflow.rl_models.modules import mlp_block
from xuance.tensorflow.rl_models.modules.distributions import (CategoricalDistribution,
                                                               DiagGaussianDistribution,
                                                               ActivatedDiagGaussianDistribution)


class CategoricalActorHead(Module):
    def __init__(self,
                 feature_dim: int,
                 hidden_size: Sequence[int],
                 action_dim: int,
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[keras.initializers.Initializer] = None,
                 activation: Optional[Type[Module]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.normalizer = normalizer
        self.initializer = initializer
        self.activation = activation

        layers = []
        input_shape = (feature_dim,)
        for h in hidden_size:
            mlp, input_shape = mlp_block(input_shape[0], h, normalizer, activation, initializer)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, None, initializer)[0])
        self.logits = keras.Sequential(layers)
        self.policy_distribution = CategoricalDistribution(action_dim=action_dim)

    def call(self,
             features: Tensor,
             avail_actions: Optional[Tensor] = None,
             **kwargs):
        logits = self.logits(features)
        if avail_actions is not None:
            logits = tf.where(tf.equal(avail_actions, 0), tf.cast(-1e10, logits.dtype), logits)
        self.policy_distribution.set_param(logits=logits)
        return self.policy_distribution

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            feature_dim=self.feature_dim,
            hidden_size=self.hidden_size,
            action_dim=self.action_dim,
            normalizer=self.normalizer,
            initializer=self.initializer,
            activation=self.activation,
            **self.kwargs
        ))
        return config


class GaussianActorHead(Module):
    def __init__(self,
                 feature_dim: int,
                 hidden_size: Sequence[int],
                 action_dim: int,
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[keras.initializers.Initializer] = None,
                 activation: Optional[Type[Module]] = None,
                 activation_action: Optional[Type[Module]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.normalizer = normalizer
        self.initializer = initializer
        self.activation = activation
        self.activation_action = activation_action

        layers = []
        input_shape = (feature_dim,)
        for h in hidden_size:
            mlp, input_shape = mlp_block(input_shape[0], h, normalizer, activation, initializer)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, activation_action, initializer)[0])
        self.mu = keras.Sequential(layers)
        self.log_std = self.add_weight(
            name="log_std",
            shape=(action_dim,),
            initializer=keras.initializers.Constant(-1.0),
            trainable=True,
            dtype=tf.float32,
        )
        self.policy_distribution = DiagGaussianDistribution(action_dim)

    def call(self,
             features: Tensor,
             avail_actions: Optional[Tensor] = None,
             **kwargs):
        mu = self.mu(features)
        std = tf.exp(self.log_std)
        self.policy_distribution.set_param(mu, std)
        return self.policy_distribution

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            feature_dim=self.feature_dim,
            hidden_size=self.hidden_size,
            action_dim=self.action_dim,
            normalizer=self.normalizer,
            initializer=self.initializer,
            activation=self.activation,
            activation_action=self.activation_action
        ))
        return config


class SAC_GaussianActorHead(Module):
    def __init__(self,
                 feature_dim: int,
                 hidden_size: Sequence[int],
                 action_dim: int,
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[keras.initializers.Initializer] = None,
                 activation: Optional[Type[Module]] = None,
                 activation_action: Optional[Type[Module]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.normalizer = normalizer
        self.initializer = initializer
        self.activation = activation
        self.activation_action = activation_action

        layers = []
        input_shape = (feature_dim,)
        for h in hidden_size:
            mlp, input_shape = mlp_block(input_shape[0], h, normalizer, activation, initializer)
            layers.extend(mlp)
        self.output = keras.Sequential(layers)

        mu_layer, _ = mlp_block(input_shape[0], action_dim, None, None, initializer)
        log_std_layer, _ = mlp_block(input_shape[0], action_dim, None, None, initializer)

        self.out_mu = keras.Sequential(mu_layer)
        self.out_log_std = keras.Sequential(log_std_layer)

        self.policy_distribution = ActivatedDiagGaussianDistribution(action_dim, activation_action)

    def call(self,
             features: Tensor,
             avail_actions: Optional[Tensor] = None,
             **kwargs):
        output = self.output(features)
        mu = self.out_mu(output)
        log_std = tf.clip_by_value(self.out_log_std(output), clip_value_min=-20.0, clip_value_max=2.0)
        self.policy_distribution.set_param(mu, tf.exp(log_std))
        return self.policy_distribution

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            feature_dim=self.feature_dim,
            hidden_size=self.hidden_size,
            action_dim=self.action_dim,
            normalizer=self.normalizer,
            initializer=self.initializer,
            activation=self.activation,
            activation_action=self.activation_action
        ))


class DeterministicActorHead(Module):
    def __init__(self,
                 feature_dim: int,
                 hidden_size: Sequence[int],
                 action_dim: int,
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[keras.initializers.Initializer] = None,
                 activation: Optional[Type[Module]] = None,
                 activation_action: Optional[Type[Module]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.normalizer = normalizer
        self.initializer = initializer
        self.activation = activation
        self.activation_action = activation_action

        layers = []
        input_shape = (feature_dim,)
        for h in hidden_size:
            mlp, input_shape = mlp_block(input_shape[0], h, normalizer, activation, initializer)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, activation_action, initializer)[0])
        self.model = keras.Sequential(layers)

    def call(self,
             features: Tensor,
             avail_actions: Optional[Tensor] = None,
             **kwargs):
        actions = self.model(features)
        if avail_actions is not None:
            actions = tf.where(tf.equal(avail_actions, 0), tf.zeros_like(actions), actions)
        return actions

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            feature_dim=self.feature_dim,
            hidden_size=self.hidden_size,
            action_dim=self.action_dim,
            normalizer=self.normalizer,
            initializer=self.initializer,
            activation=self.activation,
            activation_action=self.activation_action
        ))
        return config
