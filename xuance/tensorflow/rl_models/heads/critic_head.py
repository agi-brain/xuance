from typing import Type, Sequence, Optional
from xuance.tensorflow import tf, keras, Tensor, Module
from xuance.tensorflow.rl_models.modules import mlp_block


class ValueHead(Module):
    def __init__(self,
                 feature_dim: int,
                 hidden_size: Sequence[int],
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[keras.initializers.Initializer] = None,
                 activation: Optional[Type[Module]] = None,
                 **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.normalizer = normalizer
        self.initializer = initializer
        self.activation = activation

        layers = []
        input_shape = (feature_dim,)
        for h in hidden_size:
            mlp, input_shape = mlp_block(input_shape[0], h, normalizer, activation, initializer)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, initializer)[0])
        self.values = keras.Sequential(layers)

    def call(
            self,
            features: Tensor,
            **kwargs
    ) -> Tensor:
        values = self.values(features)
        return tf.squeeze(values, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            feature_dim=self.feature_dim,
            hidden_size=self.hidden_size,
            normalizer=self.normalizer,
            initializer=self.initializer,
            activation=self.activation
        ))
        return config
