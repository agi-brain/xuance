import numpy as np
from xuance.common import Sequence, Optional, Union
from xuance.tensorflow import keras, Tensor, Module, ModuleType
from xuance.tensorflow.rl_models.modules.layers import mlp_block
from xuance.tensorflow.rl_models.modules.outputs import RepresentationOutput


# directly returns the original observation
class Basic_Identical(Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 **kwargs):
        super(Basic_Identical, self).__init__()

        self.input_shapes = input_shape

        self.output_shapes = {'state': (np.prod(input_shape),)}
        self.model = keras.Sequential([keras.layers.Flatten()])

    def call(self, x: Union[Tensor, np.ndarray], **kwargs):
        embeddings = self.model(x)
        return RepresentationOutput(embeddings=embeddings)

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            input_shape=self.input_shapes
        ))
        return config


class Basic_MLP(Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: Sequence[int],
                 normalizer: Optional[ModuleType] = None,
                 initializer: Optional[keras.initializers.Initializer] = None,
                 activation: Optional[ModuleType] = None,
                 **kwargs):
        super(Basic_MLP, self).__init__(**kwargs)
        self.input_shapes = input_shape
        self.hidden_sizes = hidden_sizes
        self.normalizer = normalizer
        self.initializer = initializer
        self.activation = activation

        self.output_shapes = {'state': (hidden_sizes[-1],)}
        self.model = self._create_network()

    def _create_network(self):
        layers = [keras.layers.Flatten()]
        input_shape = self.input_shapes
        for h in self.hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, self.normalizer, self.activation, self.initializer)
            layers.extend(mlp)
        return keras.Sequential(layers)

    def call(self, x: Tensor, **kwargs):
        embeddings = self.model(x)
        return RepresentationOutput(embeddings=embeddings)

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            input_shape=self.input_shapes,
            hidden_sizes=self.hidden_sizes,
            normalizer=self.normalizer,
            initializer=self.initializer,
            activation=self.activation,
        ))
        return config
