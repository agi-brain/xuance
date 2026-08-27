import numpy as np
from xuance.common import Sequence, Optional
from xuance.tensorflow import tf, keras, Tensor, Module
from xuance.tensorflow.rl_models.modules.layers import cnn_block, mlp_block
from xuance.tensorflow.rl_models.modules.outputs import RepresentationOutput


# Process the input observations with stacks of CNN layers.
class Basic_CNN(Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 kernels: Sequence[int],
                 strides: Sequence[int],
                 filters: Sequence[int],
                 normalizer: Optional[keras.Layer] = None,
                 initializer: Optional[keras.initializers.Initializer] = None,
                 activation: Optional[keras.Layer] = None,
                 **kwargs):
        super(Basic_CNN, self).__init__()
        self.input_shapes = (input_shape[0], input_shape[1], input_shape[2])  # Height x Width x Channels
        self.kernels = kernels
        self.strides = strides
        self.filters = filters
        self.normalizer = normalizer
        self.initializer = initializer
        self.activation = activation
        self.output_shapes = {'state': (filters[-1],)}
        self.model = self._create_network()

    def _create_network(self):
        layers = []
        input_shape = self.input_shapes

        for k, s, f in zip(self.kernels, self.strides, self.filters):
            cnn, input_shape = cnn_block(
                input_shape, f, k, s, self.normalizer, self.activation, self.initializer)
            layers.extend(cnn)

        layers.append(
            keras.layers.GlobalMaxPooling2D(data_format='channels_last')
        )
        return keras.Sequential(layers)

    def call(
            self,
            observations: Tensor,
            **kwargs
    ) -> RepresentationOutput:
        tensor_observation = tf.cast(
            observations,
            dtype=tf.float32
        )
        tensor_observation = tensor_observation / 255.0

        embeddings = self.model(tensor_observation)

        return RepresentationOutput(
            embeddings=embeddings
        )

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            input_shape=self.input_shapes,
            kernels=self.kernels,
            strides=self.strides,
            filters=self.filters,
            normalizer=self.normalizer,
            initializer=self.initializer,
            activation=self.activation,
        ))
        return config


class AC_CNN_Atari(Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 kernels: Sequence[int],
                 strides: Sequence[int],
                 filters: Sequence[int],
                 normalizer: Optional[keras.Layer] = None,
                 initializer: Optional[keras.initializers.Initializer] = None,
                 activation: Optional[keras.Layer] = None,
                 fc_hidden_sizes: Sequence[int] = (),
                 **kwargs):
        super(AC_CNN_Atari, self).__init__()
        self.input_shapes = (input_shape[0], input_shape[1], input_shape[2])  # Height x Width x Channels
        self.kernels = kernels
        self.strides = strides
        self.filters = filters
        self.normalizer = normalizer
        self.initializer = initializer
        self.activation = activation
        self.fc_hidden_sizes = fc_hidden_sizes

        self.output_shapes = {'state': (fc_hidden_sizes[-1],)}
        self.model = self._create_network()

    def _init_layer(
            self,
            layer,
            gain=np.sqrt(2),
            bias=0.0
    ):
        initializer = keras.initializers.Orthogonal(gain=gain)
        if hasattr(layer, "kernel"):
            layer.kernel.assign(initializer(layer.kernel.shape, dtype=layer.kernel.dtype, ))
        if hasattr(layer, "bias") and layer.bias is not None:
            layer.bias.assign(tf.fill(layer.bias.shape, tf.cast(bias, layer.bias.dtype, ), ))
        return layer

    def _create_network(self):
        layers = []
        input_shape = self.input_shapes

        for k, s, f in zip(self.kernels, self.strides, self.filters):
            cnn, input_shape = cnn_block(input_shape, f, k, s, None, self.activation, None)
            cnn[0] = self._init_layer(cnn[0])
            layers.extend(cnn)

        layers.append(keras.layers.Flatten())

        input_shape = (np.prod(input_shape, dtype=np.int32),)

        for h in self.fc_hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, None, self.activation, None)
            mlp[0] = self._init_layer(mlp[0])
            layers.extend(mlp)
        return keras.Sequential(layers)

    def call(
            self,
            observations: Tensor
    ) -> RepresentationOutput:

        tensor_observation = tf.cast(
            observations,
            dtype=tf.float32,
        )

        tensor_observation = tensor_observation / 255.0

        embeddings = self.model(tensor_observation)

        return RepresentationOutput(
            embeddings=embeddings
        )

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            input_shape=self.input_shapes,
            kernels=self.kernels,
            strides=self.strides,
            filters=self.filters,
            normalizer=self.normalizer,
            initializer=self.initializer,
            activation=self.activation,
            fc_hidden_sizes=self.fc_hidden_sizes,
        ))
        return config
