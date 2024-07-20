import numpy as np
from xuance.common import Sequence, Optional, Union, Callable
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.utils.layers import cnn_block, mlp_block
from xuance.tensorflow.utils import ModuleType


class Basic_CNN(Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 kernels: Sequence[int],
                 strides: Sequence[int],
                 filters: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: Optional[str] = None,
                 **kwargs):
        super(Basic_CNN, self).__init__()
        self.input_shapes = (input_shape[2], input_shape[0], input_shape[1])
        self.kernels = kernels
        self.strides = strides
        self.filters = filters
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.device = device
        self.output_shapes = {'state': (filters[-1],)}
        self.model = self._create_network()

    def _create_network(self):
        layers = []
        input_shape = self.input_shape
        for k, s, f in zip(self.kernels, self.strides, self.filters):
            cnn, input_shape = cnn_block(input_shape, f, k, s, self.normalize, self.activation, self.initialize,
                                         self.device)
            layers.extend(cnn)
        layers.append(tfa.layers.AdaptiveMaxPooling2D((1, 1)))
        layers.append(tk.layers.Flatten())
        return tk.Sequential(*layers)

    @tf.function
    def call(self, observations: np.ndarray, **kwargs):
        with tf.device(self.device):
            tensor_observation = tf.convert_to_tensor(np.transpose(observations, (0, 3, 1, 2)), dtype=tf.float32)
            return {'state': self.model(tensor_observation)}


class AC_CNN_Atari(Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 kernels: Sequence[int],
                 strides: Sequence[int],
                 filters: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: Optional[str] = None,
                 fc_hidden_sizes: Sequence[int] = (),
                 **kwargs):
        super(AC_CNN_Atari, self).__init__()
        self.input_shapes = (input_shape[2], input_shape[0], input_shape[1])  # Channels x Height x Width
        self.kernels = kernels
        self.strides = strides
        self.filters = filters
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.device = device
        self.fc_hidden_sizes = fc_hidden_sizes
        self.output_shapes = {'state': (fc_hidden_sizes[-1],)}
        self.model = self._create_network()

    def _init_layer(self, layer, gain=np.sqrt(2), bias=0.0):
        initializer = tf.keras.initializers.Orthogonal(gain=gain)
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, bias)
        return layer

    def _create_network(self):
        layers = []
        input_shape = self.input_shape
        for k, s, f in zip(self.kernels, self.strides, self.filters):
            cnn, input_shape = cnn_block(input_shape, f, k, s, None, self.activation, None, self.device)
            cnn[0] = self._init_layer(cnn[0])
            layers.extend(cnn)
        layers.append(nn.Flatten())
        input_shape = (np.prod(input_shape, dtype=np.int), )
        for h in self.fc_hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, None, self.activation, None, self.device)
            mlp[0] = self._init_layer(mlp[0])
            layers.extend(mlp)
        return nn.Sequential(*layers)

    def forward(self, observations: np.ndarray):
        observations = observations / 255.0
        tensor_observation = torch.as_tensor(np.transpose(observations, (0, 3, 1, 2)), dtype=torch.float32,
                                             device=self.device)
        return {'state': self.model(tensor_observation)}
