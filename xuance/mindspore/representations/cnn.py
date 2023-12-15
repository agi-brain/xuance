from xuance.mindspore.representations import *


# process the input observations with stacks of CNN layers
class Basic_CNN(nn.Cell):
    def __init__(self,
                 input_shape: Sequence[int],
                 kernels: Sequence[int],
                 strides: Sequence[int],
                 filters: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(Basic_CNN, self).__init__()
        self.input_shape = (input_shape[2], input_shape[0], input_shape[1])
        self.kernels = kernels
        self.strides = strides
        self.filters = filters
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.output_shapes = {'state': (filters[-1],)}
        self._transpose = ms.ops.Transpose()
        self.model = self._create_network()

    def _create_network(self):
        layers = []
        input_shape = self.input_shape
        for k, s, f in zip(self.kernels, self.strides, self.filters):
            cnn, input_shape = cnn_block(input_shape, f, k, s, self.normalize, self.activation, self.initialize)
            layers.extend(cnn)
        layers.append(nn.AdaptiveMaxPool2d((1, 1)))
        layers.append(nn.Flatten())
        return nn.SequentialCell(*layers)

    def construct(self, observations: ms.tensor):
        tensor_observation = self._transpose(observations, (0, 3, 1, 2)).astype("float32")
        return {'state': self.model(tensor_observation)}


class AC_CNN_Atari(nn.Cell):
    def __init__(self,
                 input_shape: Sequence[int],
                 kernels: Sequence[int],
                 strides: Sequence[int],
                 filters: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 fc_hidden_sizes: Sequence[int] = ()):
        super(AC_CNN_Atari, self).__init__()
        self.input_shape = (input_shape[2], input_shape[0], input_shape[1])  # Channels x Height x Width
        self.kernels = kernels
        self.strides = strides
        self.filters = filters
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.fc_hidden_sizes = fc_hidden_sizes
        self.output_shapes = {'state': (fc_hidden_sizes[-1],)}
        self.model = self._create_network()

    def _init_layer(self, layer, gain=np.sqrt(2), bias=0.0):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, bias)
        return layer

    def _create_network(self):
        layers = []
        input_shape = self.input_shape
        for k, s, f in zip(self.kernels, self.strides, self.filters):
            cnn, input_shape = cnn_block(input_shape, f, k, s, None, self.activation, None)
            cnn[0] = self._init_layer(cnn[0])
            layers.extend(cnn)
        layers.append(nn.Flatten())
        input_shape = (np.prod(input_shape, dtype=np.int), )
        for h in self.fc_hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, None, self.activation, None)
            mlp[0] = self._init_layer(mlp[0])
            layers.extend(mlp)
        return nn.SequentialCell(*layers)

    def construct(self, observations: np.ndarray):
        observations = observations / 255.0
        tensor_observation = ms.tensor(np.transpose(observations, (0, 3, 1, 2))).astype(ms.float32)
        return {'state': self.model(tensor_observation)}
