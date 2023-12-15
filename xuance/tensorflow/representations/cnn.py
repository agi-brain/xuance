from xuance.tensorflow.representations import *


class Basic_CNN(tk.Model):
    def __init__(self,
                 input_shape: Sequence[int],
                 kernels: Sequence[int],
                 strides: Sequence[int],
                 filters: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu"):
        super(Basic_CNN, self).__init__()
        self.input_shape = (input_shape[2], input_shape[0], input_shape[1])
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

    def call(self, observations: np.ndarray, **kwargs):
        with tf.device(self.device):
            tensor_observation = tf.convert_to_tensor(np.transpose(observations, (0, 3, 1, 2)), dtype=tf.float32)
            return {'state': self.model(tensor_observation)}

