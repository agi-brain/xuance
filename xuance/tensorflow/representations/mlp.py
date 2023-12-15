from xuance.tensorflow.representations import *


class Basic_Identical(tk.Model):
    def __init__(self,
                 input_shape: Sequence[int],
                 device: str = "cpu"):
        super(Basic_Identical, self).__init__()
        self.input_shapes = input_shape
        self.output_shapes = {'state': (np.prod(input_shape),)}
        self.device = device
        self.model = tk.Sequential([tk.layers.Flatten()])

    def call(self, observations: np.ndarray, **kwargs):
        with tf.device(self.device):
            state = tf.convert_to_tensor(observations, dtype=tf.float32)
            return {'state': state}


class Basic_MLP(tk.Model):
    def __init__(self,
                 input_shapes: Sequence[int],
                 hidden_sizes: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu"):
        super(Basic_MLP, self).__init__()
        self.input_shapes = input_shapes
        self.hidden_sizes = hidden_sizes
        self.normalize = normalize
        self.initializer = initializer
        self.activation = activation
        self.device = device
        self.output_shapes = {'state': (hidden_sizes[-1],)}
        self.model = self._create_network()

    def _create_network(self):
        layers = [tk.layers.Flatten()]
        input_shapes = (np.prod(self.input_shapes),)
        for h in self.hidden_sizes:
            mlp, input_shapes = mlp_block(input_shapes[0], h, self.normalize, self.activation, self.initializer,
                                          self.device)
            layers.extend(mlp)
        return tk.Sequential(layers)

    def call(self, observations: np.ndarray, **kwargs):
        with tf.device(self.device):
            tensor_observation = tf.convert_to_tensor(observations, dtype=tf.float32)
            return {'state': self.model(tensor_observation)}

