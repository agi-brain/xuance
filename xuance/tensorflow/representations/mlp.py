import numpy as np
from xuance.common import Sequence, Optional
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.utils.layers import mlp_block
from xuance.tensorflow.utils import ModuleType


# directly returns the original observation
class Basic_Identical(Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 device: Optional[str] = None):
        super(Basic_Identical, self).__init__()
        self.input_shapes = input_shape
        self.output_shapes = {'state': (np.prod(input_shape),)}
        self.device = device
        self.model = tk.Sequential([tk.layers.Flatten()])

    def call(self, observations: np.ndarray, **kwargs):
        with tf.device(self.device):
            state = tf.convert_to_tensor(observations, dtype=tf.float32)
            return {'state': state}


class Basic_MLP(Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[str] = None,
                 **kwargs):
        super(Basic_MLP, self).__init__()
        self.input_shapes = input_shape
        self.hidden_sizes = hidden_sizes
        self.normalize = normalize
        self.initializer = initializer
        self.activation = activation
        self.device = device
        self.output_shapes = {'state': (hidden_sizes[-1],)}
        self.model = self._create_network()

    def _create_network(self):
        layers = [tk.layers.Flatten()]
        input_shape = self.input_shapes
        for h in self.hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, self.normalize, self.activation, self.initializer,
                                         device=self.device)
            layers.extend(mlp)
        return tk.Sequential(layers)

    def call(self, observations: np.ndarray, **kwargs):
        with tf.device(self.device):
            tensor_observation = tf.convert_to_tensor(observations, dtype=tf.float32)
            return {'state': self.model(tensor_observation)}
