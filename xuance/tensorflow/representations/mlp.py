import numpy as np
from xuance.common import Sequence, Optional, Union
from xuance.tensorflow import tf, tk, Module, Tensor
from xuance.tensorflow.utils.layers import mlp_block
from xuance.tensorflow.utils import ModuleType


# directly returns the original observation
class Basic_Identical(Module):
    def __init__(self,
                 input_shape: Sequence[int], **kwargs):
        super(Basic_Identical, self).__init__()
        self.input_shapes = input_shape
        self.output_shapes = {'state': (np.prod(input_shape),)}
        self.model = tk.Sequential([tk.layers.Flatten()])

    @tf.function
    def call(self, x: Union[Tensor, np.ndarray], **kwargs):
        return {'state': self.model(x)}


class Basic_MLP(Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[ModuleType] = None,
                 **kwargs):
        super(Basic_MLP, self).__init__()
        self.input_shapes = input_shape
        self.hidden_sizes = hidden_sizes
        self.normalize = normalize
        self.initializer = initializer
        self.activation = activation
        self.output_shapes = {'state': (hidden_sizes[-1],)}
        self.model = self._create_network()

    def _create_network(self):
        layers = [tk.layers.Flatten()]
        input_shape = self.input_shapes
        for h in self.hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, self.normalize, self.activation, self.initializer)
            layers.extend(mlp)
        return tk.Sequential(layers)

    @tf.function
    def call(self, x: Union[Tensor, np.ndarray], **kwargs):
        return {'state': self.model(x)}
