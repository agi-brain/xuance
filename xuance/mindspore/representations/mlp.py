import numpy as np
from xuance.common import Sequence, Optional, Callable
from xuance.mindspore import Module, Tensor
from xuance.mindspore.utils import nn, mlp_block, ModuleType


# directly returns the original observation
class Basic_Identical(Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 **kwargs):
        super(Basic_Identical, self).__init__()
        assert len(input_shape) == 1
        self.output_shapes = {'state': (input_shape[0],)}

    def construct(self, observations: np.ndarray):
        state = Tensor(observations)
        return {'state': state}


# process the input observations with stacks of MLP layers
class Basic_MLP(Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 **kwargs):
        super(Basic_MLP, self).__init__()
        self.input_shape = input_shape
        self.hidden_sizes = hidden_sizes
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.output_shapes = {'state': (hidden_sizes[-1],)}
        self.model = self._create_network()

    def _create_network(self):
        layers = []
        input_shape = self.input_shape
        for h in self.hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, self.normalize, self.activation, self.initialize)
            layers.extend(mlp)
        return nn.SequentialCell(*layers)

    def construct(self, observations: np.ndarray):
        tensor_observation = Tensor(observations)
        return {'state': self.model(tensor_observation)}
