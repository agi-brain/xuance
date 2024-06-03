import torch
import torch.nn as nn
import numpy as np
from typing import Sequence, Optional, Union, Callable
from xuance.torch.utils.layers import mlp_block
from xuance.torch.utils import ModuleType


# directly returns the original observation
class Basic_Identical(nn.Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 device: Optional[Union[str, int, torch.device]] = None):
        super(Basic_Identical, self).__init__()
        assert len(input_shape) == 1
        self.output_shapes = {'state': (input_shape[0],)}
        self.device = device

    def forward(self, observations: np.ndarray):
        state = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        return {'state': state}


# process the input observations with stacks of MLP layers
class Basic_MLP(nn.Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(Basic_MLP, self).__init__()
        self.input_shape = input_shape
        self.hidden_sizes = hidden_sizes
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.device = device
        self.output_shapes = {'state': (hidden_sizes[-1],)}
        self.model = self._create_network()

    def _create_network(self):
        layers = []
        input_shape = self.input_shape
        for h in self.hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, self.normalize, self.activation, self.initialize,
                                         device=self.device)
            layers.extend(mlp)
        return nn.Sequential(*layers)

    def forward(self, observations: np.ndarray):
        tensor_observation = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        return {'state': self.model(tensor_observation)}
