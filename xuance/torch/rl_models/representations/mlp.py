import torch
import torch.nn as nn
from typing import Sequence, Optional, Union, Callable
from xuance.torch import Module, Tensor
from xuance.torch.rl_models.modules.layers import mlp_block, ModuleType
from xuance.torch.rl_models.modules.outputs import RepresentationOutput


# directly returns the original observation
class Basic_Identical(Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super(Basic_Identical, self).__init__()
        assert len(input_shape) == 1
        self.output_shapes = {'state': (input_shape[0],)}
        self.device = device

    def forward(self, observations: Tensor) -> RepresentationOutput:
        return RepresentationOutput(
            embeddings=torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        )


# process the input observations with stacks of MLP layers
class Basic_MLP(Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
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

    def forward(self, observations: Tensor, **kwargs) -> RepresentationOutput:
        tensor_observation = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        return RepresentationOutput(embeddings=self.model(tensor_observation))
