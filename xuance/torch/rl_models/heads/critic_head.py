import torch
import torch.nn as nn
from typing import Type, Sequence, Optional, Callable, Union
from torch import Tensor
from torch.nn import Module
from xuance.torch.rl_models.modules import mlp_block


class ValueHead(Module):
    def __init__(self,
                 feature_dim: int,
                 hidden_size: Sequence[int],
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        layers = []
        input_shape = (feature_dim, )
        for h in hidden_size:
            mlp, input_shape = mlp_block(input_shape[0], h, normalizer, activation, initializer, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, initializer, device)[0])
        self.values = nn.Sequential(*layers)

    def forward(self,
                features: Tensor,
                **kwargs) -> Tensor:
        return self.values(features).squeeze(-1)
