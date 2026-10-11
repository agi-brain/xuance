from typing import Type, Sequence, Optional, Callable

from xuance.mindspore import nn, Tensor, Module
from xuance.mindspore.rl_models.modules import mlp_block


class ValueHead(Module):
    def __init__(self,
                 feature_dim: int,
                 hidden_size: Sequence[int],
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        layers = []
        input_shape = (feature_dim,)
        for h in hidden_size:
            mlp, input_shape = mlp_block(input_shape[0], h, normalizer, activation, initializer)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, initializer)[0])
        self.values = nn.SequentialCell(*layers)

    def construct(self,
                  features: Tensor,
                  **kwargs) -> Tensor:
        return self.values(features).squeeze(-1)
