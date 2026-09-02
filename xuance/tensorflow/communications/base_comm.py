import torch
import torch.nn as nn
from xuance.common import Optional, Callable, Union, Sequence
from xuance.tensorflow import Module, Tensor, ModuleType
from xuance.tensorflow.rl_models.modules import mlp_block


class BaseComm(Module):
    def __init__(self,
                 state_dim: int,
                 n_agents: int,
                 hidden_sizes_comm: Sequence[int],
                 msg_dim: int,
                 normalizer: Optional[ModuleType] = None,
                 initializer: Optional[keras.initializers.Initializer] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super().__init__()
        self.n_agents = n_agents
        self.msg_dim = msg_dim
        self.hidden_sizes_comm = hidden_sizes_comm
        layers_ = []
        input_shape = (state_dim,)
        for h in hidden_sizes_comm:
            mlp, input_shape = mlp_block(input_shape[0], h, normalizer, activation, initializer, device)
            layers_.extend(mlp)
        layers_.extend(mlp_block(input_shape[0], msg_dim, None, None, initializer, device)[0])
        self.msg_encoder = nn.Sequential(*layers_)

    def forward(self, hidden_features: Tensor):
        encoded_msg = self.msg_encoder(hidden_features)
        return encoded_msg


class NoneComm(Module):
    def __init__(self):
        super().__init__()

    def forward(self, msg: Tensor, **kwargs):
        return msg


