import argparse
import torch
import torch.nn as nn
from .util import init
from typing import List, Tuple, Union

"""CNN Modules and utils."""


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNNLayer(nn.Module):
    def __init__(
        self,
        obs_shape: Union[List, Tuple],
        hidden_size: int,
        use_orthogonal: bool,
        use_ReLU: bool,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        """
        obs_shape: Union[List, Tuple]
            The shape of the input. NOTE: Should be of length equal to 3
            (channels, width, height)
        hidden_size: int
            Hidden layer size of the linear layer after CNN
        use_orthogonal: bool
            Whether to use orthogonal weight init or Xavier uniform
        use_ReLU: bool
            Whether to use ReLU or tanh non-linear activations
        kernel_size: int
            The kernel size of the CNN filter. Default=3
        stride: int
            The stride for the CNN filters. Default=1
        """
        super(CNNLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(["tanh", "relu"][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]

        self.cnn = nn.Sequential(
            init_(
                nn.Conv2d(
                    in_channels=input_channel,
                    out_channels=hidden_size // 2,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            ),
            active_func,
            Flatten(),
            init_(
                nn.Linear(
                    hidden_size
                    // 2
                    * (input_width - kernel_size + stride)
                    * (input_height - kernel_size + stride),
                    hidden_size,
                )
            ),
            active_func,
            init_(nn.Linear(hidden_size, hidden_size)),
            active_func,
        )

    def forward(self, x: torch.tensor):
        x = x / 255.0
        x = self.cnn(x)
        return x


class CNNBase(nn.Module):
    def __init__(self, args: argparse.Namespace, obs_shape: Union[Tuple, List]):
        super(CNNBase, self).__init__()

        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self.hidden_size = args.hidden_size

        self.cnn = CNNLayer(
            obs_shape, self.hidden_size, self._use_orthogonal, self._use_ReLU
        )

    def forward(self, x):
        x = self.cnn(x)
        return x
