import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence, Union, Optional, Callable
from xuance.torch.utils.layers import *

from .mlp import Basic_Identical, Basic_MLP
from .cnn import Basic_CNN, AC_CNN_Atari
from .rnn import Basic_RNN

REGISTRY = {
    "Basic_Identical": Basic_Identical,
    "Basic_MLP": Basic_MLP,
    "Basic_CNN": Basic_CNN,
    "AC_CNN_Atari": AC_CNN_Atari,
    "Basic_RNN": Basic_RNN

}

Representation_Inputs = {
    "Basic_MLP": ["input_shape", "hidden_sizes", "normalize", "initialize", "activation", "device"],
    "Basic_RNN": ["input_shape", "hidden_sizes", "normalize", "initialize", "activation", "device"],
    "Basic_Identical": ["input_shape", "device"],
    "Basic_CNN": ["input_shape", "kernels", "strides", "filters", "normalize", "initialize", "activation", "device"],
    "AC_CNN_Atari": ["input_shape", "kernels", "strides", "filters", "normalize", "initialize", "activation", "device",
                     "fc_hidden_sizes"],
    "CoG_MLP": ["input_shape", "hidden_sizes", "normalize", "initialize", "activation", "device"],
    "CoG_RNN": ["input_shape", "normalize", "initialize", "activation", "device"],
    "CoG_CNN": ["input_shape", "kernels", "strides", "filters", "normalize", "initialize", "activation", "device"],
}

Representation_Inputs_All = {
    "input_shape": None,
    "kernels": None,
    "strides": None,
    "filters": None,
    "hidden_sizes": None,
    "normalize": None,
    "initialize": None,
    "activation": None,
    "device": None
}
