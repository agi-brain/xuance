import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence, Union, Optional, Callable
from xuanpolicy.torch.utils.layers import *

from .mlp import Basic_Identical, Basic_MLP, CoG_MLP
from .cnn import Basic_CNN, CoG_CNN, AC_CNN_Atari
from .rnn import Basic_RNN, CoG_RNN

REGISTRY = {
    "Basic_MLP": Basic_MLP,
    "Basic_Identical": Basic_Identical,
    "Basic_CNN": Basic_CNN,
    "AC_CNN_Atari": AC_CNN_Atari,
    "CoG_MLP": CoG_MLP,
    "CoG_RNN": CoG_RNN,
    "CoG_CNN": CoG_CNN,
}

Representation_Inputs = {
    "Basic_MLP": ["input_shape", "hidden_sizes", "normalize", "initialize", "activation", "device"],
    "Basic_Identical": ["input_shape", "device"],
    "Basic_CNN": ["input_shape", "kernels", "strides", "filters", "normalize", "initialize", "activation", "device"],
    "AC_CNN_Atari": ["input_shape", "kernels", "strides", "filters", "normalize", "initialize", "activation", "device", "fc_hidden_sizes"],
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
