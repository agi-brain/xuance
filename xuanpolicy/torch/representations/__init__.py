import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence, Union, Optional, Callable
from xuanpolicy.torch.utils.layers import *

from .networks import Basic_MLP
from .networks import Basic_Identical
from .networks import Basic_CNN, CNN_FC
from .networks import CoG_MLP, CoG_RNN, CoG_CNN

REGISTRY = {
    "Basic_MLP": Basic_MLP,
    "Basic_Identical": Basic_Identical,
    "Basic_CNN": Basic_CNN,
    "CNN_FC": CNN_FC,
    "CoG_MLP": CoG_MLP,
    "CoG_RNN": CoG_RNN,
    "CoG_CNN": CoG_CNN,
}

Representation_Inputs = {
    "Basic_MLP": ["input_shape", "hidden_sizes", "normalize", "initialize", "activation", "device"],
    "Basic_Identical": ["input_shape", "device"],
    "Basic_CNN": ["input_shape", "kernels", "strides", "filters", "normalize", "initialize", "activation", "device"],
    "CNN_FC": ["input_shape", "kernels", "strides", "filters", "normalize", "initialize", "activation", "device", "fc_hidden_sizes"],
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
