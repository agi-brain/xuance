import mindspore as ms
import mindspore.nn as nn
from typing import Sequence, Optional, Union, Callable
import numpy as np
from xuance.mindspore.utils.layers import *

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
    "Basic_MLP": ["input_shape", "hidden_sizes", "normalize", "initialize", "activation"],
    "Basic_Identical": ["input_shape"],
    "Basic_CNN": ["input_shape", "kernels", "strides", "filters", "normalize", "initialize", "activation"],
}

Representation_Inputs_All = {
    "input_shape": None,
    "kernels": None,
    "strides": None,
    "filters": None,
    "hidden_sizes": None,
    "normalize": None,
    "initialize": None,
    "activation": None
}
