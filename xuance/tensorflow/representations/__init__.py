import tensorflow as tf
import numpy as np
from typing import Sequence, Union, Optional, Callable
from xuance.tensorflow.utils.layers import *

from .mlp import Basic_Identical, Basic_MLP
from .cnn import Basic_CNN
from .rnn import *

REGISTRY = {
    "Basic_MLP": Basic_MLP,
    "Basic_Identical": Basic_Identical,

    "Basic_CNN": Basic_CNN,

    "Basic_RNN": None
}

Representation_Inputs = {
    "Basic_MLP": ["input_shape", "hidden_sizes", "normalize", "initialize", "activation", "device"],
    "Basic_Identical": ["input_shape", "device"],

    "Basic_CNN": ["input_shape", "kernels", "strides", "filters", "normalize", "initialize", "activation", "device"],
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
