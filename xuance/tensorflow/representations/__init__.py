import tensorflow as tf
import numpy as np
from typing import Sequence, Union, Optional, Callable
from xuance.tensorflow.utils.layers import *

from .networks import Basic_Identical, Basic_MLP, Basic_CNN
from .networks import CoG_MLP, CoG_RNN, CoG_CNN
from .networks import C_DQN, L_DQN, CL_DQN

REGISTRY = {
    "Basic_MLP": Basic_MLP,
    "Basic_Identical": Basic_Identical,
    "Basic_CNN": Basic_CNN,
    "CoG_MLP": CoG_MLP,
    "CoG_RNN": CoG_RNN,
    "CoG_CNN": CoG_CNN,
    "C_DQN": C_DQN,
    "L_DQN": L_DQN,
    "CL_DQN": CL_DQN,
}

Representation_Inputs = {
    "Basic_MLP": ["input_shape", "hidden_sizes", "normalize", "initialize", "activation", "device"],
    "Basic_Identical": ["input_shape", "device"],
    "Basic_CNN": ["input_shape", "kernels", "strides", "filters", "normalize", "initialize", "activation", "device"],
    "CoG_MLP": ["input_shape", "hidden_sizes", "normalize", "initialize", "activation", "device"],
    "CoG_RNN": ["input_shape", "normalize", "initialize", "activation", "device"],
    "CoG_CNN": ["input_shape", "kernels", "strides", "filters", "normalize", "initialize", "activation", "device"],
    "C_DQN": ["input_shape", "kernels", "strides", "filters", "normalize", "initialize", "activation", "device"],
    "L_DQN": ["input_shape", "hidden_sizes", "normalize", "initialize", "activation", "device"],
    "CL_DQN": ["input_shape", "kernels", "strides", "filters", "normalize", "initialize", "activation", "device"]
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
