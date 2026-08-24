import keras
import tensorflow as tf
from .module import Module, ModuleList, ModuleDict
from .data import AgentGroupedTensor
from .device import set_device, collect_device_info
from .operations import (update_linear_decay,
                         set_seed,
                         get_flat_params,
                         assign_from_flat_params,
                         assign_from_flat_grads)
from .value_norm import ValueNorm
from .tensor_memory import (TensorOnPolicyBuffer, TensorOnPolicyBufferAtari,
                            TensorOffPolicyBuffer, TensorOffPolicyBufferAtari)
from .tensor_env import TensorEnvWrapper, TensorMultiAgentEnvWrapper
from .tensor_statistics import TensorRunningMeanStd

from typing import Type

ModuleType = Type[Module]

ActivationFunctions = {
    "relu": tf.nn.relu,
    "leaky_relu": lambda x: tf.nn.leaky_relu(x, alpha=0.01),
    "tanh": tf.nn.tanh,
    "sigmoid": tf.nn.sigmoid,
    "softmax": tf.nn.softmax,
    "elu": tf.nn.elu,
}

normalizerFunctions = {
    "LayerNorm": keras.layers.LayerNormalization,
    "GroupNorm": keras.layers.GroupNormalization,
    "BatchNorm": keras.layers.BatchNormalization,
    "BatchNorm2d": keras.layers.BatchNormalization,
}

initializerFunctions = {
    "orthogonal": keras.initializers.Orthogonal,
    "normal": keras.initializers.RandomNormal,
    "zeros": keras.initializers.Zeros,
    "ones": keras.initializers.Ones
}
