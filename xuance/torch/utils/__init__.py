import torch.nn as nn
from .data import AgentGroupedTensor
from .device import set_device, collect_device_info
from .operations import (init_distributed_mode, update_linear_decay, set_seed,
                         get_flat_grad, get_flat_params, assign_from_flat_grads,
                         assign_from_flat_params,
                         init_weights, uniform_init_weights, sym_log, sym_exp,
                         two_hot_encoder, two_hot_decoder, compute_stochastic_state, compute_lambda_values,
                         dotdict)
from .value_norm import ValueNorm
from .tensor_memory import (TensorOnPolicyBuffer, TensorOnPolicyBufferAtari,
                            TensorOffPolicyBuffer, TensorOffPolicyBufferAtari)
from .tensor_env import TensorEnvWrapper, TensorMultiAgentEnvWrapper
from .tensor_statistics import TensorRunningMeanStd

from typing import Type
ModuleType = Type[nn.Module]

ActivationFunctions = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "softmax2d": nn.Softmax2d,
    "elu": nn.ELU,
}

NormalizeFunctions = {
    "LayerNorm": nn.LayerNorm,
    "GroupNorm": nn.GroupNorm,
    "BatchNorm": nn.BatchNorm1d,
    "BatchNorm2d": nn.BatchNorm2d,
    "InstanceNorm2d": nn.InstanceNorm2d
}

InitializeFunctions = {
    "orthogonal": nn.init.orthogonal_,
    "normal": nn.init.normal_,
    "zeros": nn.init.zeros_,
    "ones": nn.init.ones_,
}
