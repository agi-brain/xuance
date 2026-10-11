import mindspore.nn as nn
from mindspore.common.initializer import Normal, One, Zero, Orthogonal

from .data import AgentGroupedTensor
from .device import set_device, collect_device_info
from .operations import (update_linear_decay, set_seed,
                         get_flat_grad, get_flat_params,
                         assign_from_flat_params, assign_from_flat_grads,
                         clip_grads)
from .value_norm import ValueNorm
from .tensor_memory import (TensorOnPolicyBuffer, TensorOnPolicyBufferAtari,
                            TensorOffPolicyBuffer, TensorOffPolicyBufferAtari)
from .tensor_env import TensorEnvWrapper, TensorMultiAgentEnvWrapper
from .tensor_statistics import TensorRunningMeanStd


class LeakyReLU_01(nn.LeakyReLU):
    def __init__(self):
        super(LeakyReLU_01, self).__init__(alpha=0.01)


ActivationFunctions = {
    "relu": nn.ReLU,
    "leaky_relu": LeakyReLU_01,  # nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "softmax2d": nn.Softmax2d,
    "elu": nn.ELU,
    "None": None
}

NormalizeFunctions = {
    "LayerNorm": nn.LayerNorm,
    "GroupNorm": nn.GroupNorm,
    "BatchNorm": nn.BatchNorm1d,
    "BatchNorm2d": nn.BatchNorm2d,
    "InstanceNorm2d": nn.InstanceNorm2d
}


InitializeFunctions = {
    "orthogonal": Orthogonal,
    "normal": Normal,
    "ones": One,
    "zeros": Zero,
}
