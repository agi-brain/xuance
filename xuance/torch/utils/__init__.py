from .layers import (
    torch, nn,
    ModuleType,
    mlp_block, cnn_block, pooling_block, gru_block, lstm_block
)
from .distributions import (
    Distribution,
    CategoricalDistribution,
    DiagGaussianDistribution
)
from .operations import (update_linear_decay, set_seed, get_flat_grad, get_flat_params, assign_from_flat_grads,
                         assign_from_flat_params, split_distributions, merge_distributions)
from .value_norm import ValueNorm

ActivationFunctions = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "Tanh": nn.Tanh,
    "Sigmoid": nn.Sigmoid,
    "Softmax": nn.Softmax,
    "Softmax2d": nn.Softmax2d,
    "Elu": nn.ELU,
}

NormalizeFunctions = {
    "LayerNorm": nn.LayerNorm,
    "GroupNorm": nn.GroupNorm,
    "BatchNorm": nn.BatchNorm1d,
    "BatchNorm2d": nn.BatchNorm2d,
    "InstanceNorm2d": nn.InstanceNorm2d
}

InitializeFunctions = {
    "orthogonal": torch.nn.init.orthogonal_
}
