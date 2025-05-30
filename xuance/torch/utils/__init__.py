from .layers import (
    torch, nn,
    ModuleType,
    mlp_block, cnn_block, pooling_block, gru_block, lstm_block,
    Moments,
)
from .distributions import (
    kl_div,
    split_distributions, merge_distributions,
    Distribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    ActivatedDiagGaussianDistribution,
    SymLogDistribution,
)
from .operations import (init_distributed_mode, update_linear_decay, set_seed,
                         get_flat_grad, get_flat_params, assign_from_flat_grads,
                         assign_from_flat_params,
                         init_weights, uniform_init_weights, sym_log, sym_exp,
                         two_hot_encoder, two_hot_decoder, compute_stochastic_state, compute_lambda_values,
                         dotdict)
from .value_norm import ValueNorm

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
    "orthogonal": torch.nn.init.orthogonal_,
    "normal": torch.nn.init.normal_,
    "zeros": torch.nn.init.zeros_,
    "ones": torch.nn.init.ones_,
}
