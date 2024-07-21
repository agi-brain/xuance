from .layers import tk, ModuleType, mlp_block, cnn_block, pooling_block, gru_block, lstm_block
from .distributions import Distribution, CategoricalDistribution, DiagGaussianDistribution, \
    ActivatedDiagGaussianDistribution
from .operations import update_linear_decay, set_seed, get_flat_params, assign_from_flat_params, assign_from_flat_grads, \
    merge_distributions, split_distributions
from .value_norm import ValueNorm

ActivationFunctions = {
    "relu": tk.layers.Activation('relu'),
    "leaky_relu": tk.layers.Activation('leaky_relu'),
    "tanh": tk.layers.Activation('tanh'),
    "sigmoid": tk.layers.Activation('sigmoid'),
    "softmax": tk.layers.Activation('softmax'),
    "elu": tk.layers.Activation('elu'),
}

NormalizeFunctions = {
    "LayerNorm": tk.layers.LayerNormalization,
    "BatchNorm": tk.layers.BatchNormalization,
    "BatchNorm2d": tk.layers.BatchNormalization,
}

InitializeFunctions = {
    "orthogonal": tk.initializers.orthogonal
}
