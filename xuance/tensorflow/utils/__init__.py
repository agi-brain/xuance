from .layers import tk, ModuleType, mlp_block, cnn_block, pooling_block, gru_block, lstm_block
from .distributions import Distribution, CategoricalDistribution, DiagGaussianDistribution
from .operations import update_linear_decay, set_seed, get_flat_params, assign_from_flat_params, assign_from_flat_grads, \
    merge_distributions, split_distributions
from .value_norm import ValueNorm

ActivationFunctions = {
    "ReLU": tk.layers.Activation('relu'),
    "LeakyReLU": tk.layers.Activation('leaky_relu'),
    "Tanh": tk.layers.Activation('tanh'),
    "Sigmoid": tk.layers.Activation('sigmoid'),
    "Softmax": tk.layers.Activation('softmax'),
    "Elu": tk.layers.Activation('elu'),
}

NormalizeFunctions = {
    "LayerNorm": tk.layers.LayerNormalization,
    "BatchNorm": tk.layers.BatchNormalization,
    "BatchNorm2d": tk.layers.BatchNormalization,
}

InitializeFunctions = {
    "orthogonal": tk.initializers.orthogonal
}
