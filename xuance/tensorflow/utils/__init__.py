import tensorflow as tf
from .device import set_device, collect_device_info
from .layers import tk, ModuleType, mlp_block, cnn_block, pooling_block, gru_block, lstm_block
from .distributions import (merge_distributions, split_distributions,
                            Distribution, CategoricalDistribution,
                            DiagGaussianDistribution, ActivatedDiagGaussianDistribution)
from .operations import update_linear_decay, set_seed, get_flat_params, assign_from_flat_params, assign_from_flat_grads
from .value_norm import ValueNorm

ActivationFunctions = {
    "relu": tf.nn.relu,
    "leaky_relu": lambda x: tf.nn.leaky_relu(x, alpha=0.01),
    "tanh": tf.nn.tanh,
    "sigmoid": tf.nn.sigmoid,
    "softmax": tf.nn.softmax,
    "elu": tf.nn.elu,
}

NormalizeFunctions = {
    "LayerNorm": tk.layers.LayerNormalization,
    "BatchNorm": tk.layers.BatchNormalization,
    "BatchNorm2d": tk.layers.BatchNormalization,
}

InitializeFunctions = {
    "orthogonal": tk.initializers.Orthogonal
}
