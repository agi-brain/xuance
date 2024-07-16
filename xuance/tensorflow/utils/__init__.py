from .layers import tk, ModuleType, mlp_block, cnn_block, pooling_block, gru_block, lstm_block

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
