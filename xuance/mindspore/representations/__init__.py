from .networks import Basic_MLP
from .networks import Basic_Identical
from .networks import Basic_CNN
from .networks import CoG_MLP, CoG_RNN, CoG_CNN, C_DQN, L_DQN, CL_DQN

REGISTRY = {
    "Basic_MLP": Basic_MLP,
    "Basic_Identical": Basic_Identical,
    "Basic_CNN": Basic_CNN,
    "CoG_MLP": CoG_MLP,
    "CoG_RNN": CoG_RNN,
    "CoG_CNN": CoG_CNN,
    "C_DQN": C_DQN,
    "L_DQN": L_DQN
}

Representation_Inputs = {
    "Basic_MLP": ["input_shape", "hidden_sizes", "normalize", "initialize", "activation"],
    "Basic_Identical": ["input_shape"],
    "Basic_CNN": ["input_shape", "kernels", "strides", "filters", "normalize", "initialize", "activation"],
    "CoG_MLP": ["input_shape", "hidden_sizes", "normalize", "initialize", "activation"],
    "CoG_RNN": ["input_shape", "normalize", "initialize", "activation"],
    "CoG_CNN": ["input_shape", "kernels", "strides", "filters", "normalize", "initialize", "activation"],
    "C_DQN": ["input_shape", "kernels", "strides", "filters", "normalize", "initialize", "activation"],
    "L_DQN": ["input_shape", "normalize", "initialize", "activation"],
    "CL_DQN": ["input_shape", "kernels", "strides", "filters", "normalize", "initialize", "activation"],
}

Representation_Inputs_All = {
    "input_shape": None,
    "kernels": None,
    "strides": None,
    "filters": None,
    "hidden_sizes": None,
    "normalize": None,
    "initialize": None,
    "activation": None
}
