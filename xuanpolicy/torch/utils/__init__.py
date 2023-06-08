from .operations import *
from .layers import *
from .distributions import *

ActivatioFunctions = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "Tanh": nn.Tanh,
    "Sigmoid": nn.Sigmoid,
    "Softmax": nn.Softmax,
    "Softmax2d": nn.Softmax2d,
    "Elu": nn.ELU,
}
