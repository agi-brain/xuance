import mindspore as ms
import mindspore.nn.probability.distribution as msd
from typing import Type
from mindspore import Tensor, ops, nn, Parameter, ParameterTuple
from mindspore import mutable
from mindspore.nn import Cell as Module
from mindspore.nn import CellDict as ModuleDict
from mindspore.nn import CellList as ModuleList
from mindspore.experimental import optim
from xuance.mindspore.utils import set_device, collect_device_info
from xuance.mindspore.rl_models.representations import REGISTRY_Representation
from xuance.mindspore.learners import REGISTRY_Learners
from xuance.mindspore.agents import REGISTRY_Agents

ModuleType = Type[Module]

__all__ = [
    "ms",
    "nn",
    "Parameter", "ParameterTuple",
    "mutable",
    "msd",
    "Tensor",
    "Module", "ModuleDict", "ModuleList",
    "ModuleType",
    "ops",
    "optim",
    "set_device", "collect_device_info",
    "REGISTRY_Representation", "REGISTRY_Learners", "REGISTRY_Agents"
]
