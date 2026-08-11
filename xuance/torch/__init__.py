import torch.nn as nn
from typing import Type
from torch import Tensor
from torch.nn import Module, ModuleDict, ModuleList
from torch.nn.parallel import DistributedDataParallel
from xuance.torch.utils import set_device, collect_device_info
from xuance.torch.rl_models.representations import REGISTRY_Representation
from xuance.torch.learners import REGISTRY_Learners
from xuance.torch.agents import REGISTRY_Agents

ModuleType = Type[nn.Module]

__all__ = [
    "nn",
    "Tensor",
    "Module",
    "ModuleDict",
    "ModuleList",
    "ModuleType",
    "DistributedDataParallel",
    "set_device", "collect_device_info",
    "REGISTRY_Representation", "REGISTRY_Policy", "REGISTRY_Learners", "REGISTRY_Agents",
]
