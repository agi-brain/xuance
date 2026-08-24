import keras
import tensorflow as tf
from typing import Type
from tensorflow import Tensor
from xuance.tensorflow.utils.module import Module, ModuleList, ModuleDict
from xuance.tensorflow.utils import set_device, collect_device_info

ModuleType = Type[Module]

from xuance.tensorflow.rl_models.representations import REGISTRY_Representation
from xuance.tensorflow.learners import REGISTRY_Learners
from xuance.tensorflow.agents import REGISTRY_Agents

__all__ = [
    "tf",
    "keras",
    "Tensor",
    "Module", "ModuleDict", "ModuleList",
    "ModuleType",
    "set_device", "collect_device_info",
    "REGISTRY_Representation", "REGISTRY_Learners", "REGISTRY_Agents"
]
