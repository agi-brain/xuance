from xuance.common import Dict, Any
from .wrapper import XuanCeEnvWrapper, XuanCeAtariEnvWrapper, XuanCeMultiAgentEnvWrapper
from .base import RawEnvironment, RawMultiAgentEnv, MultiAgentDict, AgentKeys
from .shapes import space2shape, combined_shape


EnvName = Any
EnvObject = Any
EnvironmentDict = Dict[EnvName, EnvObject]


__all__ = [
    "RawEnvironment",
    "RawMultiAgentEnv",
    "XuanCeEnvWrapper",
    "XuanCeAtariEnvWrapper",
    "XuanCeMultiAgentEnvWrapper",
    "EnvironmentDict",
    "MultiAgentDict",
    "AgentKeys",
    "space2shape",
    "combined_shape"
]
