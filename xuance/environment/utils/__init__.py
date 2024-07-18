from xuance.common import Dict, Any
from .wrapper import XuanCeEnvWrapper, XuanCeMultiAgentEnvWrapper
from .base import RawEnvironment, RawMultiAgentEnv, MultiAgentDict, AgentKeys


EnvName = Any
EnvObject = Any
EnvironmentDict = Dict[EnvName, EnvObject]


__all__ = [
    "RawEnvironment",
    "RawMultiAgentEnv",
    "XuanCeEnvWrapper",
    "XuanCeMultiAgentEnvWrapper",
    "EnvironmentDict",
    "MultiAgentDict",
    "AgentKeys",
]
