from xuance.common import Dict, Any
from .wrapper import XuanCeEnvWrapper, XuanCeAtariEnvWrapper, XuanCeMultiAgentEnvWrapper
from .base import RawEnvironment, RawMultiAgentEnv, MultiAgentDict, AgentKeys


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
]
