from typing import Dict, Any
from xuance.environment.utils.wrapper import XuanCeEnvWrapper, XuanCeMultiAgentEnvWrapper
from xuance.environment.utils.base import RawEnvironment, RawMultiAgentEnv, MultiAgentDict, AgentKeys


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
