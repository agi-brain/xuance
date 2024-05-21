from typing import Dict, Any
from xuance.environment.utils.new import RawEnvironment, RawMultiAgentEnv
from xuance.environment.utils.wrapper import XuanCeEnvWrapper, XuanCeMultiAgentEnvWrapper
from xuance.environment.utils.base import MakeEnvironment, MakeMultiAgentEnvironment

from xuance.environment.utils.new import MultiAgentDict, AgentKeys

EnvName = Any
EnvObject = Any
EnvironmentDict = Dict[EnvName, EnvObject]


__all__ = [
    "RawEnvironment",
    "RawMultiAgentEnv",
    "XuanCeEnvWrapper",
    "XuanCeMultiAgentEnvWrapper",
    "MakeEnvironment",
    "MakeMultiAgentEnvironment",
    "EnvironmentDict",
    "MultiAgentDict",
    "AgentKeys"
]
