from argparse import Namespace
from typing import Optional, Union
from xuance.environment.utils import MakeEnvironment, MakeMultiAgentEnvironment, XuanCeEnvWrapper, RawEnvironment, RawMultiAgentEnv
from xuance.environment.vector_envs import DummyVecEnv, SubprocVecEnv, \
    DummyVecEnv_Atari, SubprocVecEnv_Atari, DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.environment.single_agent_env.gym import Gym_Env
from xuance.environment.single_agent_env import REGISTRY_ENV
from xuance.environment.multi_agent_env import REGISTRY_MULTI_AGENT_ENV
from xuance.environment.vector_envs import REGISTRY_VEC_ENV


def make_envs(config: Namespace,
              raw_env: Optional[Union[RawEnvironment]] = None,
              raw_multi_agent_env: Optional[Union[RawMultiAgentEnv]] = None):
    def _thunk():
        if config.env_name in REGISTRY_ENV.keys():
            return MakeEnvironment(REGISTRY_ENV[config.env_name](config))
        elif config.env_name in REGISTRY_MULTI_AGENT_ENV.keys():
            return MakeMultiAgentEnvironment(REGISTRY_MULTI_AGENT_ENV[config.env_name](config))
        else:
            raise AttributeError(f"The environment named {config.env_name} cannot be created.")

    if config.vectorize in REGISTRY_VEC_ENV.keys():
        env_fn = [_thunk for _ in range(config.parallels)]
        return REGISTRY_VEC_ENV[config.vectorize](env_fn)
    elif config.vectorize == "NOREQUIRED":
        return _thunk()
    else:
        raise AttributeError(f"The vectorized method {config.vectorize} is not implemented.")