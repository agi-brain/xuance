import importlib

from .custom_envs.toy_env import Toy_Env
from .custom_envs.mujoco_env import MuJoCo_Env
from .custom_envs.atari_env import Atari_Env
from xuanpolicy.environment.custom_envs.pettingzoo_env import PettingZooWrapper

from .custom_envs.atari_env import ENVIRONMENT_IDS as ATARI_ENVIRONMENTS
from .custom_envs.mujoco_env import ENVIRONMENT_IDS as MUJOCO_ENVIRONMENTS
from .custom_envs.robotics_env import ENVIRONMENT_IDS as ROBOTICS_ENVIRONMENTS
from .custom_envs.toy_env import ENVIRONMENT_IDS as TOY_ENVIRONMENTS
from xuanpolicy.environment.custom_envs.pettingzoo_env import PETTINGZOO_ENVS

from .vector_envs.vector_env import VecEnv
from .vector_envs.dummy_vec_env import DummyVecEnv, DummyVecEnv_MAS
from .vector_envs.subproc_vec_env import SubprocVecEnv


def make_envs(env_name: str,
              env_id: str,
              seed: int,
              vectorize: str,
              parallels: int,
              continuous_action: bool = False,
              render_mode: str = 'human',
              ):
    def _thunk():
        if env_id in TOY_ENVIRONMENTS:
            env = Toy_Env(env_id, seed)
        elif env_id in MUJOCO_ENVIRONMENTS:
            env = MuJoCo_Env(env_id, seed)
        elif env_id in ATARI_ENVIRONMENTS:
            env = Atari_Env(env_id, seed)
        elif env_name in PETTINGZOO_ENVS:
            scienario = importlib.import_module('pettingzoo.' + env_name + '.' + env_id)
            env = PettingZooWrapper(scienario.parallel_env(continuous_actions=continuous_action,
                                                           render_mode=render_mode),
                                    env_name+'.'+env_id)
        else:
            raise NotImplementedError
        return env

    if vectorize == "Subproc":
        return SubprocVecEnv([_thunk for _ in range(parallels)])
    elif vectorize == "Dummy":
        return DummyVecEnv([_thunk for _ in range(parallels)])
    elif vectorize == "Dummy_MAS":
        return DummyVecEnv_MAS([_thunk for _ in range(parallels)])
    elif vectorize == "NOREQUIRED":
        return _thunk()
    else:
        raise NotImplementedError

