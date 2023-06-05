import importlib

from .custom_envs.pettingzoo_env import PettingZoo_Env
from .custom_envs.gym_env import Gym_Env

from .custom_envs.pettingzoo_env import PETTINGZOO_ENVIRONMENTS

from .vector_envs.vector_env import VecEnv
from .vector_envs.dummy_vec_env import DummyVecEnv, DummyVecEnv_MAS
from .vector_envs.subproc_vec_env import SubprocVecEnv


def make_envs(env_name: str,
              env_id: str,
              seed: int,
              vectorize: str,
              parallels: int,
              continuous_action: bool = False,
              render_mode: str = 'rgb_array',
              ):
    def _thunk():
        if env_name in PETTINGZOO_ENVIRONMENTS:
            scienario = importlib.import_module('pettingzoo.' + env_name + '.' + env_id)
            env = PettingZoo_Env(scienario.parallel_env(continuous_actions=continuous_action, render_mode=render_mode),
                                 env_name+'.'+env_id)
        else:
            env = Gym_Env(env_id, seed, render_mode)
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

