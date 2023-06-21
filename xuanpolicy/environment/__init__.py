import importlib
from argparse import Namespace

from .custom_envs.pettingzoo_env import PettingZoo_Env
from .custom_envs.gym_env import Gym_Env, MountainCar, Atari_Env

from .custom_envs.pettingzoo_env import PETTINGZOO_ENVIRONMENTS

from .vector_envs.vector_env import VecEnv
from .vector_envs.dummy_vec_env import DummyVecEnv, DummyVecEnv_MAS, DummyVecEnv_Atari
from .vector_envs.subproc_vec_env import SubprocVecEnv


def make_envs(config: Namespace):
    def _thunk():
        if config.env_name in PETTINGZOO_ENVIRONMENTS:
            scienario = importlib.import_module('pettingzoo.' + config.env_name + '.' + config.env_id)
            env = PettingZoo_Env(scienario.parallel_env(continuous_actions=config.continuous_action,
                                                        render_mode=config.render_mode),
                                 config.env_name+'.'+config.env_id)
        elif config.env_name == "Atari":
            env = Atari_Env(config.env_id, config.seed, config.render_mode,
                            config.obs_type, config.frame_skip, config.num_stack, config.img_size, config.noop_max)
        elif config.env_id.__contains__("MountainCar"):
            env = MountainCar(config.env_id, config.seed, config.render_mode)
        elif config.env_id.__contains__("CarRacing"):
            env = Gym_Env(config.env_id, config.seed, config.render_mode, continuous=False)
        else:
            env = Gym_Env(config.env_id, config.seed, config.render_mode)
        return env

    if config.vectorize == "Subproc":
        return SubprocVecEnv([_thunk for _ in range(config.parallels)])
    elif config.vectorize == "Dummy":
        return DummyVecEnv([_thunk for _ in range(config.parallels)])
    elif config.vectorize == "Dummy_MAS":
        return DummyVecEnv_MAS([_thunk for _ in range(config.parallels)])
    elif config.vectorize == "Dummy_Atari":
        return DummyVecEnv_Atari([_thunk for _ in range(config.parallels)])
    elif config.vectorize == "NOREQUIRED":
        return _thunk()
    else:
        raise NotImplementedError

