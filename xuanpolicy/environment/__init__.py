from argparse import Namespace

from xuanpolicy.environment.pettingzoo_envs.pettingzoo_env import PettingZoo_Env
from xuanpolicy.environment.gym_envs.gym_env import Gym_Env, MountainCar, Atari_Env

from .pettingzoo_envs import PETTINGZOO_ENVIRONMENTS

from .vector_envs.vector_env import VecEnv
from xuanpolicy.environment.gym_envs.gym_vec_env import DummyVecEnv_Gym, DummyVecEnv_Atari
from xuanpolicy.environment.pettingzoo_envs.pettingzoo_vec_env import DummyVecEnv_Pettingzoo
from .vector_envs.subproc_vec_env import SubprocVecEnv


def make_envs(config: Namespace):
    def _thunk():
        if config.env_name in PETTINGZOO_ENVIRONMENTS:
            env = PettingZoo_Env(config.env_name, config.env_id, config.seed,
                                 continuous=config.continuous_action,
                                 render_mode=config.render_mode)
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
    elif config.vectorize == "Dummy_Gym":
        return DummyVecEnv_Gym([_thunk for _ in range(config.parallels)])
    elif config.vectorize == "Dummy_Pettingzoo":
        return DummyVecEnv_Pettingzoo([_thunk for _ in range(config.parallels)])
    elif config.vectorize == "Dummy_Atari":
        return DummyVecEnv_Atari([_thunk for _ in range(config.parallels)])
    elif config.vectorize == "NOREQUIRED":
        return _thunk()
    else:
        raise NotImplementedError

