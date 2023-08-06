from argparse import Namespace

from xuanpolicy.environment.gym.gym_env import Gym_Env, MountainCar
from .pettingzoo import PETTINGZOO_ENVIRONMENTS

from .vector_envs.vector_env import VecEnv
from xuanpolicy.environment.gym.gym_vec_env import DummyVecEnv_Gym, DummyVecEnv_Atari, SubprocVecEnv_Gym, SubprocVecEnv_Atari
from xuanpolicy.environment.pettingzoo.pettingzoo_vec_env import DummyVecEnv_Pettingzoo
from xuanpolicy.environment.magent2.magent_vec_env import DummyVecEnv_MAgent
from xuanpolicy.environment.starcraft2.sc2_vec_env import SubprocVecEnv_StarCraft2
from xuanpolicy.environment.football.gfootball_vec_env import DummyVecEnv_GFootball

from .vector_envs.subproc_vec_env import SubprocVecEnv


def make_envs(config: Namespace):
    def _thunk():
        if config.env_name in PETTINGZOO_ENVIRONMENTS:
            from xuanpolicy.environment.pettingzoo.pettingzoo_env import PettingZoo_Env
            env = PettingZoo_Env(config.env_name, config.env_id, config.seed,
                                 continuous=config.continuous_action,
                                 render_mode=config.render_mode)
        elif config.env_name == "StarCraft2":
            from xuanpolicy.environment.starcraft2.sc2_env import StarCraft2_Env
            env = StarCraft2_Env(map_name=config.env_id)
        elif config.env_name == "Football":
            from xuanpolicy.environment.football.gfootball_env import GFootball_Env
            env = GFootball_Env(config)
        elif config.env_name == "MAgent2":
            from xuanpolicy.environment.magent2.magent_env import MAgent_Env
            env = MAgent_Env(config.env_id, config.seed,
                             minimap_mode=config.minimap_mode,
                             max_cycles=config.max_cycles,
                             extra_features=config.extra_features,
                             map_size=config.map_size,
                             render_mode=config.render_mode)
        elif config.env_name == "Atari":
            from xuanpolicy.environment.gym.gym_env import Atari_Env
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
    elif config.vectorize == "Dummy_MAgent":
        return DummyVecEnv_MAgent([_thunk for _ in range(config.parallels)])
    elif config.vectorize == "Dummy_StarCraft2":
        return SubprocVecEnv_StarCraft2([_thunk for _ in range(config.parallels)])
    elif config.vectorize == "Dummy_Football":
        return DummyVecEnv_GFootball([_thunk for _ in range(config.parallels)])
    elif config.vectorize == "Dummy_Atari":
        return DummyVecEnv_Atari([_thunk for _ in range(config.parallels)])
    elif config.vectorize == "NOREQUIRED":
        return _thunk()
    else:
        raise NotImplementedError

