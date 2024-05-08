from argparse import Namespace

from xuance.environment.gym import Gym_Env, MountainCar
from .pettingzoo import PETTINGZOO_ENVIRONMENTS

from .vector_envs.vector_env import VecEnv
from xuance.environment.gym import DummyVecEnv_Gym, SubprocVecEnv_Gym
from xuance.environment.gym import DummyVecEnv_Atari, SubprocVecEnv_Atari
from xuance.environment.pettingzoo import DummyVecEnv_Pettingzoo, SubprocVecEnv_Pettingzoo
from xuance.environment.starcraft2 import DummyVecEnv_StarCraft2, SubprocVecEnv_StarCraft2
from xuance.environment.football import DummyVecEnv_GFootball, SubprocVecEnv_GFootball
from xuance.environment.minigrid import DummyVecEnv_MiniGrid, SubprocVecEnv_MiniGrid
from xuance.environment.drones import DummyVecEnv_Drones, SubprocVecEnv_Drones
from xuance.environment.drones import DummyVecEnv_Drones_MAS, SubprocVecEnv_Drones_MAS
from xuance.environment.metadrive import SubprocVecEnv_MetaDrive
from xuance.environment.robotic_warehouse import DummyVecEnv_RoboticWarehouse, SubprocVecEnv_RoboticWarehouse
from xuance.environment.new_env import DummyVecEnv_New, SubprocVecEnv_New
from xuance.environment.new_env_mas import DummyVecEnv_New_MAS, SubprocVecEnv_New_MAS

from .vector_envs.subproc_vec_env import SubprocVecEnv

REGISTRY_VEC_ENV = {
    "Dummy_Gym": DummyVecEnv_Gym,
    "Dummy_Pettingzoo": DummyVecEnv_Pettingzoo,
    "Dummy_StarCraft2": DummyVecEnv_StarCraft2,
    "Dummy_Football": DummyVecEnv_GFootball,
    "Dummy_Atari": DummyVecEnv_Atari,
    "Dummy_MiniGrid": DummyVecEnv_MiniGrid,
    "Dummy_Drone": DummyVecEnv_Drones,
    "Dummy_Drone_MAS": DummyVecEnv_Drones_MAS,
    "Dummy_RoboticWarehouse": DummyVecEnv_RoboticWarehouse,
    "Dummy_NewEnv": DummyVecEnv_New,  # Add the newly defined vectorized environment
    "Dummy_NewEnv_MAS": DummyVecEnv_New_MAS,  # Add the newly defined vectorized environment for multi-agent systems

    # multiprocess #
    "Subproc_Gym": SubprocVecEnv_Gym,
    "Subproc_Pettingzoo": SubprocVecEnv_Pettingzoo,
    "Subproc_StarCraft2": SubprocVecEnv_StarCraft2,
    "Subproc_Football": SubprocVecEnv_GFootball,
    "Subproc_Atari": SubprocVecEnv_Atari,
    "Subproc_MiniGrid": SubprocVecEnv_MiniGrid,
    "Subproc_Drone": SubprocVecEnv_Drones,
    "Subproc_Drone_MAS": SubprocVecEnv_Drones_MAS,
    "Subproc_MetaDrive": SubprocVecEnv_MetaDrive,
    "Subproc_RoboticWarehouse": SubprocVecEnv_RoboticWarehouse,
    "Subproc_NewEnv": SubprocVecEnv_New,  # Add the newly defined vectorized environment
    "Subproc_NewEnv_MAS": SubprocVecEnv_New_MAS,  # Add the newly defined vectorized environment for multi-agent systems
}


def make_envs(config: Namespace):
    def _thunk():
        if config.env_name in PETTINGZOO_ENVIRONMENTS:
            from xuance.environment.pettingzoo.pettingzoo_env import PettingZoo_Env
            env = PettingZoo_Env(config.env_name, config.env_id, config.seed,
                                 continuous=config.continuous_action,
                                 render_mode=config.render_mode)

        elif config.env_name == "StarCraft2":
            from xuance.environment.starcraft2.sc2_env import StarCraft2_Env
            env = StarCraft2_Env(map_name=config.env_id)

        elif config.env_name == "Football":
            from xuance.environment.football.gfootball_env import GFootball_Env
            env = GFootball_Env(config)

        elif config.env_name == "MAgent2":
            from xuance.environment.magent2.magent_env import MAgent_Env
            env = MAgent_Env(config.env_id, config.seed,
                             minimap_mode=config.minimap_mode,
                             max_cycles=config.max_cycles,
                             extra_features=config.extra_features,
                             map_size=config.map_size,
                             render_mode=config.render_mode)

        elif config.env_name == "RoboticWarehouse":
            from xuance.environment.robotic_warehouse.robotic_warehouse_env import RoboticWarehouseEnv
            env = RoboticWarehouseEnv(config, render_mode=config.render_mode)

        elif config.env_name == "Atari":
            from xuance.environment.gym.gym_env import Atari_Env
            env = Atari_Env(config.env_id, config.seed, config.render_mode,
                            config.obs_type, config.frame_skip, config.num_stack, config.img_size, config.noop_max)

        elif config.env_id.__contains__("MountainCar"):
            env = MountainCar(config.env_id, config.seed, config.render_mode)

        elif config.env_id.__contains__("CarRacing"):
            env = Gym_Env(config.env_id, config.seed, config.render_mode, continuous=False)

        elif config.env_id.__contains__("Platform"):
            from xuance.environment.gym_platform.envs.platform_env import PlatformEnv
            env = PlatformEnv()

        elif config.env_name == "MiniGrid":
            from xuance.environment.minigrid.minigrid_env import MiniGridEnv
            env = MiniGridEnv(config.env_id, config.seed, config.render_mode,
                              rgb_img_partial_obs_wrapper=config.RGBImgPartialObsWrapper,
                              img_obs_wrapper=config.ImgObsWrapper)

        elif config.env_name == "Drones":
            from xuance.environment.drones.drones_env import Drones_Env
            env = Drones_Env(config)

        elif config.env_name == "MetaDrive":
            from xuance.environment.metadrive.metadrive_env import MetaDrive_Env
            env = MetaDrive_Env(config)

        elif config.env_name == "NewEnv":  # Add the newly defined vectorized environment
            from xuance.environment.new_env.new_env import New_Env
            env = New_Env(config.env_id, config.seed, continuous=False)

        elif config.env_name == "NewEnv_MAS":  # Add the newly defined vectorized environment
            from xuance.environment.new_env_mas.new_env_mas import New_Env_MAS
            env = New_Env_MAS(config, continuous=config.continuous_action)

        else:
            env = Gym_Env(config.env_id, config.seed, config.render_mode)

        return env

    if config.vectorize in ["Dummy_MAgent", "Subproc_MAgent"]:  # for the support of magent2 environment
        from xuance.environment.magent2.magent_vec_env import DummyVecEnv_MAgent, SubprocVecEnv_Magent
        REGISTRY_VEC_ENV.update({
            "Dummy_MAgent": DummyVecEnv_MAgent,
            "Subproc_MAgent": SubprocVecEnv_Magent
        })
    if config.vectorize in REGISTRY_VEC_ENV.keys():
        return REGISTRY_VEC_ENV[config.vectorize]([_thunk for _ in range(config.parallels)])
    elif config.vectorize == "NOREQUIRED":
        return _thunk()
    else:
        raise NotImplementedError

