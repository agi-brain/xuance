from argparse import Namespace

from xuance.environment.utils import MakeEnvironment, MakeMultiAgentEnvironment, XuanCeEnvWrapper, RawEnvironment, RawMultiAgentEnv
from xuance.environment.vector_envs import DummyVecEnv, SubprocVecEnv, \
    DummyVecEnv_Atari, SubprocVecEnv_Atari, DummyVecMutliAgentEnv

from xuance.environment.single_agent_env import Gym_Env
from .pettingzoo import PETTINGZOO_ENVIRONMENTS

from xuance.environment.pettingzoo import DummyVecEnv_Pettingzoo, SubprocVecEnv_Pettingzoo
from xuance.environment.starcraft2 import DummyVecEnv_StarCraft2, SubprocVecEnv_StarCraft2
from xuance.environment.football import DummyVecEnv_GFootball, SubprocVecEnv_GFootball
from xuance.environment.drones import DummyVecEnv_Drones_MAS, SubprocVecEnv_Drones_MAS
from xuance.environment.robotic_warehouse import DummyVecEnv_RoboticWarehouse, SubprocVecEnv_RoboticWarehouse
from xuance.environment.new_env_mas import DummyVecEnv_New_MAS, SubprocVecEnv_New_MAS

from xuance.environment.vector_envs.subprocess.subproc_vec_env import SubprocVecEnv

REGISTRY_VEC_ENV = {
    "DummyVecEnv": DummyVecEnv,
    # "DummyVecEnv_MAS": DummyVecEnv_MAS,
    "Dummy_Pettingzoo": DummyVecMutliAgentEnv,
    "Dummy_StarCraft2": DummyVecEnv_StarCraft2,
    "Dummy_Football": DummyVecEnv_GFootball,
    "Dummy_Atari": DummyVecEnv_Atari,
    "Dummy_Drone_MAS": DummyVecEnv_Drones_MAS,
    "Dummy_RoboticWarehouse": DummyVecEnv_RoboticWarehouse,
    "Dummy_NewEnv_MAS": DummyVecEnv_New_MAS,  # Add the newly defined vectorized environment for multi-agent systems

    # multiprocess #
    "SubprocVecEnv": SubprocVecEnv,
    "Subproc_Pettingzoo": SubprocVecEnv_Pettingzoo,
    "Subproc_StarCraft2": SubprocVecEnv_StarCraft2,
    "Subproc_Football": SubprocVecEnv_GFootball,
    "Subproc_Atari": SubprocVecEnv_Atari,
    "Subproc_Drone_MAS": SubprocVecEnv_Drones_MAS,
    "Subproc_RoboticWarehouse": SubprocVecEnv_RoboticWarehouse,
    "Subproc_NewEnv_MAS": SubprocVecEnv_New_MAS,  # Add the newly defined vectorized environment for multi-agent systems
}


def make_envs(config: Namespace):
    def _thunk():
        if config.env_name in PETTINGZOO_ENVIRONMENTS:
            from xuance.environment.pettingzoo.pettingzoo_env import PettingZoo_Env
            env = PettingZoo_Env(config.env_name, config.env_id, config.seed,
                                 continuous=config.continuous_action,
                                 render_mode=config.render_mode)
            return MakeMultiAgentEnvironment(env)

        if config.env_name == "mpe":
            from xuance.environment.multi_agent_env.mpe import MPE_Env as RawEnv
            env = RawEnv(config)
            return MakeMultiAgentEnvironment(env)

        elif config.env_name == "StarCraft2":
            from xuance.environment.multi_agent_env.starcraft2 import StarCraft2_Env as RawEnv
            env = RawEnv(config)
            return MakeMultiAgentEnvironment(env)

        elif config.env_name == "Football":
            from xuance.environment.football.gfootball_env import GFootball_Env
            env = GFootball_Env(config)
            return MakeMultiAgentEnvironment(env)

        elif config.env_name == "MAgent2":
            from xuance.environment.magent2.magent_env import MAgent_Env
            env = MAgent_Env(config.env_id, config.seed,
                             minimap_mode=config.minimap_mode,
                             max_cycles=config.max_cycles,
                             extra_features=config.extra_features,
                             map_size=config.map_size,
                             render_mode=config.render_mode)
            return MakeMultiAgentEnvironment(env)

        elif config.env_name == "RoboticWarehouse":
            from xuance.environment.robotic_warehouse.robotic_warehouse_env import RoboticWarehouseEnv
            env = RoboticWarehouseEnv(config, render_mode=config.render_mode)
            return MakeMultiAgentEnvironment(env)

        elif config.env_name == "Atari":
            from xuance.environment.single_agent_env import Atari_Env
            env = Atari_Env(config.env_id, config.seed, config.render_mode,
                            config.obs_type, config.frame_skip, config.num_stack, config.img_size, config.noop_max)
            return MakeEnvironment(env)

        elif config.env_id.__contains__("CarRacing"):
            env = Gym_Env(config.env_id, config.seed, config.render_mode, continuous=False)
            return MakeEnvironment(env)

        elif config.env_id.__contains__("Platform"):
            from xuance.environment.single_agent_env.platform import PlatformEnv
            env = PlatformEnv(config)
            return env

        elif config.env_name == "MiniGrid":
            from xuance.environment.single_agent_env.minigrid import MiniGridEnv as RawEnv
            env = RawEnv(config)
            return MakeEnvironment(env)

        elif config.env_name == "Drones":
            if config.agent_name == "iddpg":
                from xuance.environment.multi_agent_env.drones import Drones_MultiAgentEnv as RawEnv
            else:
                from xuance.environment.single_agent_env.drones import Drone_Env as RawEnv
            env = RawEnv(config)
            return MakeEnvironment(env)

        elif config.env_name == "MetaDrive":
            from xuance.environment.single_agent_env.metadrive import MetaDrive_Env as RawEnv
            env = RawEnv(config)
            return MakeEnvironment(env)

        else:
            env = Gym_Env(config.env_id, config.seed, config.render_mode)
            return MakeEnvironment(env)

    if config.vectorize in ["Dummy_MAgent", "Subproc_MAgent"]:  # for the support of magent2 environment
        from xuance.environment.magent2.magent_vec_env import DummyVecEnv_MAgent, SubprocVecEnv_Magent
        REGISTRY_VEC_ENV.update({
            "Dummy_MAgent": DummyVecEnv_MAgent,
            "Subproc_MAgent": SubprocVecEnv_Magent
        })
    if config.vectorize in REGISTRY_VEC_ENV.keys():
        env_fn = [_thunk for _ in range(config.parallels)]
        return REGISTRY_VEC_ENV[config.vectorize](env_fn)
    elif config.vectorize == "NOREQUIRED":
        return _thunk()
    else:
        raise NotImplementedError

