from argparse import Namespace

from xuance.environment.utils import MakeEnvironment, MakeMultiAgentEnvironment, XuanCeEnvWrapper, RawEnvironment, RawMultiAgentEnv
from xuance.environment.vector_envs import DummyVecEnv, SubprocVecEnv, \
    DummyVecEnv_Atari, SubprocVecEnv_Atari, DummyVecMultiAgentEnv

from xuance.environment.single_agent_env import Gym_Env
from xuance.environment.vector_envs.subprocess.subproc_vec_env import SubprocVecEnv

REGISTRY_VEC_ENV = {
    "DummyVecEnv": DummyVecEnv,
    "DummyVecMultiAgentEnv": DummyVecMultiAgentEnv,
    "Dummy_Atari": DummyVecEnv_Atari,

    # multiprocess #
    "SubprocVecEnv": SubprocVecEnv,
    "Subproc_Atari": SubprocVecEnv_Atari,
}


def make_envs(config: Namespace):
    def _thunk():
        if config.env_name == "mpe":
            from xuance.environment.multi_agent_env.mpe import MPE_Env as RawEnv
            env = RawEnv(config)
            return MakeMultiAgentEnvironment(env)

        elif config.env_name == "StarCraft2":
            from xuance.environment.multi_agent_env.starcraft2 import StarCraft2_Env as RawEnv
            env = RawEnv(config)
            return MakeMultiAgentEnvironment(env)

        elif config.env_name == "Football":
            from xuance.environment.multi_agent_env.football import GFootball_Env
            return MakeMultiAgentEnvironment(GFootball_Env(config))

        elif config.env_name == "RoboticWarehouse":
            from xuance.environment.multi_agent_env.robotic_warehouse import RoboticWarehouseEnv
            return MakeMultiAgentEnvironment(RoboticWarehouseEnv(config))

        elif config.env_name == "Atari":
            from xuance.environment.single_agent_env import Atari_Env
            return MakeEnvironment(Atari_Env(config))

        elif config.env_id.__contains__("CarRacing"):
            return MakeEnvironment(Gym_Env(config, continuous=False))

        elif config.env_name == "Platform":
            from xuance.environment.single_agent_env.platform import PlatformEnv
            env = PlatformEnv(config)
            return env

        elif config.env_name == "MiniGrid":
            from xuance.environment.single_agent_env.minigrid import MiniGridEnv as RawEnv
            env = RawEnv(config)
            return MakeEnvironment(env)

        elif config.env_name == "Drones":
            if config.num_drones > 1:
                from xuance.environment.multi_agent_env.drones import Drones_MultiAgentEnv
                return MakeMultiAgentEnvironment(Drones_MultiAgentEnv(config))
            else:
                from xuance.environment.single_agent_env.drones import Drone_Env
                return MakeEnvironment(Drone_Env(config))

        elif config.env_name == "MetaDrive":
            from xuance.environment.single_agent_env.metadrive import MetaDrive_Env
            return MakeEnvironment(MetaDrive_Env(config))

        else:
            return MakeEnvironment(Gym_Env(config))

    if config.vectorize in REGISTRY_VEC_ENV.keys():
        env_fn = [_thunk for _ in range(config.parallels)]
        return REGISTRY_VEC_ENV[config.vectorize](env_fn)
    elif config.vectorize == "NOREQUIRED":
        return _thunk()
    else:
        raise NotImplementedError

