from xuance.environment.single_agent_env.gym import Gym_Env
from xuance.environment.utils import EnvironmentDict
from typing import Optional

REGISTRY_ENV: Optional[EnvironmentDict] = {
    "Classic Control": Gym_Env,
    "Box2D": Gym_Env,
    "MuJoCo": Gym_Env
}

try:
    from xuance.environment.single_agent_env.gym import Atari_Env
    REGISTRY_ENV['Atari'] = Atari_Env
except Exception as error:
    REGISTRY_ENV["Atari"] = str(error)

try:
    from xuance.environment.single_agent_env.minigrid import MiniGridEnv
    REGISTRY_ENV['MiniGrid'] = MiniGridEnv
except Exception as error:
    REGISTRY_ENV["MiniGrid"] = str(error)

try:
    from xuance.environment.single_agent_env.drones import Drone_Env
    REGISTRY_ENV['Drone'] = Drone_Env
except Exception as error:
    REGISTRY_ENV["Drone"] = str(error)

try:
    from xuance.environment.single_agent_env.metadrive import MetaDrive_Env
    REGISTRY_ENV['MetaDrive'] = MetaDrive_Env
except Exception as error:
    REGISTRY_ENV["MetaDrive"] = str(error)

try:
    from xuance.environment.single_agent_env.platform import PlatformEnv
    REGISTRY_ENV['Platform'] = PlatformEnv
except Exception as error:
    REGISTRY_ENV["Platform"] = str(error)

__all__ = [
    "REGISTRY_ENV",
]
