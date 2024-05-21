from xuance.environment.single_agent_env.gym import Gym_Env
from xuance.environment.utils import EnvironmentDict
from typing import Optional

REGISTRY_ENV: Optional[EnvironmentDict] = {
    "classic_control": Gym_Env,
    "box2d": Gym_Env,
    "mujoco": Gym_Env
}

try:
    from xuance.environment.single_agent_env.gym import Atari_Env
    REGISTRY_ENV['atari'] = Atari_Env
except Exception as error:
    REGISTRY_ENV["atari"] = str(error)

try:
    from xuance.environment.single_agent_env.minigrid import MiniGridEnv
    REGISTRY_ENV['minigrid'] = MiniGridEnv
except Exception as error:
    REGISTRY_ENV["minigrid"] = str(error)

try:
    from xuance.environment.single_agent_env.drones import Drone_Env
    REGISTRY_ENV['drones'] = Drone_Env
except Exception as error:
    REGISTRY_ENV["drones"] = str(error)

try:
    from xuance.environment.single_agent_env.metadrive import MetaDrive_Env
    REGISTRY_ENV['metadrive'] = MetaDrive_Env
except Exception as error:
    REGISTRY_ENV["metadrive"] = str(error)

__all__ = [
    "REGISTRY_ENV",
    "Gym_Env",
    "Atari_Env",
    "MiniGridEnv",
    "Drone_Env",
    "MetaDrive_Env",
]
