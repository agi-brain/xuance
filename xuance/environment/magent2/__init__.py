from . import gridworld
from . import utility
from .render import Renderer

from xuance.environment.magent2.magent_vec_env import DummyVecEnv_MAgent, SubprocVecEnv_Magent


# some alias
GridWorld = gridworld.GridWorld

__version__ = "0.3.2"


AGENT_NAME_DICT = {
    "adversarial_pursuit_v4": ['predator', 'prey'],
    "battle_v4": ['red', 'blue'],
    "battlefield_v4": ['red', 'blue'],
    "combined_arms_v6": ['redmelee', 'redranged', 'bluemelee', 'blueranged'],
    "gather_v4": ['omnivore'],
    "tiger_deer_v4": ['deer', 'tiger']
}

MAGENT2_MARL = ['adversarial_pursuit_v4',
                'battle_v4',
                'battlefield_v4',
                'combined_arms_v6',
                'gather_v4',
                'tiger_deer_v4']
