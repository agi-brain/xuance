"""
This is an example of creating a new environment in XuanCe for multi-agent system.
This example
"""
from gym.spaces import Box, Discrete
import numpy as np


class New_Env_MAS:
    def __init__(self, env_name: str, env_id: str, seed: int, **kwargs):
        self.handles = [0]
        self.n_handles = len(self.handles)
        self.side_names = ['agent']
        self.state_space = Box(low=0, high=1, shape=[10, ], dtype=np.float, seed=seed)
        self.observation_spaces = {Box(low=0, high=1, shape=[10, ], dtype=np.float, seed=seed)}

    def close(self):
        pass

    def render(self):
        pass

    def reset(self):
        return

    def step(self, actions):
        return