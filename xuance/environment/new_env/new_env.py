"""
This is an example of creating a new environment in XuanCe.
"""
from gym.spaces import Box, Discrete
import numpy as np


class New_Env:
    def __init__(self, env_id: str, seed: int, *args, **kwargs):
        continuous = kwargs['continuous']
        self.env_id = env_id
        self._episode_step = 0
        self._episode_score = 0.0
        self.observation_space = Box(low=0, high=1, shape=[8, ], dtype=np.float, seed=seed)
        if continuous:
            self.action_space = Box(low=0, high=1, shape=[2, ], dtype=np.float, seed=seed)
        else:
            self.action_space = Discrete(n=2, seed=seed)
        self.max_episode_steps = 100

    def close(self):
        pass

    def render(self):
        pass

    def reset(self):
        obs, info = self.observation_space.sample(), {}  # reset the environment and get observations and info here.
        self._episode_step = 0
        self._episode_score = 0.0
        info["episode_step"] = self._episode_step
        return obs, info

    def step(self, actions):
        # Execute the actions and get next observations, rewards, and other information.
        observation, reward, terminated, truncated, info = self.observation_space.sample(), 0, False, False, {}

        self._episode_step += 1
        self._episode_score += reward
        info["episode_step"] = self._episode_step  # current episode step
        info["episode_score"] = self._episode_score  # the accumulated rewards
        return observation, reward, terminated, truncated, info
