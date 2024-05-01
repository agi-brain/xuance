"""
This is an example of creating a new environment in XuanCe.
"""
from gym.spaces import Box, Discrete
import numpy as np


class New_Env:
    def __init__(self, env_id: str, seed: int, *args, **kwargs):
        continuous = kwargs['continuous']
        self.env_id = env_id  # The name of the map or scenario that to be specified.
        self._episode_step = 0  # The count of steps for current episode.
        self._episode_score = 0.0  # The cumulated rewards for current episode.
        self.observation_space = Box(low=0, high=1, shape=[8, ], dtype=np.float, seed=seed)
        if continuous:
            """For environment with continuous action space."""
            self.action_space = Box(low=0, high=1, shape=[2, ], dtype=np.float, seed=seed)
        else:
            """For environment with discrete action space."""
            self.action_space = Discrete(n=2, seed=seed)
        self.max_episode_steps = 100  # The max steps for each episode.

    def close(self):
        """Close your environment here"""
        pass

    def render(self, *args, **kwargs):
        """Render the environment, and return the images"""
        return np.zeros([2, 2, 2])

    def reset(self):
        """Reset your environment, and return initialized observations and other information."""
        obs, info = self.observation_space.sample(), {}  # reset the environment and get observations and info here.
        self._episode_step = 0
        self._episode_score = 0.0
        info["episode_step"] = self._episode_step
        return obs, info

    def step(self, actions):
        """Execute the actions and get next observations, rewards, and other information."""
        observation, reward, terminated, info = self.observation_space.sample(), 0, False, {}
        truncated = True if (self._episode_step >= self.max_episode_steps) else False

        self._episode_step += 1
        self._episode_score += reward
        info["episode_step"] = self._episode_step  # current episode step
        info["episode_score"] = self._episode_score  # the accumulated rewards
        return observation, reward, terminated, truncated, info
