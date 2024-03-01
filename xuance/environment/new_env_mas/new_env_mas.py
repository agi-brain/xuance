"""
This is an example of creating a new environment in XuanCe for multi-agent system.
This example
"""
from gym.spaces import Box, Discrete
import numpy as np


class New_Env_MAS:
    def __init__(self, env_id: str, seed: int, **kwargs):
        self.n_agents = 2
        self.state_space = Box(low=0, high=1, shape=[10, ], dtype=np.float32, seed=seed)
        self.observation_spaces = Box(low=0, high=1, shape=[self.n_agents, 10], dtype=np.float32, seed=seed)
        self.action_spaces = Box(low=0, high=1, shape=[2, ], dtype=np.float32, seed=seed)
        self.dim_state = 20

        self._episode_step = 0
        self._episode_score = 0.0

        self.max_episode_steps = 100
        self.env_info = {
            "n_agents": self.n_agents,
            "obs_shape": self.observation_spaces.shape,
            "act_space": self.action_spaces,
            "state_shape": self.dim_state,
            "n_actions": self.action_spaces.shape[-1],
            "episode_limit": self.max_episode_steps,
        }

    def close(self):
        pass

    def render(self):
        return np.zeros([2, 2, 2])

    def reset(self):
        obs, info = self.observation_spaces.sample(), {}  # reset the environment and get observations and info here.
        self._episode_step = 0
        self._episode_score = 0.0
        info["episode_step"] = self._episode_step
        return obs, info

    def step(self, actions):
        # Execute the actions and get next observations, rewards, and other information.
        observation, reward, terminated, truncated, info = self.observation_spaces.sample(), 0, False, False, {}

        self._episode_step += 1
        self._episode_score += reward
        info["episode_step"] = self._episode_step  # current episode step
        info["episode_score"] = self._episode_score  # the accumulated rewards
        return observation, reward, terminated, truncated, info

    def get_agent_mask(self):
        return np.ones(self.n_agents, dtype=np.bool_)

    def state(self):
        return np.zeros([self.dim_state])
