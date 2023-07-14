import copy

from smac.env import StarCraft2Env
import numpy as np


class StarCraft2_Env:
    def __init__(self, map_name):
        self.env = StarCraft2Env(map_name=map_name)
        env_info = self.env.get_env_info()

        self.n_agents = env_info["n_agents"]
        self.n_enemies = self.env.n_enemies
        self.dim_state = env_info["state_shape"]
        self.dim_obs = env_info["obs_shape"]
        self.dim_act = self.n_actions = env_info["n_actions"]
        self.max_cycles = env_info["episode_limit"]
        self._episode_step = 0
        self._episode_score = 0
        self.filled = np.zeros([self.max_cycles, 1], np.bool)
        self.env.reset()
        self.buf_info = {
            'battle_won': 0,
            'dead_allies': 0,
            'dead_enemies': 0,
        }

    def close(self):
        self.env.close()

    def render(self, mode):
        return self.env.render(mode)

    def reset(self):
        obs, state = self.env.reset()
        self._episode_step = 0
        self._episode_score = 0.0
        info = {
            "episode_step": self._episode_step,
            "episode_score": self._episode_score,
        }
        return obs, state, info

    def step(self, actions):
        reward, terminated, info = self.env.step(actions)
        if info == {}:
            info = self.buf_info
        obs = self.env.get_obs()
        state = self.env.get_state()
        self._episode_step += 1
        self._episode_score += reward
        self.buf_info = copy.deepcopy(info)
        info["episode_step"] = self._episode_step
        info["episode_score"] = self._episode_score
        truncated = True if self._episode_step >= self.max_cycles else False
        return obs, state, reward, terminated, truncated, info

    def get_avail_actions(self):
        return self.env.get_avail_actions()
