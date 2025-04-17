import gym
import numpy as np
from gym.spaces import Box
from xuance.environment import  RawEnvironment
from utils import wrap_env

class Hopper(RawEnvironment):
    def __init__(self, env_config):
        super(Hopper, self).__init__()
        self.env_id = env_config.env_id
        self.observation_space = Box(-np.inf, np.inf, shape=[11, ], seed=env_config.env_seed)
        self.action_space = Box(-1.0, 1.0, shape=[3, ], seed=env_config.env_seed)
        self.max_episode_steps = env_config.test_steps // env_config.test_episode
        self._current_step = 0
        env = gym.make(self.env_id, seed=env_config.env_seed)
        self.env = wrap_env(env, state_mean=env_config.state_mean,state_std=env_config.state_std)
        self.env.seed(env_config.env_seed)
        self.env.action_space.seed(env_config.env_seed)

    def reset(self, **kwargs):
        self._current_step = 0
        state = self.env.reset(**kwargs)
        return state, {}

    def step(self, action):
        self._current_step += 1
        truncated = False if self._current_step < self.max_episode_steps else True
        observation, rewards, terminated, info = self.env.step(action)
        return observation, rewards, terminated, truncated, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        self.env.close()
