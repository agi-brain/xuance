import os
import argparse

import numpy as np
from gym.spaces import Box, Discrete
from xuance.environment import RawEnvironment
from xuance.common import get_configs
from xuance.torch.agents import DQN_Agent
from xuance.environment import REGISTRY_ENV, make_envs


class MyNewEnv(RawEnvironment):
    def __init__(self, env_config):
        super(MyNewEnv, self).__init__()
        self.env_id = env_config.env_id
        self.observation_space = Box(-np.inf, np.inf, shape=[18, ])
        self.action_space = Discrete(n=5)
        self.max_episode_steps = 32
        self._current_step = 0

    def reset(self, **kwargs):
        self._current_step = 0
        return self.observation_space.sample(), {}

    def step(self, action):
        observation = self.observation_space.sample()
        rewards = np.random.random()
        terminated = False
        truncated = False if self._current_step < self.max_episode_steps else True
        info = {}
        return observation, rewards, terminated, truncated, info

    def render(self, *args, **kwargs):
        return np.ones([64, 64, 64])

    def close(self):
        return


REGISTRY_ENV['MyNewEnv'] = MyNewEnv  # Add your environment to the REGISTRY_ENV dictionary.
config_dict = get_configs(os.path.join(os.getcwd(), "new_env.yaml"))  # Get config files and return a dictionary.
config = argparse.Namespace(**config_dict)  # Convert dictionary to a Namespace object with attributes.

envs = make_envs(config)  # Create environments.
Agent = DQN_Agent(config, envs)  # Create a DRL agent.
Agent.train(config.running_steps)  # Train the DRL model.
Agent.finish()  # Finish training.
Agent.test(make_envs, test_episodes=5)  # Test the trained model.
