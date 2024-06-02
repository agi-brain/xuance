import gym
import rware
from gym.spaces import Box
import numpy as np
from xuance.environment import RawMultiAgentEnv


class RoboticWarehouseEnv(RawMultiAgentEnv):
    """
    Note: To make this environment successfully, the gym verison is suggested to be 0.21.0.
    """
    def __init__(self, config):
        super(RoboticWarehouseEnv, self).__init__()
        self.env = gym.make(config.env_id)
        self.num_agents = self.env.env.n_agents  # the number of agents
        self.agents = [f'agent_{i}' for i in range(self.num_agents)]
        self.seed = config.seed  # random seed
        self.env.seed(self.seed)

        self.observation_space = {k: self.env.observation_space[i] for i, k in enumerate(self.agents)}
        self.action_space = {k: self.env.action_space[i] for i, k in enumerate(self.agents)}
        self.dim_state = sum([self.observation_space[k].shape[-1] for k in self.agents])
        self.state_space = Box(-np.inf, np.inf, shape=[self.dim_state, ], dtype=np.float32)

        self.max_episode_steps = self.env.env.max_steps
        self._episode_step = 0  # initialize the current step

    def close(self):
        """Close your environment here"""
        self.env.close()

    def render(self, render_mode):
        """Render the environment, and return the images"""
        return self.env.render(render_mode)

    def reset(self):
        """Reset your environment, and return initialized observations and other information."""
        obs = self.env.reset()
        obs = np.array(obs)
        obs_dict = {k: obs[i] for i, k in enumerate(self.agents)}
        info = {}
        self._episode_step = 0
        return obs_dict, info

    def step(self, actions):
        """Execute the actions and get next observations, rewards, and other information."""
        actions_list = [actions[k] for k in self.agents]
        observation, reward, terminated, info = self.env.step(actions_list)
        obs_dict = {k: observation[i] for i, k in enumerate(self.agents)}
        reward_dict = {k: reward[i] for i, k in enumerate(self.agents)}
        terminated_dict = {k: terminated[i] for i, k in enumerate(self.agents)}
        self._episode_step += 1  # initialize the current step

        truncated = True if (self._episode_step >= self.max_episode_steps) else False

        return obs_dict, reward_dict, terminated_dict, truncated, info

    def state(self):
        """Get the global state of the environment in current step."""
        return self.state_space.sample()
