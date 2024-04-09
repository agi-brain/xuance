"""
This is an example of creating a new environment in XuanCe for multi-agent system.
This example
"""
from gymnasium.spaces import Box, Discrete
import numpy as np


class New_Env_MAS:
    def __init__(self, args, **kwargs):
        self.n_agents = 3  # the number of agents
        self.dim_obs = 10  # dimension of one agent's observation
        self.dim_state = 12  # dimension of global state
        self.dim_action = 2  # dimension of actions (continuous)
        self.n_actions = 5  # number of discrete actions (discrete)
        self.seed = args.seed  # random seed
        self.state_space = Box(low=0, high=1, shape=[self.dim_state, ], dtype=np.float32, seed=self.seed)
        self.observation_space = Box(low=0, high=1, shape=[self.dim_obs, ], dtype=np.float32, seed=self.seed)
        if kwargs['continuous']:
            self.action_space = Box(low=0, high=1, shape=[self.dim_action, ], dtype=np.float32, seed=self.seed)
        else:
            self.action_space = Discrete(n=self.n_actions, seed=self.seed)

        self._episode_step = 0  # initialize the current step
        self._episode_score = 0.0  # initialize the episode score

        # Set the max steps for each episode.
        try:
            self.max_episode_steps = args.max_episode_steps
        except:
            self.max_episode_steps = 100
        self.env_info = {
            "n_agents": self.n_agents,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "state_space": self.state_space,
            "episode_limit": self.max_episode_steps,
        }

    def close(self):
        """Close your environment here"""
        pass

    def render(self, **kwargs):
        """Render the environment, and return the images"""
        images = np.zeros([2, 2, 2])
        return images

    def reset(self):
        """Reset your environment, and return initialized observations and other information."""
        obs = np.array([self.observation_space.sample() for _ in range(self.n_agents)])
        info = {}
        self._episode_step = 0
        self._episode_score = 0.0
        info["episode_step"] = self._episode_step
        return obs, info

    def step(self, actions):
        """Execute the actions and get next observations, rewards, and other information."""
        observation = np.array([self.observation_space.sample() for _ in range(self.n_agents)])
        reward, info = np.zeros([self.n_agents, 1]), {}
        terminated = [False for _ in range(self.n_agents)]
        truncated = [True for _ in range(self.n_agents)] if (self._episode_step >= self.max_episode_steps) else [False for _ in range(self.n_agents)]

        self._episode_step += 1
        self._episode_score += reward
        info["episode_step"] = self._episode_step  # current episode step
        info["episode_score"] = self._episode_score  # the accumulated rewards
        return observation, reward, terminated, truncated, info

    def get_agent_mask(self):
        """Get mask variables of agents, 1 means the agent is activated."""
        return np.ones(self.n_agents, dtype=np.bool_)

    def state(self):
        """Get the global state of the environment in current step."""
        return np.zeros([self.dim_state])
