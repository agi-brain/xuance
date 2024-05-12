import gym
from gym.spaces import Box
import numpy as np
from xuance.environment.robotic_warehouse import ENV_IDs


class RoboticWarehouseEnv:
    def __init__(self, args, **kwargs):
        self.env = gym.make(ENV_IDs[args.env_id])
        self.n_agents = self.env.n_agents  # the number of agents
        self.seed = args.seed  # random seed
        self.env.seed(self.seed)
        self.observation_space = self.env.observation_space[0]
        self.action_space = self.env.action_space[0]

        self.dim_obs = self.observation_space.shape[-1]
        self.n_actions = self.action_space.n
        self.dim_state = self.dim_obs * self.n_agents
        self.state_space = Box(low=0, high=1, shape=[self.dim_state, ], dtype=np.float32)

        self._episode_step = 0  # initialize the current step
        self._episode_score = np.zeros([self.n_agents, 1])  # initialize the episode score

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
        self.env.close()

    def render(self, render_mode):
        """Render the environment, and return the images"""
        images = self.env.render(render_mode)
        return images

    def reset(self):
        """Reset your environment, and return initialized observations and other information."""
        obs = self.env.reset()
        obs = np.array(obs)
        info = {}
        self._episode_step = 0
        self._episode_score = np.zeros([self.n_agents, 1])
        info["episode_step"] = self._episode_step
        return obs, info

    def step(self, actions):
        """Execute the actions and get next observations, rewards, and other information."""
        observation, reward, terminated, info = self.env.step(actions)
        observation, reward = np.array(observation), np.reshape(reward, [self.n_agents, 1])
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
