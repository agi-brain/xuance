import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from gym.spaces import Box, Discrete
import numpy as np


class MiniGridEnv():
    """
    The wrapper of minigrid environment.

    Args:
        env_id: The environment id of minigrid.
        seed: random seed.
        render_mode: "rgb_array", "human".
        rgb_img_partial_wrapper: whether to apply the RGB image's partial observation wrapper.
        img_obs_wrapper:  whether to apply the image observation wrapper.
    """
    def __init__(self, env_id: str, seed: int, render_mode: str,
                 rgb_img_partial_obs_wrapper=False,
                 img_obs_wrapper=False):
        self.env = gym.make(env_id, render_mode=render_mode)
        if rgb_img_partial_obs_wrapper:
            self.env = RGBImgPartialObsWrapper(self.env)
        if img_obs_wrapper:
            self.env = ImgObsWrapper(self.env)

        self.env_id = env_id
        self.render_mode = render_mode
        self._episode_step = 0
        self._episode_score = 0.0
        self.image_size = np.prod(self.env.observation_space['image'].shape)  # height * width * channels
        self.dim_obs = self.image_size + 1  # direction
        self.observation_space = Box(low=0, high=255, shape=[self.dim_obs, ], dtype=np.uint8, seed=seed)
        self.action_space = self.env.action_space
        self.max_episode_steps = self.env.env.env.max_steps

    def close(self):
        """Close the environment."""
        self.env.close()

    def render(self, *args):
        """Return the rendering result"""
        return self.env.render()

    def reset(self):
        """Reset the environment."""
        obs_raw, info = self.env.reset()
        obs = self.flatten_obs(obs_raw)
        self._episode_step = 0
        self._episode_score = 0.0
        info["episode_step"] = self._episode_step
        return obs, info

    def step(self, actions):
        """Execute the actions and get next observations, rewards, and other information."""
        obs_raw, reward, terminated, truncated, info = self.env.step(actions)
        observation = self.flatten_obs(obs_raw)

        reward *= 10

        self._episode_step += 1
        self._episode_score += reward
        info["episode_step"] = self._episode_step  # current episode step
        info["episode_score"] = self._episode_score  # the accumulated rewards
        return observation, reward, terminated, truncated, info

    def flatten_obs(self, obs_raw):
        """Convert image observation to vectors"""
        image = obs_raw['image']
        direction = obs_raw['direction']
        observations = np.append(image.reshape(-1), direction)
        return observations


