import gym
import numpy as np
from collections import deque
from typing import Sequence
import cv2


class Gym_Env(gym.Wrapper):
    """
    Args:
        env_id: The environment id of Atari, such as "Breakout-v5", "Pong-v5", etc.
        seed: random seed.
        render_mode: "rgb_array", "human"
    """
    def __init__(self, env_id: str, seed: int, render_mode: str):
        self.env = gym.make(env_id, render_mode=render_mode)
        self.env.action_space.seed(seed=seed)
        self.env.reset(seed=seed)
        super(Gym_Env, self).__init__(self.env)
        # self.env.seed(seed)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata
        self.reward_range = self.env.reward_range

    def close(self):
        self.env.close()

    def render(self, mode):
        return self.env.render()


class MountainCar(Gym_Env):
    def __init__(self, env_id: str, seed: int, render_mode: str):
        super(MountainCar, self).__init__(env_id, seed, render_mode)

    def step(self, actions):
        observation, reward, terminated, truncated, info = self.env.step(actions)
        if observation[0] >= self.env.unwrapped.goal_position:
            terminated = True
            reward += 10
        return observation, reward, terminated, truncated, info


class Atari_Env(gym.Wrapper):
    """
    We modify the Atari environment to accelerate the training with some tricks:
        Episode termination: Make end-of-life == end-of-episode, but only reset on true game over. Done by DeepMind for the DQN and co. since it helps value estimation.
        Frame skipping: Return only every `skip`-th frame.
        Observation resize: Warp frames from 210x160 to 84x84 as done in the Nature paper and later work.
        Frame Stacking: Stack k last frames. Returns lazy array, which is much more memory efficient.
    Args:
        env_id: The environment id of Atari, such as "Breakout-v5", "Pong-v5", etc.
        seed: random seed.
        obs_type: This argument determines what observations are returned by the environment. Its values are:
                    ram: The 128 Bytes of RAM are returned
                    rgb: An RGB rendering of the game is returned
                    grayscale: A grayscale rendering is returned
        frame_skip: int or a tuple of two ints. This argument controls stochastic frame skipping, as described in the section on stochasticity.
        num_stack: int, the number of stacked frames if you use the frame stacking trick.
        image_size: This argument determines the size of observation image, default is [210, 160].
    """
    def __init__(self,
                 env_id: str,
                 seed: int,
                 render_mode: str,
                 obs_type: str,
                 frame_skip: int,
                 num_stack: int,
                 image_size: Sequence[int],
                 ):
        self.env = gym.make(env_id,
                            render_mode=render_mode,
                            obs_type=obs_type,
                            frameskip=frame_skip)
        self.env.action_space.seed(seed=seed)
        self.env.reset(seed=seed)
        super(Atari_Env, self).__init__(self.env)
        # self.env.seed(seed)
        self.num_stack = num_stack
        self.obs_type = obs_type
        self.frames = deque([], maxlen=self.num_stack)
        self.image_size = image_size
        self.lifes = self.env.unwrapped.ale.lives()
        self.episode_done = False
        if self.obs_type == "rgb":
            self.observation_space = gym.spaces.Box(low=0, high=1,
                                                    shape=(image_size[0], image_size[1], 3 * self.num_stack),
                                                    dtype=self.env.observation_space.dtype)
        elif self.obs_type == "grayscale":
            self.observation_space = gym.spaces.Box(low=0, high=1,
                                                    shape=(image_size[0], image_size[1], self.num_stack),
                                                    dtype=self.env.observation_space.dtype)
        else:  # ram type
            self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata
        self.reward_range = self.env.reward_range
        self._render_mode = render_mode
        self._episode_step = 0

    def close(self):
        self.env.close()

    def reset(self):
        if self.episode_done:
            obs = self.env.reset()
        else:
            obs, _, _, _ = self.env.unwrapped.step(0)  # no-op action
        for _ in range(self.num_stack):
            self.frames.append(self.observation(obs))
        self._episode_step = 0
        self.lifes = self.env.unwrapped.ale.lives()
        self.episode_done = False
        return self._get_obs(), {}

    def step(self, actions):
        observation, reward, terminated, info = self.env.unwrapped.step(actions)
        self._episode_step += 1
        self.frames.append(self.observation(observation))
        lives = self.env.unwrapped.ale.lives()
        self.episode_done = terminated
        if (lives < self.lifes) and (lives > 0):
            terminated = True
        if self._episode_step >= self.env._max_episode_steps:
            truncated = True
        else:
            truncated = False
        self.lifes = lives
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        assert len(self.frames) == self.num_stack
        return LazyFrames(list(self.frames))

    def observation(self, frame):
        if self.obs_type == "grayscale":
            return np.expand_dims(cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA), -1)
        elif self.obs_type == "rgb":
            return cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA)
        else:
            return frame


class LazyFrames(object):
    """
    This object ensures that common frames between the observations are only stored once.
    It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay buffers.
    This object should only be converted to numpy array before being passed to the model.
    """

    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[..., i]
