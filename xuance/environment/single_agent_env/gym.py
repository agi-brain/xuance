import gymnasium as gym
import numpy as np
from collections import deque
try:
    import cv2
except ImportError:
    print("The module opencv-python might not be installed."
          "Please ensure you have installed opencv-python via `pip install opencv-python==4.5.4.58`.")


class Gym_Env(gym.Wrapper):
    """
    Args:
        env_id (str): The environment id of Atari, such as "Breakout-v5", "Pong-v5", etc.
        env_seed (int): The random seed to set the environment.
        render_mode (str): "rgb_array", "human"
    """

    def __init__(self, config, **kwargs):
        if config.env_id == "CarRacing-v2":
            kwargs['continuous'] = False
        self.env = gym.make(config.env_id, render_mode=config.render_mode, **kwargs)
        self.env.action_space.seed(seed=config.env_seed)
        self.env.reset(seed=config.env_seed)
        super(Gym_Env, self).__init__(self.env)
        # self.env.seed(config.env_seed)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata
        self.max_episode_steps = self.env._max_episode_steps

    def render(self, *args):
        return self.env.render()

    def reset(self):
        obs, info = self.env.reset()
        self._episode_step = 0
        self._episode_score = 0.0
        info["episode_step"] = self._episode_step
        return obs, info

    def step(self, actions):
        observation, reward, terminated, truncated, info = self.env.step(actions)
        self._episode_step += 1
        self._episode_score += reward
        info["episode_step"] = self._episode_step
        info["episode_score"] = self._episode_score
        return observation, reward, terminated, truncated, info


class MountainCar(Gym_Env):
    def __init__(self, env_id: str, env_seed: int, render_mode: str):
        super(MountainCar, self).__init__(env_id, env_seed, render_mode)
        self.num_stack = 4
        self.frames = deque([], maxlen=self.num_stack)
        self.observation_space = gym.spaces.Box(low=np.array([-1.2, -0.07, -1.2, -0.07, -1.2, -0.07, -1.2, -0.07]),
                                                high=np.array([0.6, 0.07, 0.6, 0.07, 0.6, 0.07, 0.6, 0.07]),
                                                shape=(8,), dtype=np.float32)
        self.pre_position = 0.0

    def reset(self):
        obs, info = self.env.reset()
        self._episode_step = 0
        self._episode_score = 0.0
        info["episode_step"] = self._episode_step
        for i in range(self.num_stack):
            self.frames.append(obs)
        self.pre_position = obs[0]
        return LazyFrames(list(self.frames)), info

    def step(self, actions):
        observation, reward, terminated, truncated, info = self.env.step(actions)
        self._episode_step += 1
        self._episode_score += reward
        info["episode_step"] = self._episode_step
        info["episode_score"] = self._episode_score

        # reward += 10 * observation[0]
        # reward + 10 * (observation[0] - self.pre_position)
        # reward += observation[1] ** 2
        self.frames.append(observation)
        self.pre_position = observation[0]

        return LazyFrames(list(self.frames)), reward, terminated, truncated, info


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
