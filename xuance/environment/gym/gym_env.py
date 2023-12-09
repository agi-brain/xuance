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

    def __init__(self, env_id: str, seed: int, render_mode: str, **kwargs):
        self.env = gym.make(env_id, render_mode=render_mode, **kwargs)
        self.env.action_space.seed(seed=seed)
        self.env.reset(seed=seed)
        super(Gym_Env, self).__init__(self.env)
        # self.env.seed(seed)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata
        self.reward_range = self.env.reward_range
        self.max_episode_steps = self.env._max_episode_steps
        self._episode_step = 0
        self._episode_score = 0.0

    def close(self):
        self.env.close()

    def render(self, mode):
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
    def __init__(self, env_id: str, seed: int, render_mode: str):
        super(MountainCar, self).__init__(env_id, seed, render_mode)
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
        noop_max: max times of noop action for env.reset().
    """

    def __init__(self,
                 env_id: str,
                 seed: int,
                 render_mode: str = "rgb_array",
                 obs_type: str = "grayscale",
                 frame_skip: int = 4,
                 num_stack: int = 4,
                 image_size: Sequence[int] = None,
                 noop_max: int = 30,
                 ):
        self.env = gym.make(env_id,
                            render_mode=render_mode,
                            obs_type=obs_type,
                            frameskip=frame_skip)
        self.env.action_space.seed(seed=seed)
        self.env.unwrapped.reset(seed=seed)
        self.max_episode_steps = self.env._max_episode_steps
        super(Atari_Env, self).__init__(self.env)
        # self.env.seed(seed)
        self.num_stack = num_stack
        self.obs_type = obs_type
        self.frames = deque([], maxlen=self.num_stack)
        self.image_size = [210, 160] if image_size is None else image_size
        self.noop_max = noop_max
        self.lifes = self.env.unwrapped.ale.lives()
        self.was_real_done = True
        self.grayscale, self.rgb = False, False
        if self.obs_type == "rgb":
            self.rgb = True
            self.observation_space = gym.spaces.Box(low=0, high=255,
                                                    shape=(image_size[0], image_size[1], 3 * self.num_stack),
                                                    dtype=np.uint8)
        elif self.obs_type == "grayscale":
            self.grayscale = True
            self.observation_space = gym.spaces.Box(low=0, high=255,
                                                    shape=(image_size[0], image_size[1], self.num_stack),
                                                    dtype=np.uint8)
        else:  # ram type
            self.observation_space = self.env.observation_space
        # assert self.env.unwrapped.get_action_meanings()[0] == "NOOP"
        # assert self.env.unwrapped.get_action_meanings()[1] == "FIRE"
        # assert len(self.env.unwrapped.get_action_meanings()) >= 3
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata
        self.reward_range = self.env.reward_range
        self._render_mode = render_mode
        self._episode_step = 0
        self._episode_score = 0.0

    def close(self):
        self.env.close()

    def render(self, render_mode):
        return self.env.unwrapped.render(render_mode)

    def reset(self):
        info = {}
        if self.was_real_done:
            self.env.unwrapped.reset()
            # Execute NoOp actions
            num_noops = np.random.randint(0, self.noop_max)
            for _ in range(num_noops):
                obs, _, done, _ = self.env.unwrapped.step(0)
                if done:
                    self.env.unwrapped.reset()
            # try to fire
            obs, _, done, _ = self.env.unwrapped.step(1)
            if done:
                obs = self.env.unwrapped.reset()
            # stack reset observations
            for _ in range(self.num_stack):
                self.frames.append(self.observation(obs))

            self._episode_step = 0
            self._episode_score = 0.0
            info["episode_step"] = 0
        else:
            obs, _, done, _ = self.env.unwrapped.step(0)
            for _ in range(self.num_stack):
                self.frames.append(self.observation(obs))

        self.lifes = self.env.unwrapped.ale.lives()
        self.was_real_done = False
        return self._get_obs(), info

    def step(self, actions):
        observation, reward, terminated, info = self.env.unwrapped.step(actions)
        self.frames.append(self.observation(observation))
        lives = self.env.unwrapped.ale.lives()
        # avoid environment bug
        if self._episode_step >= self.max_episode_steps:
            terminated = True
        self.was_real_done = terminated
        if (lives < self.lifes) and (lives > 0):
            terminated = True
        truncated = self.was_real_done
        self.lifes = lives
        self._episode_step += 1
        self._episode_score += reward
        info["episode_score"] = self._episode_score
        info["episode_step"] = self._episode_step
        return self._get_obs(), self.reward(reward), terminated, truncated, info

    def _get_obs(self):
        assert len(self.frames) == self.num_stack
        return LazyFrames(list(self.frames))

    def observation(self, frame):
        if self.grayscale:
            return np.expand_dims(cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA), -1)
        elif self.rgb:
            return cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA)
        else:
            return frame

    def reward(self, reward):
        return np.sign(reward)


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
