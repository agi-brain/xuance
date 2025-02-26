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
        self.reward_range = self.env.reward_range
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


class Atari_Env(gym.Wrapper):
    """
    We modify the Atari environment to accelerate the training with some tricks:
        Episode termination: Make end-of-life == end-of-episode, but only reset on true game over. Done by DeepMind for the DQN and co. since it helps value estimation.
        Frame skipping: Return only every `skip`-th frame.
        Observation resize: Warp frames from 210x160 to 84x84 as done in the Nature paper and later work.
        Frame Stacking: Stack k last frames. Returns lazy array, which is much more memory efficient.
    """

    def __init__(self, config):
        full_action_space = config.full_action_space if hasattr(config, 'full_action_space') else False
        self.env = gym.make(config.env_id,
                            render_mode=config.render_mode,
                            obs_type=config.obs_type,
                            frameskip=config.frame_skip,
                            full_action_space=full_action_space)
        self.env.action_space.seed(seed=config.env_seed)
        self.env.unwrapped.reset(seed=config.env_seed)
        self.max_episode_steps = self.env._max_episode_steps if hasattr(self.env, '_max_episode_steps') else 1e5
        super(Atari_Env, self).__init__(self.env)
        # self.env.seed(config.env_seed)
        self.num_stack = config.num_stack
        self.obs_type = config.obs_type
        self.frames = deque([], maxlen=self.num_stack)
        self.image_size = [210, 160] if config.img_size is None else config.img_size
        self.noop_max = config.noop_max
        self.lifes = self.env.unwrapped.ale.lives()
        self.was_real_done = True
        self.grayscale, self.rgb = False, False
        if self.obs_type == "rgb":
            self.rgb = True
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(config.img_size[0], config.img_size[1], 3 * self.num_stack), dtype=np.uint8)
        elif self.obs_type == "grayscale":
            self.grayscale = True
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(config.img_size[0], config.img_size[1], self.num_stack), dtype=np.uint8)
        else:  # ram type
            self.observation_space = self.env.observation_space
        # assert self.env.unwrapped.get_action_meanings()[0] == "NOOP"
        # assert self.env.unwrapped.get_action_meanings()[1] == "FIRE"
        # assert len(self.env.unwrapped.get_action_meanings()) >= 3
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata
        self.reward_range = self.env.reward_range
        self._render_mode = config.render_mode
        self._episode_step = 0

    def close(self):
        self.env.close()

    def render(self, *args, **kwargs):
        return self.env.render()

    def reset(self):
        info = {}
        if self.was_real_done:
            self.env.reset()
            # Execute NoOp actions
            num_noops = np.random.randint(0, self.noop_max)
            for _ in range(num_noops):
                obs, _, done, _, _ = self.env.step(0)
                if done:
                    self.env.reset()
            # try to fire
            obs, _, done, _, _ = self.env.step(1)
            if done:
                obs = self.env.reset()
            # stack reset observations
            for _ in range(self.num_stack):
                self.frames.append(self.observation(obs))

            self._episode_step = 0
        else:
            obs, _, done, _, _ = self.env.step(0)
            for _ in range(self.num_stack):
                self.frames.append(self.observation(obs))

        self.lifes = self.env.ale.lives()
        self.was_real_done = False
        return self._get_obs(), info

    def step(self, actions):
        observation, reward, terminated, truncated, info = self.env.step(actions)
        self.frames.append(self.observation(observation))
        lives = self.env.ale.lives()
        # avoid environment bug
        if self.max_episode_steps is not None:
            if self._episode_step >= self.max_episode_steps:
                terminated = True
        self.was_real_done = terminated
        if (lives < self.lifes) and (lives > 0):
            terminated = True
        truncated = self.was_real_done
        self.lifes = lives
        self._episode_step += 1
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
