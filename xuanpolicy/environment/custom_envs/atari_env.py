# Note: THE tricks use for atari-wrappers are important to the benchmark results #
# Tricks includes: FireReset,
#                  NoOpReset,
#                  ResizeImage
#                  StackImage
#                  ActionRepeat
#                  MaxImage
# Referenced from openai/baselines

ENVIRONMENT_IDS = ['PongNoFrameskip-v4',
                   'BreakoutNoFrameskip-v4']

IMAGE_SIZE = 84
STACK_SIZE = 4
IMAGE_CHANNEL = 1
ACTION_REPEAT = 4

import gym
import cv2
import numpy as np
from gym.spaces import Space, Box, Discrete, Dict


class Atari_Env(gym.Env):
    def __init__(self, env_id: str, seed: int):
        assert env_id in ENVIRONMENT_IDS
        self.env = gym.make(env_id)
        self.env.seed(seed)
        self.observation_space = Box(0, 1, (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL * STACK_SIZE))
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata
        self.reward_range = self.env.reward_range
        self.spec = self.env.spec
        super(Atari_Env, self).__init__()

    def _process_reset_image(self, image):
        resize_image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        if IMAGE_CHANNEL == 1:
            resize_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)
            resize_image = np.expand_dims(resize_image, axis=2)
        resize_image = resize_image.astype(np.float32) / 255.0
        self.stack_image = np.tile(resize_image, (1, 1, STACK_SIZE))
        return self.stack_image

    def _process_step_image(self, image):
        resize_image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        if IMAGE_CHANNEL == 1:
            resize_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)
            resize_image = np.expand_dims(resize_image, axis=2)
        resize_image = resize_image.astype(np.float32) / 255.0
        self.stack_image = np.concatenate((self.stack_image[:, :, :-IMAGE_CHANNEL], resize_image), axis=2)
        return self.stack_image

    # FireReset
    # NoOpReset
    def reset(self):
        obs = self.env.reset()
        self.lives = self.env.unwrapped.ale.lives()
        if self.env.unwrapped.get_action_meanings()[1] == 'FIRE':
            obs, _, done, _ = self.env.step(1)
            if done:
                obs = self.env.reset()
        noop = np.random.randint(0, 30)
        for i in range(noop):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
                break
        return self._process_reset_image(obs)

    def step(self, action):
        cum_reward = 0
        last_image = np.zeros(self.env.observation_space.shape, np.uint8)
        for i in range(ACTION_REPEAT):
            obs, rew, done, info = self.env.step(action)
            cum_reward += rew
            concat_image = np.concatenate((np.expand_dims(last_image, axis=0), np.expand_dims(obs, axis=0)), axis=0)
            max_image = np.max(concat_image, axis=0)
            last_image = obs
            done = (done or (self.lives > self.env.unwrapped.ale.lives()))
            if done:
                break
        return self._process_step_image(max_image), cum_reward, done, info

    def seed(self, seed):
        return self.env.seed(seed)

    def render(self, mode):
        return self.env.render(mode)

    def close(self):
        self.env.close()
