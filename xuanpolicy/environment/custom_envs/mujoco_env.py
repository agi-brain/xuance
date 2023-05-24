ENVIRONMENT_IDS = ['Ant-v2', 'Ant-v3',
                   'HalfCheetah-v2', 'HalfCheetah-v3',
                   'Walker2d-v2', 'Walker2d-v3',
                   'Hopper-v2', 'Hopper-v3',
                   'Swimmer-v2', 'Swimmer-v3',
                   'Reacher-v2', 'Reacher-v3',
                   'Humanoid-v2', 'Humanoid-v3',
                   'InvertedPendulum-v2']
import gym
import numpy as np


class MuJoCo_Env(gym.Env):
    def __init__(self, env_id: str, seed: int):
        assert env_id in ENVIRONMENT_IDS
        self.env = gym.make(env_id)
        self.env.seed(seed)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata
        self.reward_range = self.env.reward_range
        self.spec = self.env.spec
        super(MuJoCo_Env, self).__init__()

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(self._action_transform(action))

    def seed(self, seed):
        return self.env.seed(seed)

    def render(self, mode):
        return self.env.render(mode)

    def close(self):
        self.env.close()

    def _action_transform(self, action):
        a_range = self.action_space.high - self.action_space.low
        t_action = np.clip(action, -1, 1)
        t_action = a_range * (t_action + 1) / 2. + self.action_space.low
        return t_action
