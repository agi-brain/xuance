import numpy as np
from xuance.common import combined_shape
from xuance.environment import DummyVecEnv, SubprocVecEnv


class DummyVecEnv_Atari(DummyVecEnv):
    def __init__(self, env_fns):
        super(DummyVecEnv_Atari, self).__init__(env_fns)
        self.buf_obs = np.zeros(combined_shape(self.num_envs, self.obs_shape), dtype=np.uint8)


class SubprocVecEnv_Atari(SubprocVecEnv):
    def __init__(self, env_fns):
        super(SubprocVecEnv_Atari, self).__init__(env_fns)
        self.buf_obs = np.zeros(combined_shape(self.num_envs, self.obs_shape), dtype=np.uint8)
