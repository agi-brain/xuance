from xuance.environment.pettingzoo.pettingzoo_vec_env import SubprocVecEnv_Pettingzoo, DummyVecEnv_Pettingzoo
import numpy as np


class SubprocVecEnv_Magent(SubprocVecEnv_Pettingzoo):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, context="spawn"):
        super(SubprocVecEnv_Magent, self).__init__(env_fns, context="spawn")
        self.buf_obs = [np.zeros((self.num_envs, n, np.prod(self.obs_shapes[h])), dtype=self.obs_dtype) for h, n in
                        enumerate(self.n_agents)]


class DummyVecEnv_MAgent(DummyVecEnv_Pettingzoo):
    def __init__(self, env_fns):
        super(DummyVecEnv_MAgent, self).__init__(env_fns)
        self.buf_obs = [np.zeros((self.num_envs, n, np.prod(self.obs_shapes[h])), dtype=self.obs_dtype) for h, n in
                        enumerate(self.n_agents)]
