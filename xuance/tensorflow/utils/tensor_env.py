import numpy as np
from typing import Union

import tensorflow as tf
from tensorflow import Tensor
from xuance.environment.vector_envs import DummyVecEnv, SubprocVecEnv, DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv


class TensorEnvWrapper:
    def __init__(
            self,
            vec_envs: Union[DummyVecEnv, SubprocVecEnv],
    ):
        self.envs = vec_envs
        self.num_envs = vec_envs.num_envs
        self.observation_space = vec_envs.observation_space
        self.action_space = vec_envs.action_space

    def _to_tensor(self, data, dtype=tf.float32):
        if isinstance(data, dict):
            return {k: self._to_tensor(v) for k, v in data.items()}
        else:
            return tf.convert_to_tensor(data, dtype=dtype)

    @property
    def max_episode_steps(self):
        return self.envs.max_episode_steps

    @property
    def buf_obs(self):
        return self._to_tensor(self.envs.buf_obs)

    def reset(self) -> (Tensor, dict):
        obs, infos = self.envs.reset()
        return self._to_tensor(obs), infos

    def step(self, actions: Union[Tensor, np.ndarray]) -> (Tensor, Tensor, Tensor, Tensor, dict):
        if isinstance(actions, Tensor):
            actions_np = actions.numpy()
        else:
            actions_np = actions
        next_obs, rewards, terminals, truncations, infos = self.envs.step(actions_np)
        for e in range(self.num_envs):
            if terminals[e] or truncations[e]:
                infos[e]["reset_obs"] = self._to_tensor(np.array(infos[e]["reset_obs"]))
        return (self._to_tensor(next_obs),
                self._to_tensor(rewards),
                self._to_tensor(terminals),
                self._to_tensor(truncations),
                infos)

    def render(self, *args, **kwargs):
        return self.envs.render(*args, **kwargs)


class TensorMultiAgentEnvWrapper:
    def __init__(
            self,
            vec_envs: Union[DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv],
    ):
        self.envs = vec_envs

    def reset(self):
        # Convert numpy data to tensor data.
        return self.envs.reset()

    def step(self, actions):
        # Convert tensor data to numpy data, and then convert numpy data to tensor data.
        return self.envs.step(actions)
