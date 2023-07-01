from xuanpolicy.environment.vector_envs.vector_env import VecEnv, AlreadySteppingError, NotSteppingError
from xuanpolicy.common import space2shape, combined_shape
from gym.spaces import Dict
import numpy as np


class DummyVecEnv_Gym(VecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    """
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.obs_shape = space2shape(self.observation_space)
        if isinstance(self.observation_space, Dict):
            self.buf_obs = {k: np.zeros(combined_shape(self.num_envs, v)) for k, v in
                            zip(self.obs_shape.keys(), self.obs_shape.values())}
        else:
            self.buf_obs = np.zeros(combined_shape(self.num_envs, self.obs_shape), dtype=np.float32)
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_trunctions = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.max_episode_length = env.env._max_episode_steps

    def reset(self):
        for e in range(self.num_envs):
            obs, info = self.envs[e].reset()
            self._save_obs(e, obs)
            self._save_infos(e, info)
        return self.buf_obs.copy(), self.buf_infos.copy()

    def step_async(self, actions):
        if self.waiting:
            raise AlreadySteppingError
        listify = True
        try:
            if len(actions) == self.num_envs:
                listify = False
        except TypeError:
            pass
        if not listify:
            self.actions = actions
        else:
            assert self.num_envs == 1, "actions {} is either not a list or has a wrong size - cannot match to {} environments".format(
                actions, self.num_envs)
            self.actions = [actions]
        self.waiting = True

    def step_wait(self):
        if not self.waiting:
            raise NotSteppingError
        for e in range(self.num_envs):
            action = self.actions[e]
            obs, self.buf_rews[e], self.buf_dones[e], self.buf_trunctions[e], self.buf_infos[e] = self.envs[e].step(action)
            if self.buf_dones[e] or self.buf_trunctions[e]:
                obs_reset, _ = self.envs[e].reset()
                self.buf_infos[e]["reset_obs"] = obs_reset
            self._save_obs(e, obs)
        self.waiting = False
        return self.buf_obs.copy(), self.buf_rews.copy(), self.buf_dones.copy(), self.buf_trunctions.copy(), self.buf_infos.copy()

    def close_extras(self):
        self.closed = True
        for env in self.envs:
            env.close()

    def get_images(self):
        return [env.render("rgb_array") for env in self.envs]

    def render(self, mode):
        return [env.render(mode) for env in self.envs]
        # return super().render(mode=mode)

    # save observation of indexes of e environment
    def _save_obs(self, e, obs):
        if isinstance(self.observation_space, Dict):
            for k in self.obs_shape.keys():
                self.buf_obs[k][e] = obs[k]
        else:
            self.buf_obs[e] = obs

    def _save_infos(self, e, info):
        self.buf_infos[e] = info


class DummyVecEnv_Atari(DummyVecEnv_Gym):
    def __init__(self, env_fns):
        super(DummyVecEnv_Atari, self).__init__(env_fns)
        if isinstance(self.observation_space, Dict):
            self.buf_obs = {k: np.zeros(combined_shape(self.num_envs, v), dtype=np.uint8)
                            for k, v in zip(self.obs_shape.keys(), self.obs_shape.values())}
        else:
            self.buf_obs = np.zeros(combined_shape(self.num_envs, self.obs_shape), dtype=np.uint8)
