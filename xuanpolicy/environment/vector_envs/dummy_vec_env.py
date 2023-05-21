from .vector_env import VecEnv, AlreadySteppingError, NotSteppingError
from .env_utils import obs_n_space_info
from xuanpolicy.common import space2shape, combined_shape
from gym.spaces import Dict
from operator import itemgetter
import numpy as np
import time


class DummyVecEnv(VecEnv):
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
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None

    def reset(self):
        for e in range(self.num_envs):
            obs = self.envs[e].reset()
            self._save_obs(e, obs)
        return self.buf_obs.copy()

    def step_async(self, actions):
        if self.waiting == True:
            raise AlreadySteppingError
        listify = True
        try:
            if len(actions) == self.num_envs:
                listify = False
        except TypeError:
            pass
        if listify == False:
            self.actions = actions
        else:
            assert self.num_envs == 1, "actions {} is either not a list or has a wrong size - cannot match to {} environments".format(
                actions, self.num_envs)
            self.actions = [actions]
        self.waiting = True

    def step_wait(self):
        if self.waiting == False:
            raise NotSteppingError
        for e in range(self.num_envs):
            action = self.actions[e]
            obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[e].step(action)
            if self.buf_dones[e]:
                obs = self.envs[e].reset()
            self._save_obs(e, obs)
        self.waiting = False
        return self.buf_obs.copy(), self.buf_rews.copy(), self.buf_dones.copy(), self.buf_infos.copy()

    def close_extras(self):
        self.closed = True
        for env in self.envs:
            env.close()

    def get_images(self):
        return [env.render("rgb_array") for env in self.envs]

    def render(self, mode='human'):
        return super().render(mode=mode)

    # save observation of indexes of e environment
    def _save_obs(self, e, obs):
        if isinstance(self.observation_space, Dict):
            for k in self.obs_shape.keys():
                self.buf_obs[k][e] = obs[k]
        else:
            self.buf_obs[e] = obs


class DummyVecEnv_MAS(DummyVecEnv):
    def __init__(self, env_fns):
        self.waiting = False
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        self.handles = env.handles
        VecEnv.__init__(self, len(env_fns), env.observation_spaces, env.action_spaces)
        self.state_space = env.state_space
        obs_n_space = env.observation_spaces  # [Box(dim_o), Box(dim_o), ...] ----> dict
        self.agent_ids = env.agent_ids
        self.n_agents = [env.get_num(h) for h in self.handles]
        # self.agent_keys = [env.get_agent_key(h) for h in self.handles]

        self.keys, self.shapes, self.dtypes = obs_n_space_info(obs_n_space)
        self.agent_keys = [[self.keys[k] for k in ids] for ids in self.agent_ids]
        self.n_agent_all = len(self.keys)
        # max_obs_shape = self._get_max_obs_shape(self.keys, self.observation_space)
        self.obs_shapes = [self.shapes[self.agent_keys[h.value][0]] for h in self.handles]
        self.obs_dtype = self.dtypes[self.keys[0]]

        # buffer of dict data
        self.buf_obs_dict = [{k: np.zeros(tuple(self.shapes[k]), dtype=self.dtypes[k]) for k in self.keys} for _ in
                             range(self.num_envs)]
        self.buf_rews_dict = [{k: 0.0 for k in self.keys} for _ in range(self.num_envs)]
        self.buf_dones_dict = [{k: False for k in self.keys} for _ in range(self.num_envs)]
        self.buf_infos_dict = [{k: {} for k in self.keys} for _ in range(self.num_envs)]
        # buffer of numpy data
        self.buf_obs = [np.zeros((self.num_envs, n) + tuple(self.obs_shapes[h]), dtype=self.obs_dtype) for h, n in
                        enumerate(self.n_agents)]
        self.buf_rews = [np.zeros((self.num_envs, n, 1), dtype=np.float32) for n in self.n_agents]
        self.buf_dones = [np.ones((self.num_envs, n), dtype=np.bool) for n in self.n_agents]
        self.buf_infos = [[None for _ in range(self.num_envs)] for _ in self.n_agents]

        self.actions = None

    def empty_dict_buffers(self, i_env):
        # buffer of dict data
        self.buf_obs_dict[i_env] = {k: np.zeros(tuple(self.shapes[k]), dtype=self.dtypes[k]) for k in self.keys}
        self.buf_rews_dict[i_env] = {k: 0.0 for k in self.keys}
        self.buf_dones_dict[i_env] = {k: False for k in self.keys}
        self.buf_infos_dict[i_env] = {k: {} for k in self.keys}

    def reset(self):
        for e in range(self.num_envs):
            o = self.envs[e].reset()
            self.buf_obs_dict[e].update(o[0])
            for h, agent_keys_h in enumerate(self.agent_keys):
                self.buf_obs[h][e] = itemgetter(*agent_keys_h)(self.buf_obs_dict[e])

        return self.buf_obs

    def reset_one_env(self, e):
        o = self.envs[e].reset()
        self.buf_obs_dict[e].update(o)
        obs_e = []
        for h, agent_keys_h in enumerate(self.agent_keys):
            self.buf_obs[h][e] = itemgetter(*agent_keys_h)(self.buf_obs_dict[e])
            obs_e.append(self.buf_obs[h][e])

        return obs_e

    def _get_max_obs_shape(self, k, observation_shape):
        obs_shape_n = itemgetter(*list(k))(observation_shape)
        size_obs_n = []
        for shape in obs_shape_n:
            size_obs_n.append(shape.shape)
        return max(size_obs_n)

    def step_async(self, actions):
        if self.waiting == True:
            raise AlreadySteppingError
        listify = True
        try:
            if len(actions) == self.num_envs:
                listify = False
        except TypeError:
            pass
        if listify == False:
            self.actions = actions
        else:
            assert self.num_envs == 1, "actions {} is either not a list or has a wrong size - cannot match to {} environments".format(
                actions, self.num_envs)
            self.actions = [actions]

    def step_wait(self):
        done_all = []

        for e in range(self.num_envs):
            action_n = self.actions[e]
            o, r, d, t, info = self.envs[e].step(action_n)
            if len(o.keys()) < self.n_agent_all:
                self.empty_dict_buffers(e)
            # update the data of alive agents
            self.buf_obs_dict[e].update(o)
            self.buf_rews_dict[e].update(r)
            self.buf_dones_dict[e].update(d)
            self.buf_infos_dict[e].update(info)
            # resort the data as group-wise
            for h, agent_keys_h in enumerate(self.agent_keys):
                getter = itemgetter(*agent_keys_h)
                self.buf_obs[h][e] = getter(self.buf_obs_dict[e])
                self.buf_rews[h][e, :, 0] = getter(self.buf_rews_dict[e])
                self.buf_dones[h][e] = getter(self.buf_dones_dict[e])
                self.buf_infos[h][e] = getter(self.buf_infos_dict[e])

            try:
                done_all.append(all(itemgetter(*list(self.keys))(self.buf_dones_dict[e])))
            except:
                done_all.append(itemgetter(*list(self.keys))(self.buf_dones_dict[e]))

        return self.buf_obs, self.buf_rews, self.buf_dones, done_all, self.buf_infos

    def render(self, time_delay=0.0, mode='human'):
        time.sleep(time_delay)
        return [env.render(mode) for env in self.envs]

    def global_state(self):
        return np.array([env.state() for env in self.envs])

    def global_state_one_env(self, e):
        return np.array(self.envs[e].state())

    def agent_mask(self):
        agent_mask = [np.ones([self.num_envs, n], dtype=np.bool) for n in self.n_agents]
        for e, env in enumerate(self.envs):
            mask = env.get_agent_mask()
            for h, ids in enumerate(self.agent_ids):
                agent_mask[h][e] = mask[ids]

        return agent_mask
