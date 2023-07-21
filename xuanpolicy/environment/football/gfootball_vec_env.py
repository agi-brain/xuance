from xuanpolicy.environment.vector_envs.vector_env import VecEnv, AlreadySteppingError, NotSteppingError
from xuanpolicy.common import combined_shape
from gymnasium.spaces import Discrete, Box
import numpy as np


class DummyVecEnv_GFootball(VecEnv):
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.dim_obs, env.n_actions)
        self.num_agents, self.num_adversaries = env.n_agents, env.n_adversaries
        self.obs_shape = (env.n_agents, env.dim_obs)
        self.act_shape = (env.n_agents, env.n_actions)
        self.dim_obs, self.dim_state, self.dim_act = env.dim_obs, env.dim_state, env.dim_act
        self.dim_reward = env.dim_reward
        self.action_space = Discrete(n=self.dim_act)
        self.state_space = Box(low=-np.inf, high=np.inf, shape=[self.dim_state, ])
        self.buf_obs = np.zeros(combined_shape(self.num_envs, self.obs_shape), dtype=np.float32)
        self.buf_state = np.zeros(combined_shape(self.num_envs, self.dim_state), dtype=np.float32)
        self.buf_dones = np.zeros((self.num_envs, 1), dtype=np.bool)
        self.buf_trunctions = np.zeros((self.num_envs, 1), dtype=np.bool)
        self.buf_rews = np.zeros((self.num_envs, self.num_agents, ), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.battles_game = np.zeros(self.num_envs, np.int32)
        self.battles_won = np.zeros(self.num_envs, np.int32)
        self.dead_allies_count = np.zeros(self.num_envs, np.int32)
        self.dead_enemies_count = np.zeros(self.num_envs, np.int32)
        self.max_episode_length = env.max_cycles

    def reset(self):
        for e in range(self.num_envs):
            obs, state, info = self.envs[e].reset()
            self.buf_obs[e] = obs
            self.buf_state[e] = state
            self.buf_infos[e] = info
        return self.buf_obs.copy(), self.buf_state.copy(), self.buf_infos.copy()

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
            obs, state, self.buf_rews[e], self.buf_dones[e], self.buf_trunctions[e], self.buf_infos[e] = self.envs[e].step(action)
            if self.buf_dones[e] or self.buf_trunctions[e]:
                self.battles_game[e] += 1
                if self.buf_infos[e]['score_reward'] > 0:
                    self.battles_won[e] += 1
                self.buf_infos[e]["avail_actions"] = np.ones([self.num_agents, self.dim_act], dtype=np.bool)
                obs_reset, state_reset, _ = self.envs[e].reset()
                self.buf_infos[e]["reset_obs"] = np.array(obs_reset)
                self.buf_infos[e]["reset_state"] = np.array(state_reset)
            self.buf_infos[e]["battles_game"] = self.battles_game[e]
            self.buf_infos[e]["battles_won"] = self.battles_won[e]
            self.buf_obs[e] = obs
            self.buf_state[e] = state
        self.waiting = False
        return self.buf_obs.copy(), self.buf_state.copy(), self.buf_rews.copy(), self.buf_dones.copy(), self.buf_trunctions.copy(), self.buf_infos.copy()

    def close_extras(self):
        self.closed = True
        for env in self.envs:
            env.close()

    def render(self, mode):
        return [env.render() for env in self.envs]

    def get_avail_actions(self):
        return np.ones([self.num_envs, self.num_agents, self.dim_act], dtype=np.bool)
