from xuance.common import combined_shape
from gymnasium.spaces import Discrete, Box
import numpy as np
import multiprocessing as mp
from xuance.environment.vector_envs.subproc_vec_env import clear_mpi_env_vars, flatten_list, CloudpickleWrapper
from xuance.environment.vector_envs.vector_env import VecEnv


def worker(remote, parent_remote, env_fn_wrappers):
    def step_env(env, action):
        obs, state, reward_n, terminated, truncated, info = env.step(action)
        return obs, state, reward_n, terminated, truncated, info

    parent_remote.close()
    envs = [env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers.x]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send([step_env(env, action) for env, action in zip(envs, data)])
            elif cmd == 'get_avail_actions':
                remote.send([env.get_avail_actions() for env in envs])
            elif cmd == 'reset':
                remote.send([env.reset() for env in envs])
            elif cmd == 'render':
                remote.send([env.render(data) for env in envs])
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_env_info':
                remote.send(CloudpickleWrapper((envs[0].env_info, envs[0].n_enemies)))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        for env in envs:
            env.close()


class SubprocVecEnv_StarCraft2(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """

    def __init__(self, env_fns, context='spawn'):
        """
        Arguments:
        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        (e.g. when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series)
        """
        self.waiting = False
        self.closed = False
        self.n_remotes = num_envs = len(env_fns)
        env_fns = np.array_split(env_fns, self.n_remotes)
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.n_remotes)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_env_info', None))
        env_info, self.num_enemies = self.remotes[0].recv().x
        self.dim_obs = env_info["obs_shape"]
        self.dim_act = self.n_actions = env_info["n_actions"]
        observation_space, action_space = (self.dim_obs,), (self.dim_act,)
        self.viewer = None
        VecEnv.__init__(self, num_envs, observation_space, action_space)

        self.num_agents = env_info["n_agents"]
        self.obs_shape = (self.num_agents, self.dim_obs)
        self.act_shape = (self.num_agents, self.dim_act)
        self.rew_shape = (self.num_agents, 1)
        self.dim_obs, self.dim_state, self.dim_act = self.dim_obs, env_info["state_shape"], self.dim_act
        self.dim_reward = self.num_agents
        self.action_space = Discrete(n=self.dim_act)
        self.state_space = Box(low=-np.inf, high=np.inf, shape=[self.dim_state, ])

        self.buf_obs = np.zeros(combined_shape(self.num_envs, self.obs_shape), dtype=np.float32)
        self.buf_state = np.zeros(combined_shape(self.num_envs, self.dim_state), dtype=np.float32)
        self.buf_terminal = np.zeros((self.num_envs, 1), dtype=np.bool_)
        self.buf_truncation = np.zeros((self.num_envs, 1), dtype=np.bool_)
        self.buf_done = np.zeros((self.num_envs,), dtype=np.bool_)
        self.buf_rew = np.zeros((self.num_envs,) + self.rew_shape, dtype=np.float32)
        self.buf_info = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.battles_game = np.zeros(self.num_envs, np.int32)
        self.battles_won = np.zeros(self.num_envs, np.int32)
        self.dead_allies_count = np.zeros(self.num_envs, np.int32)
        self.dead_enemies_count = np.zeros(self.num_envs, np.int32)
        self.max_episode_length = env_info["episode_limit"]

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        result = [remote.recv() for remote in self.remotes]
        result = flatten_list(result)
        obs, state, infos = zip(*result)
        self.buf_obs, self.buf_state, self.buf_info = np.array(obs), np.array(state), list(infos)
        self.buf_done = np.zeros((self.num_envs,), dtype=np.bool_)
        return self.buf_obs.copy(), self.buf_state.copy(), self.buf_info.copy()

    def step_async(self, actions):
        self._assert_not_closed()
        actions = np.array_split(actions, self.n_remotes)
        for env_done, remote, action in zip(self.buf_done, self.remotes, actions):
            if not env_done:
                remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        if self.waiting:
            for idx_env, env_done, remote in zip(range(self.num_envs), self.buf_done, self.remotes):
                if not env_done:
                    result = remote.recv()
                    result = flatten_list(result)
                    obs, state, rew, terminal, truncated, infos = result
                    self.buf_obs[idx_env], self.buf_state[idx_env] = np.array(obs), np.array(state)
                    self.buf_rew[idx_env], self.buf_terminal[idx_env] = np.array(rew), np.array(terminal)
                    self.buf_truncation[idx_env], self.buf_info[idx_env] = np.array(truncated), infos

                    if self.buf_terminal[idx_env].all() or self.buf_truncation[idx_env].all():
                        self.buf_done[idx_env] = True
                        self.battles_game[idx_env] += 1
                        if infos['battle_won']:
                            self.battles_won[idx_env] += 1
                        self.dead_allies_count[idx_env] += infos['dead_allies']
                        self.dead_enemies_count[idx_env] += infos['dead_enemies']
                else:
                    self.buf_terminal[idx_env, 0], self.buf_truncation[idx_env, 0] = False, False

        self.waiting = False
        return self.buf_obs.copy(), self.buf_state.copy(), self.buf_rew.copy(), self.buf_terminal.copy(), self.buf_truncation.copy(), self.buf_info.copy()

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def render(self, mode):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', mode))
        imgs = [pipe.recv() for pipe in self.remotes]
        imgs = flatten_list(imgs)
        return imgs

    def get_avail_actions(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_avail_actions', None))
        avail_actions = [remote.recv() for remote in self.remotes]
        avail_actions = flatten_list(avail_actions)
        return np.array(avail_actions)

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def __del__(self):
        if not self.closed:
            self.close()


class DummyVecEnv_StarCraft2(VecEnv):
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        num_envs = len(env_fns)

        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        env_info, self.num_enemies = env.env_info, env.n_enemies
        self.dim_obs = env_info["obs_shape"]
        self.dim_act = self.n_actions = env_info["n_actions"]
        observation_space, action_space = (self.dim_obs,), (self.dim_act,)
        self.viewer = None
        VecEnv.__init__(self, num_envs, observation_space, action_space)

        self.num_agents = env_info["n_agents"]
        self.obs_shape = (self.num_agents, self.dim_obs)
        self.act_shape = (self.num_agents, self.dim_act)
        self.rew_shape = (self.num_agents, 1)
        self.dim_obs, self.dim_state, self.dim_act = self.dim_obs, env_info["state_shape"], self.dim_act
        self.dim_reward = self.num_agents
        self.action_space = Discrete(n=self.dim_act)
        self.state_space = Box(low=-np.inf, high=np.inf, shape=[self.dim_state, ])

        self.buf_obs = np.zeros(combined_shape(self.num_envs, self.obs_shape), dtype=np.float32)
        self.buf_state = np.zeros(combined_shape(self.num_envs, self.dim_state), dtype=np.float32)
        self.buf_terminal = np.zeros((self.num_envs, 1), dtype=np.bool_)
        self.buf_truncation = np.zeros((self.num_envs, 1), dtype=np.bool_)
        self.buf_done = np.zeros((self.num_envs,), dtype=np.bool_)
        self.buf_rew = np.zeros((self.num_envs,) + self.rew_shape, dtype=np.float32)
        self.buf_info = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.battles_game = np.zeros(self.num_envs, np.int32)
        self.battles_won = np.zeros(self.num_envs, np.int32)
        self.dead_allies_count = np.zeros(self.num_envs, np.int32)
        self.dead_enemies_count = np.zeros(self.num_envs, np.int32)
        self.max_episode_length = env_info["episode_limit"]

    def reset(self):
        self._assert_not_closed()
        for i_env, env in enumerate(self.envs):
            obs, state, infos = env.reset()
            self.buf_obs[i_env], self.buf_state[i_env], self.buf_info[i_env] = np.array(obs), np.array(state), list(infos)
        self.buf_done = np.zeros((self.num_envs,), dtype=np.bool_)
        return self.buf_obs.copy(), self.buf_state.copy(), self.buf_info.copy()

    def step_async(self, actions):
        self._assert_not_closed()
        self.actions = actions
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        if self.waiting:
            for idx_env, env_done, env in zip(range(self.num_envs), self.buf_done, self.envs):
                if not env_done:
                    obs, state, rew, terminal, truncated, infos = env.step(self.actions[idx_env])
                    self.buf_obs[idx_env], self.buf_state[idx_env] = np.array(obs), np.array(state)
                    self.buf_rew[idx_env], self.buf_terminal[idx_env] = np.array(rew), np.array(terminal)
                    self.buf_truncation[idx_env], self.buf_info[idx_env] = np.array(truncated), infos

                    if self.buf_terminal[idx_env].all() or self.buf_truncation[idx_env].all():
                        self.buf_done[idx_env] = True
                        self.battles_game[idx_env] += 1
                        if infos['battle_won']:
                            self.battles_won[idx_env] += 1
                        self.dead_allies_count[idx_env] += infos['dead_allies']
                        self.dead_enemies_count[idx_env] += infos['dead_enemies']
                else:
                    self.buf_terminal[idx_env, 0], self.buf_truncation[idx_env, 0] = False, False

        self.waiting = False
        return self.buf_obs.copy(), self.buf_state.copy(), self.buf_rew.copy(), self.buf_terminal.copy(), self.buf_truncation.copy(), self.buf_info.copy()

    def close_extras(self):
        self.closed = True
        for env in self.envs:
            env.close()

    def render(self, mode):
        self._assert_not_closed()
        imgs = [env.render(mode) for env in self.envs]
        return imgs

    def get_avail_actions(self):
        self._assert_not_closed()
        avail_actions = [env.get_avail_actions() for env in self.envs]
        return np.array(avail_actions)

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def __del__(self):
        if not self.closed:
            self.close()
