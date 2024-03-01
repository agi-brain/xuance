from xuance.environment.vector_envs.vector_env import VecEnv, AlreadySteppingError, NotSteppingError
from xuance.common import space2shape, combined_shape
from gym.spaces import Dict
import numpy as np
import multiprocessing as mp
from xuance.environment.vector_envs.subproc_vec_env import clear_mpi_env_vars, flatten_list, CloudpickleWrapper


def worker(remote, parent_remote, env_fn_wrappers):
    def step_env(env, action):
        obs, reward_n, terminated, truncated, info = env.step(action)
        return obs, reward_n, terminated, truncated, info

    parent_remote.close()
    envs = [env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers.x]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send([step_env(env, action) for env, action in zip(envs, data)])
            elif cmd == 'reset':
                remote.send([env.reset() for env in envs])
            elif cmd == 'render':
                remote.send([env.render(data) for env in envs])
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send(CloudpickleWrapper((envs[0].observation_space, envs[0].action_space)))
            elif cmd == 'get_max_cycles':
                remote.send(CloudpickleWrapper((envs[0].max_episode_steps)))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        for env in envs:
            env.close()


class SubprocVecEnv_Gym(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, context='spawn', in_series=1):
        """
        Arguments:
        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        in_series: number of environments to run in series in a single process
        (e.g. when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series)
        """
        self.waiting = False
        self.closed = False
        self.in_series = in_series
        num_envs = len(env_fns)
        assert num_envs % in_series == 0, "Number of envs must be divisible by number of envs to run in series"
        self.n_remotes = num_envs // in_series
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

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv().x
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

        self.obs_shape = space2shape(self.observation_space)
        if isinstance(self.observation_space, Dict):
            self.buf_obs = {k: np.zeros(combined_shape(self.num_envs, v)) for k, v in
                            zip(self.obs_shape.keys(), self.obs_shape.values())}
        else:
            self.buf_obs = np.zeros(combined_shape(self.num_envs, self.obs_shape), dtype=np.float32)
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool_)
        self.buf_trunctions = np.zeros((self.num_envs,), dtype=np.bool_)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.remotes[0].send(('get_max_cycles', None))
        self.max_episode_length = self.remotes[0].recv().x

    def step_async(self, actions):
        self._assert_not_closed()
        actions = np.array_split(actions, self.n_remotes)
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        results = flatten_list(results)
        obs, rews, dones, truncated, infos = zip(*results)
        self.buf_obs, self.buf_rews = np.array(obs), np.array(rews)
        self.buf_dones, self.buf_trunctions, self.buf_infos = np.array(dones), np.array(truncated), list(infos)
        for e in range(self.num_envs):
            if self.buf_dones[e] or self.buf_trunctions[e]:
                self.remotes[e].send(('reset', None))
                reset_result = self.remotes[e].recv()
                obs_reset, _ = zip(*reset_result)
                self.buf_infos[e]["reset_obs"] = np.array(obs_reset)
        self.waiting = False
        return self.buf_obs.copy(), self.buf_rews.copy(), self.buf_dones.copy(), self.buf_trunctions.copy(), self.buf_infos.copy()

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        result = [remote.recv() for remote in self.remotes]
        result = flatten_list(result)
        obs, infos = zip(*result)
        self.buf_obs, self.buf_infos = np.array(obs), list(infos)
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool_)
        return self.buf_obs.copy(), self.buf_infos.copy()

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

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def __del__(self):
        if not self.closed:
            self.close()


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
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool_)
        self.buf_trunctions = np.zeros((self.num_envs,), dtype=np.bool_)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        try:
            self.max_episode_length = env.max_episode_steps
        except AttributeError:
            self.max_episode_length=1000

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

    def render(self, mode):
        return [env.render(mode) for env in self.envs]

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
        self.buf_obs = np.zeros(combined_shape(self.num_envs, self.obs_shape), dtype=np.uint8)


class SubprocVecEnv_Atari(SubprocVecEnv_Gym):
    def __init__(self, env_fns):
        super(SubprocVecEnv_Atari, self).__init__(env_fns)
        self.buf_obs = np.zeros(combined_shape(self.num_envs, self.obs_shape), dtype=np.uint8)
