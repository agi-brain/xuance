import numpy as np
from multiprocessing import Process, Pipe
from xuance.common import space2shape, combined_shape
from xuance.environment.vector_envs.vector_env import VecEnv
from xuance.environment.vector_envs import clear_mpi_env_vars, flatten_list, CloudpickleWrapper


def worker(remote, parent_remote, env_fn_wrappers):
    def step_env(env, action):
        obs, reward_n, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs_reset, _ = env.reset()
            info["reset_obs"] = obs_reset
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
                remote.send(CloudpickleWrapper(envs[0].max_episode_steps))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        for env in envs:
            env.close()


class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """

    def __init__(self, env_fns, in_series=1):
        """
        Arguments:
        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        in_series: number of environments to run in series in a single process
        (e.g. when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series)
        """
        self.waiting = False
        self.closed = False
        num_envs = len(env_fns)
        self.n_remotes = num_envs // in_series
        env_fns = np.array_split(env_fns, self.n_remotes)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_remotes)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv().x
        self.viewer = None
        VecEnv.__init__(self, num_envs, observation_space, action_space)

        self.obs_shape = space2shape(self.observation_space)
        if isinstance(self.observation_space, dict):
            self.buf_obs = {k: np.zeros(combined_shape(self.num_envs, v)) for k, v in
                            zip(self.obs_shape.keys(), self.obs_shape.values())}
        else:
            self.buf_obs = np.zeros(combined_shape(self.num_envs, self.obs_shape), dtype=np.float32)

        self.actions = None
        self.remotes[0].send(('get_max_cycles', None))
        self.max_episode_steps = self.remotes[0].recv().x

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        result = [remote.recv() for remote in self.remotes]
        result = flatten_list(result)
        obs, info = zip(*result)
        self.buf_obs = np.array(obs)
        return np.array(obs), list(info)

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
        self.waiting = False
        obs, rewards, terminated, truncated, info = zip(*results)
        self.buf_obs = np.array(obs)
        return np.array(obs), np.array(rewards), np.array(terminated), np.array(truncated), list(info)

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


class SubprocVecEnv_Atari(SubprocVecEnv):
    def __init__(self, env_fns):
        super(SubprocVecEnv_Atari, self).__init__(env_fns)
        self.buf_obs = np.zeros(combined_shape(self.num_envs, self.obs_shape), dtype=np.uint8)
