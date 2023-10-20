from .vector_env import VecEnv
import numpy as np
import multiprocessing as mp
import os
import contextlib


@contextlib.contextmanager
def clear_mpi_env_vars():
    """
    from mpi4py import MPI will call MPI_Init by default.  If the child process has MPI environment variables, MPI will think that the child process is an MPI process just like the parent and do bad things such as hang.
    This context manager is a hacky way to clear those environment variables temporarily such as when we are starting multiprocessing
    Processes.
    """
    removed_environment = {}
    for k, v in list(os.environ.items()):
        for prefix in ['OMPI_', 'PMI_']:
            if k.startswith(prefix):
                removed_environment[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_environment)


def flatten_list(l):
    assert isinstance(l, (list, tuple))
    assert len(l) > 0
    assert all([len(l_) > 0 for l_ in l])
    return [l__ for l_ in l for l__ in l_]


def flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0
    if isinstance(obs[0], dict):
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def worker(remote, parent_remote, env_fn_wrappers):
    def step_env(env, action):
        ob, reward, done, info = env.step(action)
        if done:
            ob = env.reset()
        return ob, reward, done, info

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
                remote.send([env.render(mode) for env, mode in zip(envs, data)])
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send(CloudpickleWrapper((envs[0].observation_space, envs[0].action_space)))
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

    def __init__(self, env_fns, spaces=None, context='spawn', in_series=1):
        """
        Arguments:
        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        in_series: number of environments to run in series in a single process
        (e.g. when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series)
        """
        self.waiting = False
        self.closed = False
        self.in_series = in_series
        nenvs = len(env_fns)
        assert nenvs % in_series == 0, "Number of envs must be divisible by number of envs to run in series"
        self.nremotes = nenvs // in_series
        env_fns = np.array_split(env_fns, self.nremotes)
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.nremotes)])
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
        self.viewer = None
        VecEnv.__init__(self, nenvs, observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        actions = np.array_split(actions, self.nremotes)
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        results = flatten_list(results)
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        obs = flatten_list(obs)
        return flatten_obs(obs)

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        imgs = flatten_list(imgs)
        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def __del__(self):
        if not self.closed:
            self.close()
