import numpy as np
import multiprocessing as mp
from xuance.common import space2shape
from xuance.environment.vector_envs.vector_env import VecEnv
from xuance.environment.vector_envs import clear_mpi_env_vars, flatten_list, CloudpickleWrapper


def worker(remote, parent_remote, env_fn_wrappers):
    def step_env(env, action):
        obs, reward_n, terminated, truncated, info = env.step(action)
        if all(terminated.values()) or truncated:
            obs_reset, info_reset = env.reset()
            info["reset_obs"] = obs_reset
            info["reset_avail_actions"] = info_reset['avail_actions']
            info["reset_state"] = info_reset['state']
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
            elif cmd == 'get_env_info':
                env_info = {
                    'state_space': envs[0].state_space,
                    'observation_space': envs[0].observation_space,
                    'action_space': envs[0].action_space,
                    'agents': envs[0].agents,
                    'num_agents': envs[0].num_agents,
                    'max_episode_steps': envs[0].max_episode_steps,
                }
                remote.send(CloudpickleWrapper(env_info))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        for env in envs:
            env.close()


class SubprocVecMultiAgentEnv(VecEnv):
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

        self.remotes[0].send(('get_env_info', None))
        env_info = self.remotes[0].recv().x
        self.viewer = None
        VecEnv.__init__(self, num_envs, env_info['observation_space'], env_info['action_space'])

        self.agents = env_info['agents']
        self.num_agents = env_info['num_agents']
        self.state_space = env_info['state_space']  # Type: Box
        self.buf_state = [np.zeros(space2shape(self.state_space)) for _ in range(self.num_envs)]
        self.buf_obs = [{} for _ in range(self.num_envs)]
        self.buf_avail_actions = [{} for _ in range(self.num_envs)]

        self.actions = None
        self.max_episode_steps = env_info['max_episode_steps']

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        result = [remote.recv() for remote in self.remotes]
        result = flatten_list(result)
        obs, info = zip(*result)
        self.buf_obs = list(obs)
        self.buf_state = [info[e]['state'] for e in range(self.num_envs)]
        self.buf_avail_actions = [info[e]['avail_actions'] for e in range(self.num_envs)]
        return list(obs), list(info)

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
        self.buf_obs = list(obs)
        self.buf_state = [info[e]['state'] for e in range(self.num_envs)]
        self.buf_avail_actions = [info[e]['avail_actions'] for e in range(self.num_envs)]
        return list(obs), list(rewards), list(terminated), list(truncated), list(info)

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
