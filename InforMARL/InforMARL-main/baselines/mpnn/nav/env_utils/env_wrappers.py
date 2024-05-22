import numpy as np
from multiprocessing import Process, Pipe
from .vec_env import VecEnv, CloudpickleWrapper

import gym


class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(
            self,
            len(env_fns),
            env.observation_space,
            env.share_observation_space,
            env.action_space,
        )

        obs_spaces = (
            self.observation_space.spaces
            if isinstance(self.observation_space, gym.spaces.Tuple)
            else (self.observation_space,)
        )
        self.buf_obs = [
            np.zeros((self.num_envs,) + tuple(s[0].shape), s[0].dtype)
            for s in obs_spaces
        ]
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        for i in range(self.num_envs):
            (
                obs_tuple,
                self.buf_rews[i],
                self.buf_dones[i],
                self.buf_infos[i],
            ) = self.envs[i].step(self.actions[i])
            if self.buf_dones[i]:
                obs_tuple = self.envs[i].reset()
            if isinstance(obs_tuple, (tuple, list)):
                for t, x in enumerate(obs_tuple):
                    self.buf_obs[t][i] = x
            else:
                self.buf_obs[0][i] = obs_tuple
        return (
            self._obs_from_buf(),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            self.buf_infos.copy(),
        )

    def reset(self):
        for i in range(self.num_envs):
            obs_tuple = self.envs[i].reset()
            if isinstance(obs_tuple, (tuple, list)):
                for t, x in enumerate(obs_tuple):
                    self.buf_obs[t][i] = x
            else:
                self.buf_obs[0][i] = obs_tuple
        return self._obs_from_buf()

    def close(self):
        return

    def _obs_from_buf(self):
        if len(self.buf_obs) == 1:
            return np.copy(self.buf_obs[0])
        else:
            return tuple(np.copy(x) for x in self.buf_obs)


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            ob, reward, done, info = env.step(data)
            if "bool" in done.__class__.__name__:
                if done:
                    ob = env.reset()
            else:
                if np.all(done):
                    ob = env.reset()
            # NOTE: Modified here compared to original MPNN implementation
            # if np.any(done):
            #     ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == "reset":
            ob = env.reset()
            remote.send(ob)
        elif cmd == "reset_task":
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == "close":
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send(
                (env.observation_space, env.share_observation_space, env.action_space)
            )
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(
                target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn))
            )
            for (work_remote, remote, env_fn) in zip(
                self.work_remotes, self.remotes, env_fns
            )
        ]
        for p in self.ps:
            p.daemon = (
                True  # if the main process crashes, we should not cause things to hang
            )
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, share_observation_space, action_space = self.remotes[
            0
        ].recv()
        VecEnv.__init__(
            self, len(env_fns), observation_space, share_observation_space, action_space
        )

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(("reset_task", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True
