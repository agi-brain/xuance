from xuance.environment.vector_envs.vector_env import VecEnv, AlreadySteppingError, NotSteppingError
from xuance.environment.vector_envs.env_utils import obs_n_space_info
from xuance.environment.gym.gym_vec_env import DummyVecEnv_Gym, SubprocVecEnv_Gym
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
            elif cmd == 'reset':
                remote.send([env.reset() for env in envs])
            elif cmd == 'render':
                remote.send([env.render(data) for env in envs])
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_env_info':
                remote.send(envs[0].env_info)
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        for env in envs:
            env.close()


class SubprocVecEnv_Drones_MAS(SubprocVecEnv_Gym):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """

    def __init__(self, env_fns, context='spawn', in_series=1):
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
        self.dim_obs = env_info["obs_shape"][-1]
        self.dim_act = self.n_actions = env_info["n_actions"]
        self.dim_state = env_info["state_shape"]
        observation_space, action_space = (self.dim_obs,), (self.dim_act,)
        self.viewer = None
        VecEnv.__init__(self, num_envs, observation_space, action_space)

        self.num_agents = env_info["n_agents"]
        self.obs_shape = env_info["obs_shape"]
        self.act_shape = (self.num_agents, self.dim_act)
        self.rew_shape = (self.num_agents, 1)
        self.dim_reward = self.num_agents
        self.action_space = env_info["act_space"]
        self.state_space = Box(low=-np.inf, high=np.inf, shape=[self.dim_state, ])

        self.buf_obs = np.zeros(combined_shape(self.num_envs, self.obs_shape), dtype=np.float32)
        self.buf_state = np.zeros(combined_shape(self.num_envs, self.dim_state), dtype=np.float32)
        self.buf_agent_mask = np.ones([self.num_envs, self.num_agents], dtype=np.bool_)
        self.buf_terminals = np.zeros((self.num_envs, self.num_agents), dtype=np.bool_)
        self.buf_truncations = np.zeros((self.num_envs, self.num_agents), dtype=np.bool_)
        self.buf_rews = np.zeros((self.num_envs,) + self.rew_shape, dtype=np.float32)
        self.buf_info = [{} for _ in range(self.num_envs)]

        self.max_episode_length = env_info["episode_limit"]
        self.actions = None

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        result = [remote.recv() for remote in self.remotes]
        result = flatten_list(result)
        obs, infos = zip(*result)
        self.buf_obs, self.buf_info = np.array(obs), list(infos)
        self.buf_done = np.zeros((self.num_envs,), dtype=np.bool_)
        return self.buf_obs.copy(), self.buf_info.copy()

    def step_async(self, actions):
        self._assert_not_closed()
        actions = np.array_split(actions, self.n_remotes)
        for env_done, remote, action in zip(self.buf_done, self.remotes, actions):
            if not env_done:
                remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        if not self.waiting:
            raise NotSteppingError
        for idx_env, env_done, remote in zip(range(self.num_envs), self.buf_done, self.remotes):
            result = remote.recv()
            result = flatten_list(result)
            obs, rew, done, truncated, infos = result
            self.buf_obs[idx_env] = obs
            self.buf_rews[idx_env] = rew
            self.buf_terminals[idx_env] = env_done
            self.buf_truncations[idx_env] = truncated
            self.buf_info[idx_env] = infos
            self.buf_info[idx_env]["individual_episode_rewards"] = infos["episode_score"]
            if done.all() or truncated.all():
                remote.send(('reset', None))
                result = remote.recv()
                obs_reset, _ = zip(*result)
                self.buf_info[idx_env]["reset_obs"] = obs_reset
                remote.send(('get_agent_mask', None))
                result = remote.recv()
                self.buf_info[idx_env]["reset_agent_mask"] = zip(*result)
                remote.send(('state', None))
                result = remote.recv()
                self.buf_info[idx_env]["reset_state"] = zip(*result)
        self.waiting = False
        return self.buf_obs.copy(), self.buf_rews.copy(), self.buf_terminals.copy(), self.buf_truncations.copy(), self.buf_info.copy()

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

    def global_state(self):
        return self.buf_state

    def agent_mask(self):
        return self.buf_agent_mask

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def __del__(self):
        if not self.closed:
            self.close()


class DummyVecEnv_Drones_MAS(DummyVecEnv_Gym):
    def __init__(self, env_fns):
        self.waiting = False
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        env_info = env.env_info
        self.dim_obs = env_info["obs_shape"][-1]
        self.dim_act = self.n_actions = env_info["n_actions"]
        self.dim_state = env_info["state_shape"]
        observation_space, action_space = (self.dim_obs,), (self.dim_act,)
        self.viewer = None
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

        self.num_agents = env_info["n_agents"]
        self.obs_shape = env_info["obs_shape"]
        self.act_shape = (self.num_agents, self.dim_act)
        self.rew_shape = (self.num_agents, 1)
        self.dim_reward = self.num_agents
        self.action_space = env_info["act_space"]
        self.state_space = Box(low=-np.inf, high=np.inf, shape=[self.dim_state, ])

        self.buf_obs = np.zeros(combined_shape(self.num_envs, self.obs_shape), dtype=np.float32)
        self.buf_state = np.zeros(combined_shape(self.num_envs, self.dim_state), dtype=np.float32)
        self.buf_agent_mask = np.ones([self.num_envs, self.num_agents], dtype=np.bool_)
        self.buf_terminals = np.zeros((self.num_envs, self.num_agents), dtype=np.bool_)
        self.buf_truncations = np.zeros((self.num_envs, self.num_agents), dtype=np.bool_)
        self.buf_rews = np.zeros((self.num_envs,) + self.rew_shape, dtype=np.float32)
        self.buf_info = [{} for _ in range(self.num_envs)]

        self.max_episode_length = env_info["episode_limit"]
        self.actions = None

    def reset(self):
        for i_env, env in enumerate(self.envs):
            obs, infos = env.reset()
            self.buf_obs[i_env], self.buf_info[i_env] = np.array(obs), list(infos)
        self.buf_done = np.zeros((self.num_envs,), dtype=np.bool_)
        return self.buf_obs.copy(), self.buf_info.copy()

    def step_async(self, actions):
        self.actions = actions
        self.waiting = True

    def step_wait(self):
        if not self.waiting:
            raise NotSteppingError
        for e in range(self.num_envs):
            action = self.actions[e]
            obs, rew, done, truncated, infos = self.envs[e].step(action)
            self.buf_obs[e] = obs
            self.buf_rews[e] = rew
            self.buf_terminals[e] = done
            self.buf_truncations[e] = truncated
            self.buf_info[e] = infos
            self.buf_info[e]["individual_episode_rewards"] = infos["episode_score"]
            if all(done) or all(truncated):
                obs_reset, _ = self.envs[e].reset()
                self.buf_info[e]["reset_obs"] = obs_reset
                self.buf_info[e]["reset_agent_mask"] = self.envs[e].get_agent_mask()
                self.buf_info[e]["reset_state"] = self.envs[e].state()
        self.waiting = False
        return self.buf_obs.copy(), self.buf_rews.copy(), self.buf_terminals.copy(), self.buf_truncations.copy(), self.buf_info.copy()

    def render(self, mode):
        imgs = [env.render(mode) for env in self.envs]
        return imgs

    def global_state(self):
        for e in range(self.num_envs):
            self.buf_state[e] = self.envs[e].state()
        return self.buf_state

    def agent_mask(self):
        for e in range(self.num_envs):
            self.buf_agent_mask[e] = self.envs[e].get_agent_mask()
        return self.buf_agent_mask

