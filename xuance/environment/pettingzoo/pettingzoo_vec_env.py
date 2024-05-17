from xuance.environment.vector_envs.vector_env import VecEnv, AlreadySteppingError, NotSteppingError
from xuance.environment.vector_envs.env_utils import obs_n_space_info
from xuance.environment.gym.gym_vec_env import DummyVecEnv_Gym
from operator import itemgetter
from gymnasium.spaces.box import Box
import numpy as np
from xuance.environment.vector_envs.subproc_vec_env import clear_mpi_env_vars, flatten_list, CloudpickleWrapper
import multiprocessing as mp


def worker(remote, parent_remote, env_fn_wrappers):
    """
    A worker function that is designed to run in a separate process, communicating with
    its parent process through inter-process communication (IPC).
    Parameters:
        remote (int) – a connection to the child process.
        parent_remote (int) – a connection to the parent process.
        env_fn_wrappers – a set of environment function wrappers.
    """

    def step_env(env, action):
        obs_n, reward_n, terminated, truncated, info = env.step(action)
        return obs_n, reward_n, terminated, truncated, info

    parent_remote.close()
    envs = [env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers.x]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send([step_env(env, action) for env, action in zip(envs, data)])
            elif cmd == 'state':
                remote.send([env.state() for env in envs])
            elif cmd == 'get_agent_mask':
                remote.send([env.get_agent_mask() for env in envs])
            elif cmd == 'reset':
                remote.send([env.reset() for env in envs])
            elif cmd == 'render':
                remote.send([env.render() for env in envs])
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_env_info':
                env_info = {
                    "handles": envs[0].handles,
                    "observation_spaces": envs[0].observation_spaces,
                    "state_space": envs[0].state_space,
                    "action_spaces": envs[0].action_spaces,
                    "agent_ids": envs[0].agent_ids,
                    "n_agents": [envs[0].get_num(h) for h in envs[0].handles],
                    "max_cycles": envs[0].max_cycles,
                    "side_names": envs[0].side_names
                }
                remote.send(CloudpickleWrapper(env_info))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        for env in envs:
            env.close()


class SubprocVecEnv_Pettingzoo(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in thread-level and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    Parameters:
        env_fns – environment function.
        context – the method used for creating and managing processes in a multiprocessing environment.
    """

    def __init__(self, env_fns, context="spawn"):
        """
        Arguments:
        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        in_series: number of environments to run in series in a single process
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
        env_info = self.remotes[0].recv().x
        self.handles = env_info["handles"]
        self.state_space = env_info["state_space"]
        self.state_shape = self.state_space.shape
        self.state_dtype = self.state_space.dtype
        obs_n_space = env_info["observation_spaces"]
        self.agent_ids = env_info["agent_ids"]
        self.n_agents = env_info["n_agents"]
        self.side_names = env_info["side_names"]
        VecEnv.__init__(self, num_envs, obs_n_space, env_info["action_spaces"])

        self.keys, self.shapes, self.dtypes = obs_n_space_info(obs_n_space)
        self.agent_keys = [[self.keys[k] for k in ids] for ids in self.agent_ids]
        if isinstance(env_info["action_spaces"][self.agent_keys[0][0]], Box):
            self.act_dim = [env_info["action_spaces"][keys[0]].shape[0] for keys in self.agent_keys]
        else:
            self.act_dim = [env_info["action_spaces"][keys[0]].n for keys in self.agent_keys]
        self.n_agent_all = len(self.keys)
        self.obs_shapes = [self.shapes[self.agent_keys[h.value][0]] for h in self.handles]
        self.obs_dtype = self.dtypes[self.keys[0]]

        # buffer of dict data
        self.buf_obs_dict = [{k: np.zeros(tuple(self.shapes[k]), dtype=self.dtypes[k]) for k in self.keys} for _ in
                             range(self.num_envs)]
        self.buf_rews_dict = [{k: 0.0 for k in self.keys} for _ in range(self.num_envs)]
        self.buf_dones_dict = [{k: False for k in self.keys} for _ in range(self.num_envs)]
        self.buf_trunctions_dict = [{k: False for k in self.keys} for _ in range(self.num_envs)]
        self.buf_infos_dict = [{} for _ in range(self.num_envs)]
        # buffer of numpy data
        self.buf_state = np.zeros((self.num_envs,) + self.state_shape, dtype=self.state_dtype)
        self.buf_agent_mask = [np.ones([self.num_envs, n], dtype=np.bool_) for n in self.n_agents]
        self.buf_obs = [np.zeros((self.num_envs, n) + tuple(self.obs_shapes[h]), dtype=self.obs_dtype) for h, n in
                        enumerate(self.n_agents)]
        self.buf_rews = [np.zeros((self.num_envs, n, 1), dtype=np.float32) for n in self.n_agents]
        self.buf_dones = [np.ones((self.num_envs, n), dtype=np.bool_) for n in self.n_agents]
        self.buf_trunctions = [np.ones((self.num_envs, n), dtype=np.bool_) for n in self.n_agents]

        self.max_episode_length = env_info["max_cycles"]
        self.actions = None

    def empty_dict_buffers(self, i_env):
        """Reset the buffers for dictionary data."""
        self.buf_obs_dict[i_env] = {k: np.zeros(tuple(self.shapes[k]), dtype=self.dtypes[k]) for k in self.keys}
        self.buf_rews_dict[i_env] = {k: 0.0 for k in self.keys}
        self.buf_dones_dict[i_env] = {k: False for k in self.keys}
        self.buf_trunctions_dict[i_env] = {k: False for k in self.keys}
        self.buf_infos_dict[i_env] = {k: {} for k in self.keys}

    def reset(self):
        """Reset the vectorized environments."""
        for remote in self.remotes:
            remote.send(('reset', None))
        result = [remote.recv() for remote in self.remotes]
        result = flatten_list(result)
        obs, info = zip(*result)
        for e in range(self.num_envs):
            self.buf_obs_dict[e].update(obs[e])
            self.buf_infos_dict[e].update(info[e]["infos"])
            for h, agent_keys_h in enumerate(self.agent_keys):
                self.buf_obs[h][e] = itemgetter(*agent_keys_h)(self.buf_obs_dict[e])
        return self.buf_obs.copy(), self.buf_infos_dict.copy()

    def step_async(self, actions):
        """Sends asynchronous step commands to each subprocess with the specified actions."""
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
        self.actions = np.array_split(self.actions, self.n_remotes)
        for remote, action in zip(self.remotes, self.actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        """
        Waits for the completion of asynchronous step operations and updates internal buffers with the received results.
        """
        if not self.waiting:
            raise NotSteppingError

        for e, remote in zip(range(self.num_envs), self.remotes):
            result = remote.recv()
            result = flatten_list(result)
            o, r, d, t, info = result
            remote.send(('state', None))
            self.buf_state[e] = flatten_list(remote.recv())

            if len(o.keys()) < self.n_agent_all:
                self.empty_dict_buffers(e)
            # update the data of alive agents
            self.buf_obs_dict[e].update(o)
            self.buf_rews_dict[e].update(r)
            self.buf_dones_dict[e].update(d)
            self.buf_trunctions_dict[e].update(t)
            self.buf_infos_dict[e].update(info["infos"])

            # resort the data as group-wise
            episode_scores = []
            remote.send(('get_agent_mask', None))
            mask = np.array(flatten_list(remote.recv()))
            for h, agent_keys_h in enumerate(self.agent_keys):
                getter = itemgetter(*agent_keys_h)
                self.buf_agent_mask[h][e] = mask[self.agent_ids[h]]
                self.buf_obs[h][e] = getter(self.buf_obs_dict[e])
                self.buf_rews[h][e, :, 0] = getter(self.buf_rews_dict[e])
                self.buf_dones[h][e] = getter(self.buf_dones_dict[e])
                self.buf_trunctions[h][e] = getter(self.buf_trunctions_dict[e])
                episode_scores.append(getter(info["individual_episode_rewards"]))
            self.buf_infos_dict[e]["individual_episode_rewards"] = episode_scores

            if all(self.buf_dones_dict[e].values()) or all(self.buf_trunctions_dict[e].values()):
                remote.send(('reset', None))
                obs_reset, _ = flatten_list(remote.recv())
                remote.send(('state', None))
                state_reset = flatten_list(remote.recv())
                remote.send(('get_agent_mask', None))
                mask_reset = np.array(flatten_list(remote.recv()))
                obs_reset_handles, mask_reset_handles = [], []
                for h, agent_keys_h in enumerate(self.agent_keys):
                    getter = itemgetter(*agent_keys_h)
                    obs_reset_handles.append(np.array(getter(obs_reset)))
                    mask_reset_handles.append(mask_reset[self.agent_ids[h]])

                self.buf_infos_dict[e]["reset_obs"] = obs_reset_handles
                self.buf_infos_dict[e]["reset_agent_mask"] = mask_reset_handles
                self.buf_infos_dict[e]["reset_state"] = state_reset

        self.waiting = False
        return self.buf_obs.copy(), self.buf_rews.copy(), self.buf_dones.copy(), self.buf_trunctions.copy(), self.buf_infos_dict.copy()

    def close_extras(self):
        """Closes the communication with subprocesses and joins the subprocesses."""
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def render(self, mode=None):
        """Sends a render command to each subprocess with the specified rendering mode."""
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        imgs = flatten_list(imgs)
        return imgs

    def global_state(self):
        """Return the global state of the parallel environments."""
        return self.buf_state

    def agent_mask(self):
        """Return the agent mask."""
        return self.buf_agent_mask

    def available_actions(self):
        """Return an array representing available actions for each agent."""
        act_mask = [np.ones([self.num_envs, n, self.act_dim[h]], dtype=np.bool_) for h, n in enumerate(self.n_agents)]
        return np.array(act_mask)


class DummyVecEnv_Pettingzoo(DummyVecEnv_Gym):
    """
    Work with multiple environments in parallel in process level.
    Parameters:
        env_fns – environment function.
    """

    def __init__(self, env_fns):
        self.waiting = False
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        self.handles = env.handles  # list of handles, e.g., [c_int(0), c_int(1)]. Handle means a group of agents.
        VecEnv.__init__(self, len(env_fns), env.observation_spaces, env.action_spaces)
        self.state_space = env.state_space  # Type: Box
        self.state_shape = self.state_space.shape  # Type: Tuple
        self.state_dtype = self.state_space.dtype  # Type: numpy.dtype
        obs_n_space = env.observation_spaces  # [Box(dim_o), Box(dim_o), ...] ----> dict
        self.agent_ids = env.agent_ids  # list of agent ids, e.g., [[0, 1, 2], [0, 1]]
        self.n_agents = [env.get_num(h) for h in self.handles]  # number of agents for each handle, e.g., [3, 2]
        self.side_names = env.side_names  # the name of each side, e.g., ['red', 'blue']

        self.keys, self.shapes, self.dtypes = obs_n_space_info(obs_n_space)  # self.keys: the keys for all agents.
        self.agent_keys = [[self.keys[k] for k in ids] for ids in self.agent_ids]  # the keys for each handle of agents.
        if isinstance(env.action_spaces[self.agent_keys[0][0]], Box):
            self.act_dim = [env.action_spaces[keys[0]].shape[0] for keys in self.agent_keys]
        else:
            self.act_dim = [env.action_spaces[keys[0]].n for keys in self.agent_keys]
        self.n_agent_all = len(self.keys)  # total number of agents
        self.obs_shapes = [self.shapes[self.agent_keys[h.value][0]] for h in
                           self.handles]  # suppose agents in one handle share a same observation space.
        self.obs_dtype = self.dtypes[self.keys[0]]

        # store data for current time step.
        # buffer of dict data
        self.buf_obs_dict = [{k: np.zeros(tuple(self.shapes[k]), dtype=self.dtypes[k]) for k in self.keys} for _ in
                             range(self.num_envs)]
        self.buf_rews_dict = [{k: 0.0 for k in self.keys} for _ in range(self.num_envs)]
        self.buf_dones_dict = [{k: False for k in self.keys} for _ in range(self.num_envs)]
        self.buf_trunctions_dict = [{k: False for k in self.keys} for _ in range(self.num_envs)]
        self.buf_infos_dict = [{} for _ in range(self.num_envs)]
        # buffer of numpy data
        self.buf_state = np.zeros((self.num_envs,) + self.state_shape, dtype=self.state_dtype)
        self.buf_agent_mask = [np.ones([self.num_envs, n], dtype=np.bool_) for n in self.n_agents]
        self.buf_obs = [np.zeros((self.num_envs, n) + tuple(self.obs_shapes[h]), dtype=self.obs_dtype) for h, n in
                        enumerate(self.n_agents)]
        self.buf_rews = [np.zeros((self.num_envs, n, 1), dtype=np.float32) for n in self.n_agents]
        self.buf_dones = [np.ones((self.num_envs, n), dtype=np.bool_) for n in self.n_agents]
        self.buf_trunctions = [np.ones((self.num_envs, n), dtype=np.bool_) for n in self.n_agents]

        self.max_episode_length = env.max_cycles  # the max length of one episode.
        self.actions = None  # the actions to be executed.

    def empty_dict_buffers(self, i_env):
        """Reset the buffers for dictionary data."""
        self.buf_obs_dict[i_env] = {k: np.zeros(tuple(self.shapes[k]), dtype=self.dtypes[k]) for k in self.keys}
        self.buf_rews_dict[i_env] = {k: 0.0 for k in self.keys}
        self.buf_dones_dict[i_env] = {k: False for k in self.keys}
        self.buf_trunctions_dict[i_env] = {k: False for k in self.keys}
        self.buf_infos_dict[i_env] = {k: {} for k in self.keys}

    def reset(self):
        """Reset the vectorized environments."""
        for e in range(self.num_envs):
            obs, info = self.envs[e].reset()
            self.buf_obs_dict[e].update(obs)
            self.buf_infos_dict[e].update(info["infos"])
            for h, agent_keys_h in enumerate(self.agent_keys):
                self.buf_obs[h][e] = itemgetter(*agent_keys_h)(self.buf_obs_dict[e])
        return self.buf_obs.copy(), self.buf_infos_dict.copy()

    def step_async(self, actions):
        """Sends asynchronous step commands to each subprocess with the specified actions."""
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
        """
        Waits for the completion of asynchronous step operations and updates internal buffers with the received results.
        """
        if not self.waiting:
            raise NotSteppingError

        for e in range(self.num_envs):
            action_n = self.actions[e]
            o, r, d, t, info = self.envs[e].step(action_n)
            self.buf_state[e] = self.envs[e].state()
            if len(o.keys()) < self.n_agent_all:
                self.empty_dict_buffers(e)
            # update the data of alive agents
            self.buf_obs_dict[e].update(o)
            self.buf_rews_dict[e].update(r)
            self.buf_dones_dict[e].update(d)
            self.buf_trunctions_dict[e].update(t)
            self.buf_infos_dict[e].update(info["infos"])

            # resort the data as group-wise
            episode_scores = []
            mask = self.envs[e].get_agent_mask()
            for h, agent_keys_h in enumerate(self.agent_keys):
                getter = itemgetter(*agent_keys_h)
                self.buf_agent_mask[h][e] = mask[self.agent_ids[h]]
                self.buf_obs[h][e] = getter(self.buf_obs_dict[e])
                self.buf_rews[h][e, :, 0] = getter(self.buf_rews_dict[e])
                self.buf_dones[h][e] = getter(self.buf_dones_dict[e])
                self.buf_trunctions[h][e] = getter(self.buf_trunctions_dict[e])
                episode_scores.append(getter(info["individual_episode_rewards"]))
            self.buf_infos_dict[e]["individual_episode_rewards"] = episode_scores

            if all(self.buf_dones_dict[e].values()) or all(self.buf_trunctions_dict[e].values()):
                obs_reset, _ = self.envs[e].reset()
                state_reset = self.envs[e].state()
                mask_reset = self.envs[e].get_agent_mask()
                obs_reset_handles, mask_reset_handles = [], []
                for h, agent_keys_h in enumerate(self.agent_keys):
                    getter = itemgetter(*agent_keys_h)
                    obs_reset_handles.append(np.array(getter(obs_reset)))
                    mask_reset_handles.append(mask_reset[self.agent_ids[h]])

                self.buf_infos_dict[e]["reset_obs"] = obs_reset_handles
                self.buf_infos_dict[e]["reset_agent_mask"] = mask_reset_handles
                self.buf_infos_dict[e]["reset_state"] = state_reset

        self.waiting = False
        return self.buf_obs.copy(), self.buf_rews.copy(), self.buf_dones.copy(), self.buf_trunctions.copy(), self.buf_infos_dict.copy()

    def render(self, mode=None):
        """Sends a render command to each subprocess with the specified rendering mode."""
        return [env.render() for env in self.envs]

    def global_state(self):
        """Return the global state of the parallel environments."""
        return self.buf_state

    def agent_mask(self):
        """Return the agent mask."""
        return self.buf_agent_mask

    def available_actions(self):
        """Return an array representing available actions for each agent."""
        act_mask = [np.ones([self.num_envs, n, self.act_dim[h]], dtype=np.bool_) for h, n in enumerate(self.n_agents)]
        return np.array(act_mask)

    def close_extras(self):
        """Closes the communication with subprocesses and joins the subprocesses."""
        self.closed = True
        for env in self.envs:
            try: env.close()
            except: pass
