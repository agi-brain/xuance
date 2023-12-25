PettingZoo
==============================================

MPE
-------------------------------------------

.. image:: ../../../figures/mpe/mpe_simple_spread.gif
    :height: 150px
.. image:: ../../../figures/mpe/mpe_simple_push.gif
    :height: 150px
.. image:: ../../../figures/mpe/mpe_simple_reference.gif
    :height: 150px
.. image:: ../../../figures/mpe/mpe_simple_world_comm.gif
    :height: 150px

.. raw:: html

    <br><hr>

pettingzoo_env.py
-----------------------------------------------

.. py:class::
    xuance.environment.pettingzoo.pettingzoo_env.PettingZoo_Env(env_name, env_id, seed, kwargs)

    A wrapper for PettingZoo environments,
    provide a standardized interface for interacting with the environments in the context of multi-agent reinforcement learning

    :param env_name: the name of the PettingZoo environment.
    :type env_name: str
    :param env_id: environment id.
    :type env_id: str
    :param seed: use to control randomness within the environment.
    :type seed: int
    :param kwargs: a variable-length keyword argument.
    :type kwargs: dict

.. py:function::
    xuance.environment.pettingzoo.pettingzoo_env.PettingZoo_Env.close()

    Close the environment.

.. py:function::
    xuance.environment.pettingzoo.pettingzoo_env.PettingZoo_Env.render()

    Get the rendered images of the environment.

.. py:function::
    xuance.environment.pettingzoo.pettingzoo_env.PettingZoo_Env.reset(seed=None, options=None)

    Reset the environment to its initial state.

    :param seed: specifies the seed for the environment's random number generator.
    :type seed: int
    :param options: pass additional options  to the reset process.
    :type options: dict
    :return: the reset local observations and information.
    :rtype: tuple

.. py:function::
    xuance.environment.pettingzoo.pettingzoo_env.PettingZoo_Env.step(actions)

    Take an action as input, perform a step in the underlying pettingzoo environment.

    :param actions: the executable actions for the environment.
    :type actions: np.ndarray
    :return: the next step data, including local observations, global state, rewards, terminated variables, truncated variables, and the other information.
    :rtype: tuple

.. py:function::
    xuance.environment.pettingzoo.pettingzoo_env.PettingZoo_Env.state()

    Retrieve the current state of the environment.

    :return: the current state of the environment.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.pettingzoo.pettingzoo_env.PettingZoo_Env.get_num(handle)

    Retrieve the number of agents associated with a specific handle in the environment.

    :param handle: an identifier associated with a group of agents in the environment.
    :type handle: int
    :return: the calculated or retrieved number of agents associated with the specified handle.
    :rtype: int

.. py:function::
    xuance.environment.pettingzoo.pettingzoo_env.PettingZoo_Env.get_ids(handle)

    Retrieve the agent IDs associated with that handle.

    :param handle: an identifier associated with a group of agents in the environment.
    :type handle: int
    :return: a list of integers representing the agent ids.
    :rtype: int

.. py:function::
    xuance.environment.pettingzoo.pettingzoo_env.PettingZoo_Env.get_agent_mask()

    Create a boolean mask indicating which agents are currently alive.

    :return: the status of agents.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.pettingzoo.pettingzoo_env.PettingZoo_Env.get_handles()

    Retrieve the handles associated with the agents.

    :return: the handles associated with the agents.

.. raw:: html

    <br><hr>

pettingzoo_vec_env.py
-----------------------------------------------

.. py:function::
    xuance.environment.pettingzoo.pettingzoo_vec_env.worker(remote, parent_remote, env_fn_wrappers)

    A worker function that is designed to run in a separate process,
    communicating with its parent process through inter-process communication (IPC).

    :param remote: a connection to the child process.
    :type remote: int
    :param parent_remote: a connection to the parent process.
    :type parent_remote: int
    :param env_fn_wrappers: a set of environment function wrappers.

.. py:class::
   xuance.environment.pettingzoo.pettingzoo_vec_env.SubprocVecEnv_Pettingzoo(env_fns, context='spawn')

   This class defines a vectorized environment for the Pettingzoo environments.

   :param env_fns: environment function.
   :param context: the method used for creating and managing processes in a multiprocessing environment.

.. py:function::
    xuance.environment.pettingzoo.pettingzoo_vec_env.SubprocVecEnv_Pettingzoo.empty_dict_buffers(i_env)

    Reset the buffers for dictionary data.

    :param i_env: the index of a environment.
    :type i_env: int

.. py:function::
    xuance.environment.pettingzoo.pettingzoo_vec_env.SubprocVecEnv_Pettingzoo.reset()

    Reset the vectorized environments.

    :return: the reset observations, global states, and the information.
    :rtype: tuple

.. py:function::
    xuance.environment.pettingzoo.pettingzoo_vec_env.SubprocVecEnv_Pettingzoo.step_async(actions)

    Sends asynchronous step commands to each subprocess with the specified actions.

    :param actions: the executable actions for n parallel environments.
    :type actions: np.ndarray

.. py:function::
    xuance.environment.pettingzoo.pettingzoo_vec_env.SubprocVecEnv_Pettingzoo.step_wait()

    Waits for the completion of asynchronous step operations and updates internal buffers with the received results.

    :return: the observations, states, rewards, terminal flags, truncation flags, and information.
    :rtype: tuple

.. py:function::
    xuance.environment.pettingzoo.pettingzoo_vec_env.SubprocVecEnv_Pettingzoo.close_extras()

    Closes the communication with subprocesses and joins the subprocesses.

.. py:function::
    xuance.environment.pettingzoo.pettingzoo_vec_env.SubprocVecEnv_Pettingzoo.render(mode)

    Sends a render command to each subprocess with the specified rendering mode.

    :param mode:  determine the rendering mode for the visualization.
    :type mode: str
    :return: the rendered images from subprocesses.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.pettingzoo.pettingzoo_vec_env.SubprocVecEnv_Pettingzoo.global_state()

    Return the global state of the parallel environments.

    :return: the global state of the parallel environments.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.pettingzoo.pettingzoo_vec_env.SubprocVecEnv_Pettingzoo.agent_mask()

    Return the agent mask.

    :return: the agent mask.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.pettingzoo.pettingzoo_vec_env.SubprocVecEnv_Pettingzoo.available_actions()

    Return an array representing available actions for each agent.

    :return: an array representing available actions for each agent.
    :rtype: np.ndarray

.. py:class::
    xuance.environment.pettingzoo.pettingzoo_vec_env.DummyVecEnv_Pettingzoo(env_fns)

    Work with multiple environments in parallel.

    :param env_fns: environment function.

.. py:function::
    xuance.environment.pettingzoo.pettingzoo_vec_env.DummyVecEnv_Pettingzoo.empty_dict_buffers(i_env)

    Reset the buffers for dictionary data.

    :param i_env: the index of a environment.
    :type i_env: int

.. py:function::
    xuance.environment.pettingzoo.pettingzoo_vec_env.DummyVecEnv_Pettingzoo.reset()

    Reset the vectorized environments.

    :return: the reset observations, global states, and the information.
    :rtype: tuple

.. py:function::
    xuance.environment.pettingzoo.pettingzoo_vec_env.DummyVecEnv_Pettingzoo.step_async(actions)

    Sends asynchronous step commands to each subprocess with the specified actions.

    :param actions: the executable actions for n parallel environments.
    :type actions: np.ndarray

.. py:function::
    xuance.environment.pettingzoo.pettingzoo_vec_env.DummyVecEnv_Pettingzoo.step_wait()

    Waits for the completion of asynchronous step operations and updates internal buffers with the received results.

    :return: the observations, states, rewards, terminal flags, truncation flags, and information.
    :rtype: tuple

.. py:function::
    xuance.environment.pettingzoo.pettingzoo_vec_env.DummyVecEnv_Pettingzoo.render(mode)

    Sends a render command to each subprocess with the specified rendering mode.

    :param mode: determine the rendering mode for the visualization.
    :type mode: str
    :return: the rendered images from subprocesses.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.pettingzoo.pettingzoo_vec_env.DummyVecEnv_Pettingzoo.global_state()

    Return the global state of the parallel environments.

    :return: the global state of the parallel environments.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.pettingzoo.pettingzoo_vec_env.DummyVecEnv_Pettingzoo.agent_mask()

    Return the agent mask.

    :return: the agent mask.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.pettingzoo.pettingzoo_vec_env.DummyVecEnv_Pettingzoo.available_actions()

    Return an array representing available actions for each agent.

    :return: an array representing available actions for each agent.
    :rtype: np.ndarray

.. raw:: html

    <br><hr>

Source Code
---------------------------------------------

.. tabs::

    .. group-tab:: pettingzoo_env.py

        .. code-block:: python

            from pettingzoo.utils.env import ParallelEnv
            import numpy as np
            import ctypes
            import importlib
            from xuance.environment.pettingzoo import AGENT_NAME_DICT


            class PettingZoo_Env(ParallelEnv):
                def __init__(self, env_name: str, env_id: str, seed: int, **kwargs):
                    super(PettingZoo_Env, self).__init__()
                    scenario = importlib.import_module('pettingzoo.' + env_name + '.' + env_id)
                    self.continuous_actions = kwargs["continuous"]
                    self.env = scenario.parallel_env(continuous_actions=self.continuous_actions,
                                                     render_mode=kwargs["render_mode"])
                    self.scenario_name = env_name + "." + env_id
                    self.n_handles = len(AGENT_NAME_DICT[self.scenario_name])
                    self.side_names = AGENT_NAME_DICT[self.scenario_name]
                    self.env.reset()
                    try:
                        self.state_space = self.env.state_space
                    except:
                        self.state_space = None

                    self.action_spaces = {k: self.env.action_space(k) for k in self.env.agents}
                    self.observation_spaces = {k: self.env.observation_space(k) for k in self.env.agents}
                    self.agents = self.env.agents
                    self.n_agents_all = len(self.agents)

                    self.handles = self.get_handles()

                    self.agent_ids = [self.get_ids(h) for h in self.handles]
                    self.n_agents = [self.get_num(h) for h in self.handles]

                    # self.reward_range = env.reward_range
                    self.metadata = self.env.metadata
                    # self._warn_double_wrap()
                    # assert self.spec.id in ENVIRONMENTS

                    self.max_cycles = self.env.aec_env.env.env.max_cycles
                    self.individual_episode_reward = {k: 0.0 for k in self.agents}

                def close(self):
                    self.env.close()

                def render(self):
                    return self.env.render()

                def reset(self, seed=None, options=None):
                    observations, infos = self.env.reset()
                    for agent_key in self.agents:
                        self.individual_episode_reward[agent_key] = 0.0
                    reset_info = {"infos": infos,
                                  "individual_episode_rewards": self.individual_episode_reward}
                    return observations, reset_info

                def step(self, actions):
                    if self.continuous_actions:
                        for k, v in actions.items():
                            actions[k] = np.clip(v, self.action_spaces[k].low, self.action_spaces[k].high)
                    observations, rewards, terminations, truncations, infos = self.env.step(actions)
                    for k, v in rewards.items():
                        self.individual_episode_reward[k] += v
                    step_info = {"infos": infos,
                                 "individual_episode_rewards": self.individual_episode_reward}
                    return observations, rewards, terminations, truncations, step_info

                def state(self):
                    try:
                        return np.array(self.env.state())
                    except:
                        return None

                def get_num(self, handle):
                    try:
                        n = self.env.env.get_num(handle)
                    except:
                        n = len(self.get_ids(handle))
                    return n

                def get_ids(self, handle):
                    try:
                        ids = self.env.env.get_agent_id(handle)
                    except:
                        agent_name = AGENT_NAME_DICT[self.scenario_name][handle.value]
                        ids_handle = []
                        for id, agent_key in enumerate(self.agents):
                            if agent_name in agent_key:
                                ids_handle.append(id)
                        ids = ids_handle
                    return ids

                def get_agent_mask(self):
                    if self.handles is None:
                        return np.ones(self.n_agents_all, dtype=np.bool)  # all alive
                    else:
                        mask = np.zeros(self.n_agents_all, dtype=np.bool)  # all dead
                        for handle in self.handles:
                            try:
                                alive_ids = self.get_ids(handle)
                                mask[alive_ids] = True  # get alive agents
                            except AttributeError("Cannot get the ids for alive agents!"):
                                return
                    return mask

                def get_handles(self):
                    if hasattr(self.env, 'handles'):
                        return self.env.handles
                    else:
                        try:
                            return self.env.env.get_handles()
                        except:
                            handles = [ctypes.c_int(h) for h in range(self.n_handles)]
                            return handles


    .. group-tab:: pettingzoo_vec_env.py

        .. code-block:: python

            from abc import ABC

            from xuance.environment.vector_envs.vector_env import VecEnv, AlreadySteppingError, NotSteppingError
            from xuance.environment.vector_envs.env_utils import obs_n_space_info
            from xuance.environment.gym.gym_vec_env import DummyVecEnv_Gym
            from operator import itemgetter
            from gymnasium.spaces.box import Box
            import numpy as np
            from xuance.environment.vector_envs.subproc_vec_env import clear_mpi_env_vars, flatten_list, CloudpickleWrapper
            import multiprocessing as mp


            def worker(remote, parent_remote, env_fn_wrappers):
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
                VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
                Recommended to use when num_envs > 1 and step() can be a bottleneck.
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
                    self.buf_agent_mask = [np.ones([self.num_envs, n], dtype=np.bool) for n in self.n_agents]
                    self.buf_obs = [np.zeros((self.num_envs, n) + tuple(self.obs_shapes[h]), dtype=self.obs_dtype) for h, n in
                                    enumerate(self.n_agents)]
                    self.buf_rews = [np.zeros((self.num_envs, n, 1), dtype=np.float32) for n in self.n_agents]
                    self.buf_dones = [np.ones((self.num_envs, n), dtype=np.bool) for n in self.n_agents]
                    self.buf_trunctions = [np.ones((self.num_envs, n), dtype=np.bool) for n in self.n_agents]

                    self.max_episode_length = env_info["max_cycles"]
                    self.actions = None

                def empty_dict_buffers(self, i_env):
                    # buffer of dict data
                    self.buf_obs_dict[i_env] = {k: np.zeros(tuple(self.shapes[k]), dtype=self.dtypes[k]) for k in self.keys}
                    self.buf_rews_dict[i_env] = {k: 0.0 for k in self.keys}
                    self.buf_dones_dict[i_env] = {k: False for k in self.keys}
                    self.buf_trunctions_dict[i_env] = {k: False for k in self.keys}
                    self.buf_infos_dict[i_env] = {k: {} for k in self.keys}

                def reset(self):
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
                    self.closed = True
                    if self.waiting:
                        for remote in self.remotes:
                            remote.recv()
                    for remote in self.remotes:
                        remote.send(('close', None))
                    for p in self.ps:
                        p.join()

                def render(self, mode=None):
                    for pipe in self.remotes:
                        pipe.send(('render', None))
                    imgs = [pipe.recv() for pipe in self.remotes]
                    imgs = flatten_list(imgs)
                    return imgs

                def global_state(self):
                    return self.buf_state

                def agent_mask(self):
                    return self.buf_agent_mask

                def available_actions(self):
                    act_mask = [np.ones([self.num_envs, n, self.act_dim[h]], dtype=np.bool) for h, n in enumerate(self.n_agents)]
                    return np.array(act_mask)


            class DummyVecEnv_Pettingzoo(DummyVecEnv_Gym):
                def __init__(self, env_fns):
                    self.waiting = False
                    self.envs = [fn() for fn in env_fns]
                    env = self.envs[0]
                    self.handles = env.handles
                    VecEnv.__init__(self, len(env_fns), env.observation_spaces, env.action_spaces)
                    self.state_space = env.state_space
                    self.state_shape = self.state_space.shape
                    self.state_dtype = self.state_space.dtype
                    obs_n_space = env.observation_spaces  # [Box(dim_o), Box(dim_o), ...] ----> dict
                    self.agent_ids = env.agent_ids
                    self.n_agents = [env.get_num(h) for h in self.handles]
                    self.side_names = env.side_names

                    self.keys, self.shapes, self.dtypes = obs_n_space_info(obs_n_space)
                    self.agent_keys = [[self.keys[k] for k in ids] for ids in self.agent_ids]
                    if isinstance(env.action_spaces[self.agent_keys[0][0]], Box):
                        self.act_dim = [env.action_spaces[keys[0]].shape[0] for keys in self.agent_keys]
                    else:
                        self.act_dim = [env.action_spaces[keys[0]].n for keys in self.agent_keys]
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
                    self.buf_state = np.zeros((self.num_envs, ) + self.state_shape, dtype=self.state_dtype)
                    self.buf_agent_mask = [np.ones([self.num_envs, n], dtype=np.bool) for n in self.n_agents]
                    self.buf_obs = [np.zeros((self.num_envs, n) + tuple(self.obs_shapes[h]), dtype=self.obs_dtype) for h, n in
                                    enumerate(self.n_agents)]
                    self.buf_rews = [np.zeros((self.num_envs, n, 1), dtype=np.float32) for n in self.n_agents]
                    self.buf_dones = [np.ones((self.num_envs, n), dtype=np.bool) for n in self.n_agents]
                    self.buf_trunctions = [np.ones((self.num_envs, n), dtype=np.bool) for n in self.n_agents]

                    self.max_episode_length = env.max_cycles
                    self.actions = None

                def empty_dict_buffers(self, i_env):
                    # buffer of dict data
                    self.buf_obs_dict[i_env] = {k: np.zeros(tuple(self.shapes[k]), dtype=self.dtypes[k]) for k in self.keys}
                    self.buf_rews_dict[i_env] = {k: 0.0 for k in self.keys}
                    self.buf_dones_dict[i_env] = {k: False for k in self.keys}
                    self.buf_trunctions_dict[i_env] = {k: False for k in self.keys}
                    self.buf_infos_dict[i_env] = {k: {} for k in self.keys}

                def reset(self):
                    for e in range(self.num_envs):
                        obs, info = self.envs[e].reset()
                        self.buf_obs_dict[e].update(obs)
                        self.buf_infos_dict[e].update(info["infos"])
                        for h, agent_keys_h in enumerate(self.agent_keys):
                            self.buf_obs[h][e] = itemgetter(*agent_keys_h)(self.buf_obs_dict[e])
                    return self.buf_obs.copy(), self.buf_infos_dict.copy()

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
                    return [env.render() for env in self.envs]

                def global_state(self):
                    return self.buf_state

                def agent_mask(self):
                    return self.buf_agent_mask

                def available_actions(self):
                    act_mask = [np.ones([self.num_envs, n, self.act_dim[h]], dtype=np.bool) for h, n in enumerate(self.n_agents)]
                    return np.array(act_mask)


