StarCraft2 Multi-Agent Challenge
==================================================

.. image:: ../../../figures/smac/smac.png

.. raw:: html

    <br><hr>


sc2_env.py
-------------------------------------------------

.. py:class::
    xuance.environment.starcraft2.sc2_env.StarCraft2_Env(map_name)

    This class defines an environment for the StarCraft II environment.

    :param map_name: the name of the StarCraft II map.
    :type map_name: str

.. py:function::
    xuance.environment.starcraft2.sc2_env.StarCraft2_Env.close()

    Close the environment.

.. py:function::
    xuance.environment.starcraft2.sc2_env.StarCraft2_Env.render(mode)

    Sends a render command to each subprocess with the specified rendering mode.

    :param mode:  determine the rendering mode for the visualization
    :type mode: str
    :return: the rendered images from subprocesses.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.starcraft2.sc2_env.StarCraft2_Env.reset()

    Reset the environments.

    :return: the reset local observations, global states, and information.
    :rtype: tuple

.. py:function::
    xuance.environment.starcraft2.sc2_env.StarCraft2_Env.step(actions)

    Take an action as input, perform a step in the underlying environment.

    :param actions: the executable actions for the environment.
    :type actions: np.ndarray
    :return: the next observation, modified reward, episode termination status, truncation information, and additional details for monitoring and analysis.
    :rtype: tuple

.. py:function::
    xuance.environment.starcraft2.sc2_env.StarCraft2_Env.get_avail_actions()

    Returns an array indicating that if all actions are available for each agent in each environment.

    :return: an array indicating that if all actions are available for each agent in each environment.
    :rtype: np.ndarray

.. raw:: html

    <br><hr>

sc2_vec_env.py
-------------------------------------------------

.. py:function::
    xuance.environment.starcraft2.sc2_vec_env.worker(remote, parent_remote, env_fn_wrappers)

    A worker function that is designed to run in a separate process,
    communicating with its parent process through inter-process communication (IPC).

    :param remote: a connection to the child process.
    :type remote: int
    :param parent_remote: a connection to the parent process.
    :type parent_remote: int
    :param env_fn_wrappers: a set of environment function wrappers.

.. py:class::
   xuance.environment.starcraft2.sc2_vec_env.SubprocVecEnv_StarCraft2(env_fns, context='spawn')

   This class defines a vectorized environment for the StarCraft II environments.

   :param env_fns: environment function.
   :param context: the method used for creating and managing processes in a multiprocessing environment.

.. py:function::
    xuance.environment.starcraft2.sc2_vec_env.SubprocVecEnv_StarCraft2.reset()

    Reset the vectorized environments.

    :return: the reset observations, global states, and the information.
    :rtype: tuple

.. py:function::
    xuance.environment.starcraft2.sc2_vec_env.SubprocVecEnv_StarCraft2.step_async(actions)

    Sends asynchronous step commands to each subprocess with the specified actions.

    :param actions: the executable actions for n parallel environments.
    :type actions: np.ndarray

.. py:function::
    xuance.environment.starcraft2.sc2_vec_env.SubprocVecEnv_StarCraft2.step_wait()

    Waits for the completion of asynchronous step operations and updates internal buffers with the received results.

    :return: the observations, states, rewards, terminal flags, truncation flags, and information.
    :rtype: tuple

.. py:function::
    xuance.environment.starcraft2.sc2_vec_env.SubprocVecEnv_StarCraft2.close_extras()

    Closes the communication with subprocesses and joins the subprocesses.

.. py:function::
    xuance.environment.starcraft2.sc2_vec_env.SubprocVecEnv_StarCraft2.render(mode)

    Sends a render command to each subprocess with the specified rendering mode.

    :param mode: determine the rendering mode for the visualization.
    :type mode: str
    :return: the rendered images from subprocesses.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.starcraft2.sc2_vec_env.SubprocVecEnv_StarCraft2.get_avail_actions()

    Returns an array indicating that if all actions are available for each agent in each environment.

    :return: an array indicating that if all actions are available for each agent in each environment.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.starcraft2.sc2_vec_env.SubprocVecEnv_StarCraft2._assert_not_closed()

    Raises an exception if an operation is attempted on the environment after it has been closed.

.. py:function::
    xuance.environment.starcraft2.sc2_vec_env.SubprocVecEnv_StarCraft2.__del__()

    The __del__ method ensures that the environment is properly closed when the object is deleted.

.. py:class::
    xuance.environment.starcraft2.sc2_vec_env.DummyVecEnv_StarCraft2(env_fns)

    Work with multiple environments in parallel.

    :param env_fns: environment function.

.. py:function::
    xuance.environment.starcraft2.sc2_vec_env.DummyVecEnv_StarCraft2.reset()

    Reset the vectorized environments.

    :return: the reset observations, global states, and the information.
    :rtype: tuple

.. py:function::
    xuance.environment.starcraft2.sc2_vec_env.DummyVecEnv_StarCraft2.step_async(actions)

    Sends asynchronous step commands to each subprocess with the specified actions.

    :param actions: the executable actions for n parallel environments.
    :type actions: np.ndarray

.. py:function::
    xuance.environment.starcraft2.sc2_vec_env.DummyVecEnv_StarCraft2.step_wait()

    Waits for the completion of asynchronous step operations and updates internal buffers with the received results.

    :return: the observations, states, rewards, terminal flags, truncation flags, and information.
    :rtype: tuple

.. py:function::
    xuance.environment.starcraft2.sc2_vec_env.DummyVecEnv_StarCraft2.close_extras()

    Closes the communication with subprocesses and joins the subprocesses.

.. py:function::
    xuance.environment.starcraft2.sc2_vec_env.DummyVecEnv_StarCraft2.render(mode)

    Sends a render command to each subprocess with the specified rendering mode.

    :param mode:  determine the rendering mode for the visualization
    :type mode: str
    :return: the rendered images from subprocesses.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.starcraft2.sc2_vec_env.DummyVecEnv_StarCraft2.get_avail_actions()

    Returns an array indicating that if all actions are available for each agent in each environment.

    :return: an array indicating that if all actions are available for each agent in each environment.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.starcraft2.sc2_vec_env.DummyVecEnv_StarCraft2._assert_not_closed()

    Raises an exception if an operation is attempted on the environment after it has been closed.

.. py:function::
    xuance.environment.starcraft2.sc2_vec_env.DummyVecEnv_StarCraft2.__del__()

    The __del__ method ensures that the environment is properly closed when the object is deleted.

.. raw:: html

    <br><hr>

Source Code
---------------------------------------------

.. tabs::

    .. group-tab:: sc2_env.py

        .. code-block:: python

            import copy

            from smac.env import StarCraft2Env
            import numpy as np


            class StarCraft2_Env:
                def __init__(self, map_name):
                    self.env = StarCraft2Env(map_name=map_name)
                    self.env_info = self.env.get_env_info()

                    self.n_agents = self.env_info["n_agents"]
                    self.n_enemies = self.env.n_enemies
                    self.dim_state = self.env_info["state_shape"]
                    self.dim_obs = self.env_info["obs_shape"]
                    self.dim_act = self.n_actions = self.env_info["n_actions"]
                    self.dim_reward = self.n_agents

                    self.observation_space = (self.dim_obs,)
                    self.action_space = (self.dim_act, )
                    self.max_cycles = self.env_info["episode_limit"]
                    self._episode_step = 0
                    self._episode_score = 0
                    self.filled = np.zeros([self.max_cycles, 1], np.bool)
                    self.env.reset()
                    self.buf_info = {
                        'battle_won': 0,
                        'dead_allies': 0,
                        'dead_enemies': 0,
                    }

                def close(self):
                    self.env.close()

                def render(self, mode):
                    return self.env.render(mode)

                def reset(self):
                    obs, state = self.env.reset()
                    self._episode_step = 0
                    self._episode_score = 0.0
                    info = {
                        "episode_step": self._episode_step,
                        "episode_score": self._episode_score,
                    }
                    return obs, state, info

                def step(self, actions):
                    reward, terminated, info = self.env.step(actions)
                    if info == {}:
                        info = self.buf_info
                    obs = self.env.get_obs()
                    state = self.env.get_state()
                    self._episode_step += 1
                    self._episode_score += reward
                    reward_n = np.array([[reward] for _ in range(self.n_agents)])
                    self.buf_info = copy.deepcopy(info)
                    info["episode_step"] = self._episode_step
                    info["episode_score"] = self._episode_score
                    truncated = True if self._episode_step >= self.max_cycles else False
                    return obs, state, reward_n, [terminated], [truncated], info

                def get_avail_actions(self):
                    return self.env.get_avail_actions()


    .. group-tab:: sc2_vec_env.py

        .. code-block:: python

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
                    self.buf_terminal = np.zeros((self.num_envs, 1), dtype=np.bool)
                    self.buf_truncation = np.zeros((self.num_envs, 1), dtype=np.bool)
                    self.buf_done = np.zeros((self.num_envs,), dtype=np.bool)
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
                    self.buf_done = np.zeros((self.num_envs,), dtype=np.bool)
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
                    self.buf_terminal = np.zeros((self.num_envs, 1), dtype=np.bool)
                    self.buf_truncation = np.zeros((self.num_envs, 1), dtype=np.bool)
                    self.buf_done = np.zeros((self.num_envs,), dtype=np.bool)
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
                    self.buf_done = np.zeros((self.num_envs,), dtype=np.bool)
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
