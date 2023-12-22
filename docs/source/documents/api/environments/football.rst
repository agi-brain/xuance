Google Research Football
=================================================

.. image:: ../../../figures/football/gfootball.png


.. raw:: html

    <br><hr>


raw_env.py
-------------------------------------------------

.. py:class:: 
   xuance.environment.football.raw_env.football_raw_env(args)

   This class defines an environment for the Google Research Football environment (gfootball).

   :param args: the arguments that is necessary for creating an instance of the environment.
   :type args: Namespace

.. py:function::
   xuance.environment.football.raw_env.football_raw_env.reset()

   :return: the reset observations and information.
   :rtype: tuple

.. py:function::
    xuance.environment.football.raw_env.football_raw_env.step(action)

    :param action: the executable action for the environment.
    :type action: np.ndarray
    :return: the next step data, including observations, rewards, terminated variables, truncated variables, and the other information.
    :rtype: tuple

.. py:function::
    xuance.environment.football.raw_env.football_raw_env.get_frame()

    :return: the rendered image for current step.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.football.raw_env.football_raw_env.state()

    :return: the global state.
    :rtype: np.ndarray

.. raw:: html

    <br><hr>


gfootball_env.py
-------------------------------------------------

.. py:class:: 
   xuance.environment.football.gfootball_env.GFootball_Env(args)

   This class defines a wrappered environment for the Google Research Football environment (GFootball_Env).

   :param args: the arguments that is necessary for creating an instance of the environment.
   :type args: Namespace

.. py:function::
    xuance.environment.football.gfootball_env.GFootball_Env.close()

    Close the environment.

.. py:function::
    xuance.environment.football.gfootball_env.GFootball_Env.render()

    Get the rendered images of the environment.

.. py:function::
    xuance.environment.football.gfootball_env.GFootball_Env.reset()

    :return: the reset local observations, global states, and information.
    :rtype: tuple

.. py:function::
    xuance.environment.football.gfootball_env.GFootball_Env.step(actions)

    :param actions: the executable actions for the environment.
    :type actions: np.ndarray
    :return: the next step data, including local observations, global state, rewards, terminated variables, truncated variables, and the other information.
    :rtype: tuple

.. py:function::
    xuance.environment.football.gfootball_env.GFootball_Env.get_more_info(info)

    :param info: the basic information.
    :type info: dict
    :return: the updated dict that contains more additional information.
    :rtype: dict

.. py:function::
    xuance.environment.football.gfootball_env.GFootball_Env.get_state()

    :return: the global state.
    :rtype: np.ndarray

.. raw:: html

    <br><hr>


gfootball_vec_env.py
-------------------------------------------------

.. py:function::
    xuance.environment.football.gfootball_vec_env.worker(remote, parent_remote, env_fn_wrappers)

    A worker function that is designed to run in a separate process, 
    communicating with its parent process through inter-process communication (IPC).

    :param remote: a connection to the child process.
    :type remote: int
    :param parent_remote: a connection to the parent process.
    :type parent_remote: int
    :param env_fn_wrappers: a set of environment function wrappers.


.. py:class:: 
   xuance.environment.football.gfootball_vec_env.SubprocVecEnv_GFootball(env_fns, context='spawn')

   This class defines a vectorized environment for the Google Research Football environments (GFootball_Env).

   :param env_fns: environment function.
   :param context: the method used for creating and managing processes in a multiprocessing environment.

.. py:function::
    xuance.environment.football.gfootball_vec_env.SubprocVecEnv_GFootball.reset()

    Reset the vectorized environments.

    :return: the reset observations, global states, and the information.
    :rtype: tuple

.. py:function::
    xuance.environment.football.gfootball_vec_env.SubprocVecEnv_GFootball.step_async(actions)

    Sends asynchronous 'step' commands to each subprocess with the specified actions.

    :param actions: the executable actions for n parallel environments.
    :type actions: np.ndarray

.. py:function::
    xuance.environment.football.gfootball_vec_env.SubprocVecEnv_GFootball.step_wait()

    Waits for the completion of asynchronous step operations and updates internal buffers with the received results.

    :return: the observations, states, rewards, terminal flags, truncation flags, and information.
    :rtype: tuple

.. py:function::
    xuance.environment.football.gfootball_vec_env.SubprocVecEnv_GFootball.close_extras()

    Closes the communication with subprocesses and joins the subprocesses.

.. py:function::
    xuance.environment.football.gfootball_vec_env.SubprocVecEnv_GFootball.render(mode)

    Sends a 'render' command to each subprocess with the specified rendering mode.

    :param mode:  determine the rendering mode for the visualization
    :type mode: str
    :return: the rendered images from subprocesses.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.football.gfootball_vec_env.SubprocVecEnv_GFootball.get_avail_actions()

    Returns an array indicating that if all actions are available for each agent in each environment.

    :return: an array indicating that if all actions are available for each agent in each environment.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.football.gfootball_vec_env.SubprocVecEnv_GFootball._assert_not_closed()

    Raises an exception if an operation is attempted on the environment after it has been closed.

.. py:function::
    xuance.environment.football.gfootball_vec_env.SubprocVecEnv_GFootball.__del__()

    The __del__ method ensures that the environment is properly closed when the object is deleted.

.. py:class:: 
   xuance.environment.football.gfootball_vec_env.DummyVecEnv_GFootball(env_fns)

   This class defines another vectorized environment for the Google Research Football environments (GFootball_Env).

   :param env_fns: environment function.

.. py:function::
    xuance.environment.football.gfootball_vec_env.DummyVecEnv_GFootball.reset()

    Reset the vectorized environments.

    :return: the reset observations, global states, and the information.
    :rtype: tuple

.. py:function::
    xuance.environment.football.gfootball_vec_env.DummyVecEnv_GFootball.step_async(actions)

    Sends asynchronous 'step' commands to each subprocess with the specified actions.

    :param actions: the executable actions for n parallel environments.
    :type actions: np.ndarray

.. py:function::
    xuance.environment.football.gfootball_vec_env.DummyVecEnv_GFootball.step_wait()

    Waits for the completion of asynchronous step operations and updates internal buffers with the received results.

    :return: the observations, states, rewards, terminal flags, truncation flags, and infos.
    :rtype: tuple

.. py:function::
    xuance.environment.football.gfootball_vec_env.DummyVecEnv_GFootball.close_extras()

    Closes the communication with subprocesses and joins the subprocesses.

.. py:function::
    xuance.environment.football.gfootball_vec_env.DummyVecEnv_GFootball.render(mode)

    Sends a 'render' command to each subprocess with the specified rendering mode.
    :param mode: determine the rendering mode for the visualization
    :type mode: str
    :return: the rendered images from subprocesses.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.football.gfootball_vec_env.DummyVecEnv_GFootball.get_avail_actions()

    Returns an array indicating that if all actions are available for each agent in each environment.

    :return: an array indicating that if all actions are available for each agent in each environment.
    :rtype: np.ndarray


.. raw:: html

    <br><hr>

Source Code
---------------------------------------------

.. tabs::
  
    .. group-tab:: raw_env.py
    
        .. code-block:: python

            import gfootball.env as football_env
            from . import GFOOTBALL_ENV_ID
            from gfootball.env.football_env import FootballEnv
            from gfootball.env import config
            from gfootball.env.wrappers import Simple115StateWrapper
            import numpy as np


            class football_raw_env(FootballEnv):
                def __init__(self, args):
                    write_goal_dumps = False
                    dump_frequency = 1
                    extra_players = None
                    other_config_options = {}
                    self.env_id = GFOOTBALL_ENV_ID[args.env_id]
                    if args.test:
                        write_full_episode_dumps = True
                        self.render = True
                        write_video = True
                    else:
                        write_full_episode_dumps = False
                        self.render = False
                        write_video = False
                    self.n_agents = args.num_agent

                    self.env = football_env.create_environment(
                        env_name=self.env_id,
                        stacked=args.use_stacked_frames,
                        representation=args.obs_type,
                        rewards=args.rewards_type,
                        write_goal_dumps=write_goal_dumps,
                        write_full_episode_dumps=write_full_episode_dumps,
                        render=self.render,
                        write_video=write_video,
                        dump_frequency=dump_frequency,
                        log_dir=args.videos_dir,
                        extra_players=extra_players,
                        number_of_left_players_agent_controls=args.num_agent,
                        number_of_right_players_agent_controls=args.num_adversary,
                        channel_dimensions=(args.smm_width, args.smm_height),
                        other_config_options=other_config_options
                    ).unwrapped

                    scenario_config = config.Config({'level': self.env_id}).ScenarioConfig()
                    players = [('agent:left_players=%d,right_players=%d' % (args.num_agent, args.num_adversary))]

                    # Enable MultiAgentToSingleAgent wrapper?
                    if scenario_config.control_all_players:
                        if (args.num_agent in [0, 1]) and (args.num_adversary in [0, 1]):
                            players = [('agent:left_players=%d,right_players=%d' %
                                        (scenario_config.controllable_left_players if args.num_agent else 0,
                                        scenario_config.controllable_right_players if args.num_adversary else 0))]

                    if extra_players is not None:
                        players.extend(extra_players)
                    config_values = {
                        'dump_full_episodes': write_full_episode_dumps,
                        'dump_scores': write_goal_dumps,
                        'players': players,
                        'level': self.env_id,
                        'tracesdir': args.videos_dir,
                        'write_video': write_video,
                    }
                    config_values.update(other_config_options)
                    c = config.Config(config_values)
                    super(football_raw_env, self).__init__(c)

                def reset(self):
                    obs = self.env.reset()
                    return obs, {}

                def step(self, action):
                    obs, reward, terminated, info = self.env.step(action)
                    truncated = False
                    global_reward = np.sum(reward)
                    reward_n = np.array([global_reward] * self.n_agents)
                    return obs, reward_n, terminated, truncated, info

                def get_frame(self):
                    original_obs = self.env._env._observation
                    frame = original_obs["frame"] if self.render else []
                    return frame

                def state(self):
                    def do_flatten(obj):
                        """Run flatten on either python list or numpy array."""
                        if type(obj) == list:
                            return np.array(obj).flatten()
                        elif type(obj) == int:
                            return np.array([obj])
                        else:
                            return obj.flatten()

                    original_obs = self.env._env._observation
                    state = []
                    for k, v in original_obs.items():
                        if k == "ball_owned_team":
                            if v == -1:
                                state.extend([1, 0, 0])
                            elif v == 0:
                                state.extend([0, 1, 0])
                            else:
                                state.extend([0, 0, 1])
                        elif k == "game_mode":
                            game_mode = [0] * 7
                            game_mode[v] = 1
                            state.extend(game_mode)
                        elif k == "frame":
                            pass
                        else:
                            state.extend(do_flatten(v))
                    return state


    .. group-tab:: gfootball_env.py
    
        .. code-block:: python

            from gfootball.env import _apply_output_wrappers
            from .raw_env import football_raw_env
            from gym.spaces import MultiDiscrete, Discrete
            import numpy as np


            class GFootball_Env:
                """The wrapper of original football environment.

                Args:
                    args: the SimpleNamespace variable that contains attributes to create an original env.
                """
                def __init__(self, args):
                    env = football_raw_env(args)
                    self.env = _apply_output_wrappers(env=env,
                                                    rewards=args.rewards_type,
                                                    representation=args.obs_type,
                                                    channel_dimensions=(args.smm_width, args.smm_height),
                                                    apply_single_agent_wrappers=(args.num_agent + args.num_adversary == 1),
                                                    stacked=args.num_adversary)
                    self.n_agents = args.num_agent
                    self.n_adversaries = args.num_adversary
                    self.observation_space = self.env.observation_space
                    self.dim_obs = self.observation_space.shape[-1]
                    self.action_space = self.env.action_space
                    if isinstance(self.action_space, MultiDiscrete):
                        self.dim_act = self.n_actions = self.action_space.nvec[0]
                    elif isinstance(self.action_space, Discrete):
                        self.dim_act = self.n_actions = self.action_space.n
                    else:
                        raise "Unsupported action spaces"
                    self.max_cycles = self.env.unwrapped.observation()[0]['steps_left']
                    self._episode_step = 0
                    self._episode_score = 0.0
                    self.filled = np.zeros([self.max_cycles, 1], np.bool)
                    self.env.reset()
                    state = self.get_state()
                    self.dim_state = state.shape[0]
                    self.dim_reward = self.n_agents

                def close(self):
                    """Close the environment."""
                    self.env.close()

                def render(self):
                    """Get one-step frame."""
                    return self.env.get_frame()

                def reset(self):
                    """Reset the environment."""
                    obs, info = self.env.reset()
                    obs = obs.reshape([self.n_agents, -1])
                    state = self.get_state()
                    self._episode_step = 0
                    self._episode_score = 0.0
                    info = {
                        "episode_step": self._episode_step,
                        "episode_score": self._episode_score,
                    }
                    return obs, state, info

                def step(self, actions):
                    """One-step transition of the environment.

                    Args:
                        actions: the actions for all agents.
                    """
                    obs, reward, terminated, truncated, info = self.env.step(actions)
                    obs = obs.reshape([self.n_agents, -1])
                    state = self.get_state()
                    self._episode_step += 1
                    self._episode_score += reward.mean()
                    info["episode_step"] = self._episode_step
                    info["episode_score"] = self._episode_score
                    truncated = True if self._episode_step >= self.max_cycles else False
                    return obs, state, reward, terminated, truncated, info

                def get_more_info(self, info):
                    state = self.env.unwrapped.observation()
                    info.update(state[0])
                    info["active"] = np.array([state[i]['active'] for i in range(self.n_agents)])
                    info["designated"] = np.array([state[i]["designated"] for i in range(self.n_agents)])
                    info["sticky_actions"] = np.stack([state[i]["sticky_actions"] for i in range(self.n_agents)])
                    return info

                def get_state(self):
                    """Get global state."""
                    return np.array(self.env.env.state())



    .. group-tab:: gfootball_vec_env.py
    
        .. code-block:: python

            from xuance.environment.vector_envs.vector_env import VecEnv, NotSteppingError
            from xuance.common import combined_shape
            from gymnasium.spaces import Discrete, Box
            import numpy as np
            import multiprocessing as mp
            from xuance.environment.vector_envs.subproc_vec_env import clear_mpi_env_vars, flatten_list, CloudpickleWrapper


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
                            env_info = {
                                "dim_obs": envs[0].dim_obs,
                                "n_actions": envs[0].n_actions,
                                "n_agents": envs[0].n_agents,
                                "n_adversaries": envs[0].n_adversaries,
                                "dim_state": envs[0].dim_state,
                                "dim_act": envs[0].dim_act,
                                "dim_reward": envs[0].dim_reward,
                                "max_cycles": envs[0].max_cycles
                            }
                            remote.send(CloudpickleWrapper(env_info))
                        else:
                            raise NotImplementedError
                except KeyboardInterrupt:
                    print('SubprocVecEnv worker: got KeyboardInterrupt')
                finally:
                    for env in envs:
                        env.close()


            class SubprocVecEnv_GFootball(VecEnv):
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
                    env_info = self.remotes[0].recv().x
                    VecEnv.__init__(self, num_envs, env_info["dim_obs"], env_info["n_actions"])

                    self.num_agents, self.num_adversaries = env_info["n_agents"], env_info["n_adversaries"]
                    self.obs_shape = (env_info["n_agents"], env_info["dim_obs"])
                    self.act_shape = (env_info["n_agents"], env_info["n_actions"])
                    self.rew_shape = (self.num_agents, 1)
                    self.dim_obs, self.dim_state, self.dim_act = env_info["dim_obs"], env_info["dim_state"], env_info["dim_act"]
                    self.dim_reward = env_info["dim_reward"]
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
                    self.max_episode_length = env_info["max_cycles"]

                def reset(self):
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
                                self.buf_rew[idx_env, :, 0], self.buf_terminal[idx_env, 0] = np.array(rew), terminal
                                self.buf_truncation[idx_env, 0], self.buf_info[idx_env] = truncated, infos

                                if self.buf_terminal[idx_env].all() or self.buf_truncation[idx_env].all():
                                    self.buf_done[idx_env] = True
                                    self.battles_game[idx_env] += 1
                                    if infos['score_reward'] > 0:
                                        self.battles_won[idx_env] += 1
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
                    return np.ones([self.num_envs, self.num_agents, self.dim_act], dtype=np.bool)

                def _assert_not_closed(self):
                    assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

                def __del__(self):
                    if not self.closed:
                        self.close()


            class DummyVecEnv_GFootball(VecEnv):
                def __init__(self, env_fns):
                    self.waiting = False
                    self.closed = False
                    num_envs = len(env_fns)

                    self.envs = [fn() for fn in env_fns]
                    env = self.envs[0]
                    VecEnv.__init__(self, len(env_fns), env.dim_obs, env.n_actions)

                    self.num_agents, self.num_adversaries = env.n_agents, env.n_adversaries
                    self.obs_shape = (env.n_agents, env.dim_obs)
                    self.act_shape = (env.n_agents, env.n_actions)
                    self.rew_shape = (self.num_agents, 1)
                    self.dim_obs, self.dim_state, self.dim_act = env.dim_obs, env.dim_state, env.dim_act
                    self.dim_reward = env.dim_reward
                    self.action_space = Discrete(n=self.dim_act)
                    self.state_space = Box(low=-np.inf, high=np.inf, shape=[self.dim_state, ])

                    self.buf_obs = np.zeros(combined_shape(self.num_envs, self.obs_shape), dtype=np.float32)
                    self.buf_state = np.zeros(combined_shape(self.num_envs, self.dim_state), dtype=np.float32)
                    self.buf_terminal = np.zeros((self.num_envs, 1), dtype=np.bool)
                    self.buf_truncation = np.zeros((self.num_envs, 1), dtype=np.bool)
                    self.buf_done = np.zeros((self.num_envs, ), dtype=np.bool)
                    self.buf_rew = np.zeros((self.num_envs, ) + self.rew_shape, dtype=np.float32)
                    self.buf_info = [{} for _ in range(self.num_envs)]
                    self.actions = None
                    self.battles_game = np.zeros(self.num_envs, np.int32)
                    self.battles_won = np.zeros(self.num_envs, np.int32)
                    self.max_episode_length = env.max_cycles

                def reset(self):
                    for i_env, env in enumerate(self.envs):
                        obs, state, infos = env.reset()
                        self.buf_obs[i_env], self.buf_state[i_env] = np.array(obs), np.array(state)
                        self.buf_info[i_env] = infos
                    self.buf_done = np.zeros((self.num_envs,), dtype=np.bool)
                    return self.buf_obs.copy(), self.buf_state.copy(), self.buf_info.copy()

                def step_async(self, actions):
                    self.actions = actions
                    self.waiting = True

                def step_wait(self):
                    if not self.waiting:
                        raise NotSteppingError
                    for idx_env, env_done, env in zip(range(self.num_envs), self.buf_done, self.envs):
                        if not env_done:
                            obs, state, rew, terminal, truncated, infos = env.step(self.actions[idx_env])
                            self.buf_obs[idx_env], self.buf_state[idx_env] = np.array(obs), np.array(state)
                            self.buf_rew[idx_env, :, 0], self.buf_terminal[idx_env, 0] = np.array(rew), np.array(terminal)
                            self.buf_truncation[idx_env], self.buf_info[idx_env] = np.array(truncated), infos

                            if self.buf_terminal[idx_env].all() or self.buf_truncation[idx_env].all():
                                self.buf_done[idx_env] = True
                                self.battles_game[idx_env] += 1
                                if infos['score_reward'] > 0:
                                    self.battles_won[idx_env] += 1
                        else:
                            self.buf_terminal[idx_env, 0], self.buf_truncation[idx_env, 0] = False, False
                    self.waiting = False
                    return self.buf_obs.copy(), self.buf_state.copy(), self.buf_rew.copy(), self.buf_terminal.copy(), self.buf_truncation.copy(), self.buf_info.copy()

                def close_extras(self):
                    self.closed = True
                    for env in self.envs:
                        env.close()

                def render(self, mode):
                    return [env.render() for env in self.envs]

                def get_avail_actions(self):
                    return np.ones([self.num_envs, self.num_agents, self.dim_act], dtype=np.bool)

