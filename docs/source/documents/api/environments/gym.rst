Gym
=======================================

Box2D
---------------------------------------

.. image:: ../../../figures/box2d/car_racing.gif
    :height: 120px
.. image:: ../../../figures/box2d/lunar_lander.gif
    :height: 120px
.. image:: ../../../figures/box2d/bipedal_walker.gif
    :height: 120px

Atari
---------------------------------------

.. image:: ../../../figures/atari/adventure.gif
    :height: 150px
.. image:: ../../../figures/atari/air_raid.gif
    :height: 150px
.. image:: ../../../figures/atari/alien.gif
    :height: 150px
.. image:: ../../../figures/atari/boxing.gif
    :height: 150px
.. image:: ../../../figures/atari/breakout.gif
    :height: 150px

MuJoCo
-----------------------------------------

.. image:: ../../../figures/mujoco/ant.gif
    :height: 150px
.. image:: ../../../figures/mujoco/half_cheetah.gif
    :height: 150px
.. image:: ../../../figures/mujoco/hopper.gif
    :height: 150px
.. image:: ../../../figures/mujoco/humanoid.gif
    :height: 150px

.. raw:: html

    <br><hr>

gym_env.py
-----------------------------------------------

.. py:class::
    xuance.environment.gym.gym_env.Gym_Env(env_id, seed, render_mode)

    This class is a custom wrapper for Gym environments.

    :param env_id: environment id.
    :type env_id: str
    :param seed: use to control randomness within the environment.
    :type seed: int
    :param render_mode: specifies how the environment should be rendered.
    :type render_mode: str

.. py:function::
    xuance.environment.gym.gym_env.Gym_Env.close()

    Close the underlying Gym environment.

.. py:function::
    xuance.environment.gym.gym_env.Gym_Env.render(mode)

    Get the rendered images of the environment.

    :param mode: determine the rendering mode for the visualization
    :type mode: str
    :return: the rendered images from subprocesses.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.gym.gym_env.Gym_Env.reset()

    Reset the vectorized environments.

    :return: the reset observations, global states, and the information.
    :rtype: tuple

.. py:function::
    xuance.environment.gym.gym_env.Gym_Env.step(actions)

    Take an action as input, perform a step in the underlying Gym environment.

    :param actions: the executable actions for the environment.
    :type actions: np.ndarray
    :return: the next step data, including local observations, rewards, terminaled variables, truncated variables, and the other information.
    :rtype: tuple

.. py:class::
    xuance.environment.gym.gym_env.MountainCar(env_id, seed, render_mode)

    A custom Gym environment designed for the MountainCar task.

    :param env_id: environment id.
    :type env_id: str
    :param seed: use to control randomness within the environment.
    :type seed: int
    :param render_mode: specifies how the environment should be rendered.
    :type render_mode: str

.. py:function::
    xuance.environment.gym.gym_env.MountainCar.reset()

    Reset the vectorized environments.

    :return: represent the stacked frames and additional episode-related information.
    :rtype: tuple

.. py:function::
    xuance.environment.gym.gym_env.MountainCar.step(actions)

    Take an action as input, perform a step in the underlying Gym environment

    :param actions: the executable actions for the environment.
    :type actions: np.ndarray
    :return: represent a stack of frames used as the initial observation for the environment, including rewards, terminated variables, truncated variables, and the other information.
    :rtype: tuple

.. py:class::
    xuance.environment.gym.gym_env.Atari_Env(env_id, seed, render_mode, obs_type, frame_skip, num_stack, image_size, noop_max)

    Provide a modified version of Atari environments.

    :param env_id: environment id.
    :type env_id: str
    :param seed: use to control randomness within the environment.
    :type seed: int
    :param render_mode: specifies how the environment should be rendered.
    :type render_mode: str
    :param obs_type: type of observations to be returned.
    :type obs_type: str
    :param frame_skip: number of frames to skip between each returned frame.
    :type frame_skip: int
    :param num_stack: number of frames to stack for frame stacking.
    :type num_stack: int
    :param image_size: size of the observation image.
    :type image_size: int
    :param noop_max: maximum number of no-op actions during environment reset.
    :type noop_max: int

.. py:function::
    xuance.environment.gym.gym_env.Atari_Env.close()

    Close the underlying Gym environment.

.. py:function::
    xuance.environment.gym.gym_env.Atari_Env.render(render_mode)

    Get the rendered images of the environment.

    :param render_mode: rendering mode for visualization.
    :type render_mode: str
    :return: a visual representation of the environment in the specified rendering mode.

.. py:function::
    xuance.environment.gym.gym_env.Atari_Env.reset()

    Reset the vectorized environments.

    :return: represent the stacked frames and additional episode-related information.
    :rtype: tuple

.. py:function::
    xuance.environment.gym.gym_env.Atari_Env.step(actions)

    Take an action as input, perform a step in the underlying Gym environment.

    :param actions: the executable actions for the environment.
    :type actions: np.ndarray
    :return: the next observation, modified reward, episode termination status, truncation information, and additional details for monitoring and analysis.
    :rtype: tuple

.. py:function::
    xuance.environment.gym.gym_env.Atari_Env._get_obs()

    Retrieve the current observation by stacking the last frames.

    :return: the returned observation is used as input.

.. py:function::
    xuance.environment.gym.gym_env.Atari_Env.observation(frame)

    Preprocess an individual frame obtained from the environment.

    :param frame: an individual frame obtained from the environment.
    :type frame: np.ndarray
    :return: the processed frame based on the specified observation type.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.gym.gym_env.Atari_Env.reward(reward)

    Convert the original reward to its sign.

    :param reward: represent the numerical reward obtained from the environment.
    :type reward: np.ndarray
    :return: shaped reward using the sign function.
    :rtype: np.ndarray

.. py:class::
    xuance.environment.gym.gym_env.LazyFrames(frames)

    Optimize memory usage when dealing with sequences of frames.

    :param frames: a sequence or list of individual frames.
    :type frames: np.ndarray

.. py:function::
    xuance.environment.gym.gym_env.LazyFrames._force()

    Make sure to concatenate frames only when it is necessary.

    :return: present the frames in their optimized.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.gym.gym_env.LazyFrames.__array__(dtype=None)

    Allow an object to be converted to a numPy array.

    :param dtype: specifies the desired data type for the NumPy array.
    :type dtype: np.dtype
    :return: the numPy array containing the frames.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.gym.gym_env.LazyFrames.__len__()

    Provide a way to obtain the number of frames.

    :return: return an integer representing the length of the LazyFrames object.
    :rtype: int

.. py:function::
    xuance.environment.gym.gym_env.LazyFrames.__getitem__(i)

    Retrieves a specific frame from the concatenated frames.

    :param i: the index or slice notation used to access a specific frame or a subset of frames.
    :type i: int
    :return: the selected frame or frames at the specified index i.
    :rtype: np.ndarray

.. raw:: html

    <br><hr>

gym_vec_env.py
-----------------------------------------------

.. py:function::
    xuance.environment.gym.gym_vec_env.worker(remote, parent_remote, env_fn_wrappers)

    A worker function that is designed to run in a separate process,
    communicating with its parent process through inter-process communication (IPC).

    :param remote: a connection to the child process.
    :type remote: int
    :param parent_remote: a connection to the parent process.
    :type parent_remote: int
    :param env_fn_wrappers: a set of environment function wrappers.

.. py:class::
    xuance.environment.gym.gym_vec_env.SubprocVecEnv_Gym(env_fns, context='spawn', in_series=1)

    This class defines a vectorized environment for the gym environments.

    :param env_fns: environment function.
    :param context: the method used for creating and managing processes in a multiprocessing environment.
    :param in_series: specifies the number of environments to run in series.
    :type in_series: int

.. py:function::
    xuance.environment.gym.gym_vec_env.SubprocVecEnv_Gym.step_async(actions)

    Sends asynchronous step commands to each subprocess with the specified actions.

    :param actions: the executable actions for n parallel environments.
    :type actions: np.ndarray

.. py:function::
    xuance.environment.gym.gym_vec_env.SubprocVecEnv_Gym.step_wait()

    Waits for the completion of asynchronous step operations and updates internal buffers with the received results.

    :return: the observations, states, rewards, terminal flags, truncation flags, and information.
    :rtype: tuple

.. py:function::
    xuance.environment.gym.gym_vec_env.SubprocVecEnv_Gym.reset()

    Reset the vectorized environments.

    :return: the reset observations, global states, and the information.
    :rtype: tuple

.. py:function::
    xuance.environment.gym.gym_vec_env.SubprocVecEnv_Gym.close_extras()

    Closes the communication with subprocesses and joins the subprocesses.

.. py:function::
    xuance.environment.gym.gym_vec_env.SubprocVecEnv_Gym.render(mode)

    Sends a render command to each subprocess with the specified rendering mode.

    :param mode: determine the rendering mode for the visualization.
    :type mode: str
    :return: the rendered images from subprocesses.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.gym.gym_vec_env.SubprocVecEnv_Gym._assert_not_closed()

    Raises an exception if an operation is attempted on the environment after it has been closed.

.. py:function::
    xuance.environment.gym.gym_vec_env.SubprocVecEnv_Gym.__del__()

    The __del__ method ensures that the environment is properly closed when the object is deleted.


.. py:class::
    xuance.environment.gym.gym_vec_env.DummyVecEnv_Gym(env_fns)

    A simplified vectorized environment that runs multiple environments sequentially,
    handling one environment at a time.

    :param env_fns: environment function.

.. py:function::
    xuance.environment.gym.gym_vec_env.DummyVecEnv_Gym.reset()

    Reset the vectorized environments.

    :return: the reset observations, global states, and the information.
    :rtype: tuple

.. py:function::
    xuance.environment.gym.gym_vec_env.DummyVecEnv_Gym.step_async(actions)

    Sends asynchronous step commands to each subprocess with the specified actions.

    :param actions: the executable actions for n parallel environments.
    :type actions: np.ndarray

.. py:function::
    xuance.environment.gym.gym_vec_env.DummyVecEnv_Gym.step_wait()

    Waits for the completion of asynchronous step operations and updates internal buffers with the received results.

    :return: the observations, states, rewards, terminal flags, truncation flags, and information.
    :rtype: tuple

.. py:function::
    xuance.environment.gym.gym_vec_env.DummyVecEnv_Gym.close_extras()

    Closes the communication with subprocesses and joins the subprocesses.

.. py:function::
    xuance.environment.gym.gym_vec_env.DummyVecEnv_Gym.render(mode)

    Sends a render command to each subprocess with the specified rendering mode.

    :param mode: determine the rendering mode for the visualization.
    :type mode: str
    :return: the rendered images from subprocesses.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.gym.gym_vec_env.DummyVecEnv_Gym._save_obs(e, obs)

    Store observations for a specific environment at a given index.

    :param e: the index of the environment for which the observation is being saved.
    :type e: int
    :param obs: the observation obtained from the environment.
    :type obs: np.ndarray

.. py:function::
    xuance.environment.gym.gym_vec_env.DummyVecEnv_Gym._save_infos(e, info)

    Store information for a specific environment at a given index.

    :param e: the index of the environment for which the information is being saved.
    :type e: int
    :param info: the information associated with the current step in the environment.
    :type info: dict

.. py:class::
    xuance.environment.gym.gym_vec_env.DummyVecEnv_Atari(env_fns)

    A vectorized environment wrapper that runs multiple Atari environments sequentially.

    :param env_fns: environment function.

.. py:class::
    xuance.environment.gym.gym_vec_env.SubprocVecEnv_Atari(env_fns)

    Parallelize execution of multiple Atari environments using subprocesses.

    :param env_fns: environment function.

.. raw:: html

    <br><hr>

Source Code
------------------------------------------------

.. tabs::

    .. group-tab:: gym_env.py

        .. code-block:: python

            import gym
            import numpy as np
            from collections import deque
            from typing import Sequence
            import cv2


            class Gym_Env(gym.Wrapper):
                """
                Args:
                    env_id: The environment id of Atari, such as "Breakout-v5", "Pong-v5", etc.
                    seed: random seed.
                    render_mode: "rgb_array", "human"
                """

                def __init__(self, env_id: str, seed: int, render_mode: str, **kwargs):
                    self.env = gym.make(env_id, render_mode=render_mode, **kwargs)
                    self.env.action_space.seed(seed=seed)
                    self.env.reset(seed=seed)
                    super(Gym_Env, self).__init__(self.env)
                    # self.env.seed(seed)
                    self.observation_space = self.env.observation_space
                    self.action_space = self.env.action_space
                    self.metadata = self.env.metadata
                    self.reward_range = self.env.reward_range
                    self.max_episode_steps = self.env._max_episode_steps
                    self._episode_step = 0
                    self._episode_score = 0.0

                def close(self):
                    self.env.close()

                def render(self, mode):
                    return self.env.render()

                def reset(self):
                    obs, info = self.env.reset()
                    self._episode_step = 0
                    self._episode_score = 0.0
                    info["episode_step"] = self._episode_step
                    return obs, info

                def step(self, actions):
                    observation, reward, terminated, truncated, info = self.env.step(actions)
                    self._episode_step += 1
                    self._episode_score += reward
                    info["episode_step"] = self._episode_step
                    info["episode_score"] = self._episode_score
                    return observation, reward, terminated, truncated, info


            class MountainCar(Gym_Env):
                def __init__(self, env_id: str, seed: int, render_mode: str):
                    super(MountainCar, self).__init__(env_id, seed, render_mode)
                    self.num_stack = 4
                    self.frames = deque([], maxlen=self.num_stack)
                    self.observation_space = gym.spaces.Box(low=np.array([-1.2, -0.07, -1.2, -0.07, -1.2, -0.07, -1.2, -0.07]),
                                                            high=np.array([0.6, 0.07, 0.6, 0.07, 0.6, 0.07, 0.6, 0.07]),
                                                            shape=(8,), dtype=np.float32)
                    self.pre_position = 0.0

                def reset(self):
                    obs, info = self.env.reset()
                    self._episode_step = 0
                    self._episode_score = 0.0
                    info["episode_step"] = self._episode_step
                    for i in range(self.num_stack):
                        self.frames.append(obs)
                    self.pre_position = obs[0]
                    return LazyFrames(list(self.frames)), info

                def step(self, actions):
                    observation, reward, terminated, truncated, info = self.env.step(actions)
                    self._episode_step += 1
                    self._episode_score += reward
                    info["episode_step"] = self._episode_step
                    info["episode_score"] = self._episode_score

                    # reward += 10 * observation[0]
                    # reward + 10 * (observation[0] - self.pre_position)
                    # reward += observation[1] ** 2
                    self.frames.append(observation)
                    self.pre_position = observation[0]

                    return LazyFrames(list(self.frames)), reward, terminated, truncated, info


            class Atari_Env(gym.Wrapper):
                """
                We modify the Atari environment to accelerate the training with some tricks:
                    Episode termination: Make end-of-life == end-of-episode, but only reset on true game over. Done by DeepMind for the DQN and co. since it helps value estimation.
                    Frame skipping: Return only every `skip`-th frame.
                    Observation resize: Warp frames from 210x160 to 84x84 as done in the Nature paper and later work.
                    Frame Stacking: Stack k last frames. Returns lazy array, which is much more memory efficient.
                Args:
                    env_id: The environment id of Atari, such as "Breakout-v5", "Pong-v5", etc.
                    seed: random seed.
                    obs_type: This argument determines what observations are returned by the environment. Its values are:
                                ram: The 128 Bytes of RAM are returned
                                rgb: An RGB rendering of the game is returned
                                grayscale: A grayscale rendering is returned
                    frame_skip: int or a tuple of two ints. This argument controls stochastic frame skipping, as described in the section on stochasticity.
                    num_stack: int, the number of stacked frames if you use the frame stacking trick.
                    image_size: This argument determines the size of observation image, default is [210, 160].
                    noop_max: max times of noop action for env.reset().
                """

                def __init__(self,
                             env_id: str,
                             seed: int,
                             render_mode: str = "rgb_array",
                             obs_type: str = "grayscale",
                             frame_skip: int = 4,
                             num_stack: int = 4,
                             image_size: Sequence[int] = None,
                             noop_max: int = 30,
                             ):
                    self.env = gym.make(env_id,
                                        render_mode=render_mode,
                                        obs_type=obs_type,
                                        frameskip=frame_skip)
                    self.env.action_space.seed(seed=seed)
                    self.env.unwrapped.reset(seed=seed)
                    self.max_episode_steps = self.env._max_episode_steps
                    super(Atari_Env, self).__init__(self.env)
                    # self.env.seed(seed)
                    self.num_stack = num_stack
                    self.obs_type = obs_type
                    self.frames = deque([], maxlen=self.num_stack)
                    self.image_size = [210, 160] if image_size is None else image_size
                    self.noop_max = noop_max
                    self.lifes = self.env.unwrapped.ale.lives()
                    self.was_real_done = True
                    self.grayscale, self.rgb = False, False
                    if self.obs_type == "rgb":
                        self.rgb = True
                        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                                shape=(image_size[0], image_size[1], 3 * self.num_stack),
                                                                dtype=np.uint8)
                    elif self.obs_type == "grayscale":
                        self.grayscale = True
                        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                                shape=(image_size[0], image_size[1], self.num_stack),
                                                                dtype=np.uint8)
                    else:  # ram type
                        self.observation_space = self.env.observation_space
                    # assert self.env.unwrapped.get_action_meanings()[0] == "NOOP"
                    # assert self.env.unwrapped.get_action_meanings()[1] == "FIRE"
                    # assert len(self.env.unwrapped.get_action_meanings()) >= 3
                    self.action_space = self.env.action_space
                    self.metadata = self.env.metadata
                    self.reward_range = self.env.reward_range
                    self._render_mode = render_mode
                    self._episode_step = 0
                    self._episode_score = 0.0

                def close(self):
                    self.env.close()

                def render(self, render_mode):
                    return self.env.unwrapped.render(render_mode)

                def reset(self):
                    info = {}
                    if self.was_real_done:
                        self.env.unwrapped.reset()
                        # Execute NoOp actions
                        num_noops = np.random.randint(0, self.noop_max)
                        for _ in range(num_noops):
                            obs, _, done, _ = self.env.unwrapped.step(0)
                            if done:
                                self.env.unwrapped.reset()
                        # try to fire
                        obs, _, done, _ = self.env.unwrapped.step(1)
                        if done:
                            obs = self.env.unwrapped.reset()
                        # stack reset observations
                        for _ in range(self.num_stack):
                            self.frames.append(self.observation(obs))

                        self._episode_step = 0
                        self._episode_score = 0.0
                        info["episode_step"] = 0
                    else:
                        obs, _, done, _ = self.env.unwrapped.step(0)
                        for _ in range(self.num_stack):
                            self.frames.append(self.observation(obs))

                    self.lifes = self.env.unwrapped.ale.lives()
                    self.was_real_done = False
                    return self._get_obs(), info

                def step(self, actions):
                    observation, reward, terminated, info = self.env.unwrapped.step(actions)
                    self.frames.append(self.observation(observation))
                    lives = self.env.unwrapped.ale.lives()
                    # avoid environment bug
                    if self._episode_step >= self.max_episode_steps:
                        terminated = True
                    self.was_real_done = terminated
                    if (lives < self.lifes) and (lives > 0):
                        terminated = True
                    truncated = self.was_real_done
                    self.lifes = lives
                    self._episode_step += 1
                    self._episode_score += reward
                    info["episode_score"] = self._episode_score
                    info["episode_step"] = self._episode_step
                    return self._get_obs(), self.reward(reward), terminated, truncated, info

                def _get_obs(self):
                    assert len(self.frames) == self.num_stack
                    return LazyFrames(list(self.frames))

                def observation(self, frame):
                    if self.grayscale:
                        return np.expand_dims(cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA), -1)
                    elif self.rgb:
                        return cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA)
                    else:
                        return frame

                def reward(self, reward):
                    return np.sign(reward)


            class LazyFrames(object):
                """
                This object ensures that common frames between the observations are only stored once.
                It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay buffers.
                This object should only be converted to numpy array before being passed to the model.
                """

                def __init__(self, frames):
                    self._frames = frames
                    self._out = None

                def _force(self):
                    if self._out is None:
                        self._out = np.concatenate(self._frames, axis=-1)
                        self._frames = None
                    return self._out

                def __array__(self, dtype=None):
                    out = self._force()
                    if dtype is not None:
                        out = out.astype(dtype)
                    return out

                def __len__(self):
                    return len(self._force())

                def __getitem__(self, i):
                    return self._force()[..., i]

    .. group-tab:: gym_vec_env.py

        .. code-block:: python

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
                    self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
                    self.buf_trunctions = np.zeros((self.num_envs,), dtype=np.bool)
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
                    self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
                    self.buf_trunctions = np.zeros((self.num_envs,), dtype=np.bool)
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
