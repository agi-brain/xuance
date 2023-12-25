Vectorized Environments
====================================================

vector_env.py
-------------------------------------------------

.. py:class::
    xuance.environment.vector_envs.vector_env.AlreadySteppingError()

    Signal an error condition when perform an asynchronous step while another asynchronous step is already in progress.

.. py:class::
    xuance.environment.vector_envs.vector_env.NotSteppingError()

    Indicate an error condition when an asynchronous step is expected to be in progress.

.. py:function::
    xuance.environment.vector_envs.vector_env.tile_images(images)

    Concatenate a list of input images into a larger image arranged in a grid layout.

    :param images: the input list of images.
    :type images: list
    :return: the resulting concatenated image.
    :rtype: int

.. py:class::
    xuance.environment.vector_envs.vector_env.VecEnv(num_envs, observation_space, action_space)

    This class defines the interface for vectorized environments.

    :param num_envs: the number of environments in the vectorized environment.
    :type num_envs: int
    :param observation_space: the observation space of the environments.
    :type observation_space: Space
    :param action_space: the action space of the environments.
    :type action_space: Space

.. py:function::
    xuance.environment.vector_envs.vector_env.VecEnv.reset()

    Reset all the environments and return an array of observations, or a dict of observation arrays.

.. py:function::
    xuance.environment.vector_envs.vector_env.VecEnv.step_async()

    Start to take a step with the given actions in the environments.

.. py:function::
    xuance.environment.vector_envs.vector_env.VecEnv.step_wait()

    Wait for the step taken with step_async().

.. py:function::
    xuance.environment.vector_envs.vector_env.VecEnv.close_extras()

    Clean up the extra resources, beyond what's in this base class.

.. py:function::
    xuance.environment.vector_envs.vector_env.VecEnv.step(actions)

    Combine the asynchronous step initiation with the subsequent waiting for the step to complete.

    :param actions: the executable actions for the environment.
    :type actions: np.ndarray
    :return: the results of the step taken in the vectorized environment.
    :rtype: tuple

.. py:function::
    xuance.environment.vector_envs.vector_env.VecEnv.render(mode)

    Sends a render command to each subprocess with the specified rendering mode.

    :param mode: determine the rendering mode for the visualization.
    :type mode: str
    :return: the rendered images from subprocesses.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.vector_envs.vector_env.VecEnv.close()

    Close the vectorized environment.

.. raw:: html

    <br><hr>

subproc_vec_env.py
-------------------------------------------------

.. py:function::
    xuance.environment.vector_envs.subproc_vec_env.clear_mpi_env_vars()

    Clear MPI environment variables temporarily.

.. py:function::
    xuance.environment.vector_envs.subproc_vec_env.flatten_list(l)

    Flatten a nested list or tuple into a single-level list.

    :param l: a nested structure.
    :type l: list
    :return: a flattened list containing all the elements from the nested structure.
    :rtype: list

.. py:function::
    xuance.environment.vector_envs.subproc_vec_env.flatten_obs(obs)

    Flatten a list or tuple of observations.

    :param obs: a list or tuple containing observations.
    :type obs: list, tuple
    :return: The flat data.
    :rtype: np.ndarray

.. py:class::
    xuance.environment.vector_envs.subproc_vec_env.CloudpickleWrapper(x)

    A workaround with the default pickle serialization in multiprocessing scenarios.

    :param x: the object that you want to wrap and handle serialization.

.. py:function::
    xuance.environment.vector_envs.subproc_vec_env.CloudpickleWrapper.__getstate__()

    Serialize the object's state using cloudpickle,
    ensuring that the object can be correctly transmitted between processes in a multiprocessing context.

    :return: the serialized state of the object.

.. py:function::
    xuance.environment.vector_envs.subproc_vec_env.CloudpickleWrapper.__setstate__(ob)

    deserialize the object's state, reconstructing the original object, and assigning it to the instance variable self.x

    :param ob: the serialized state of the object as a byte stream.

.. py:function::
    xuance.environment.vector_envs.subproc_vec_env.worker(remote, parent_remote, env_fn_wrappers)

    A worker function that is designed to run in a separate process,
    communicating with its parent process through inter-process communication.

    :param remote: a connection to the child process.
    :type remote: int
    :param parent_remote: a connection to the parent process.
    :type parent_remote: int
    :param env_fn_wrappers: a set of environment function wrappers.

.. py:class::
    xuance.environment.vector_envs.subproc_vec_env.SubprocVecEnv(env_fns, spaces=None, context='spawn', in_series=1)

    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.

    :param env_fns: environment function.
    :param spaces: A dictionary specifying observation and action spaces.
    :type spaces: dict
    :param context: the method used for creating and managing processes in a multiprocessing environment.
    :param in_series: specifies the number of environments to run in series.
    :type in_series: int

.. py:function::
    xuance.environment.vector_envs.subproc_vec_env.SubprocVecEnv.step_async(actions)

    Sends asynchronous step commands to each subprocess with the specified actions.

    :param actions: the executable actions for n parallel environments.
    :type actions: np.ndarray

.. py:function::
    xuance.environment.vector_envs.subproc_vec_env.SubprocVecEnv.step_wait()

    Waits for the completion of asynchronous step operations and updates internal buffers with the received results.

    :return: the observations, states, rewards, terminal flags, truncation flags, and information.
    :rtype: tuple

.. py:function::
    xuance.environment.vector_envs.subproc_vec_env.SubprocVecEnv.reset()

    Reset the vectorized environments.

    :return: the reset observations, global states, and the information.
    :rtype: tuple

.. py:function::
    xuance.environment.vector_envs.subproc_vec_env.SubprocVecEnv.close_extras()

    Closes the communication with subprocesses and joins the subprocesses.

.. py:function::
    xuance.environment.vector_envs.subproc_vec_env.SubprocVecEnv.get_images()

    retrieve rendered images from the environments.

    :return: the rendered images from subprocesses.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.vector_envs.subproc_vec_env.SubprocVecEnv._assert_not_closed()

    Raises an exception if an operation is attempted on the environment after it has been closed.

.. raw:: html

    <br><hr>


env_utils.py
-------------------------------------------------

.. py:function::
    xuance.environment.vector_envs.env_utils.tile_images(images)

    Concatenate a list of input images into a larger image arranged in a grid layout.

    :param images: the input list of images.
    :type images: list
    :return: the resulting concatenated image.
    :rtype: int

.. py:function::
    xuance.environment.vector_envs.env_utils.copy_obs_dict(obs)

    A deep copy of a dictionary containing observations.

    :param obs: a dictionary containing observations.
    :type obs: dict
    :return: a new dictionary with the same keys as the input.
    :rtype: dict

.. py:function::
    xuance.environment.vector_envs.env_utils.dict_to_obs(obs)

    Convert a dictionary representation of observations to a more standard form.

    :param obs: a dictionary containing observations.
    :type obs: dict
    :return: the corresponding value or the original dictionary.
    :rtype: dict

.. py:function::
    xuance.environment.vector_envs.env_utils.obs_space_info(obs_space)

    Extract information about the structure of observation spaces.

    :param obs_space: an observation space.
    :type obs_space: Space
    :return: a tuple containing information about the subspaces: keys, shapes, and data types.
    :rtype: tuple

.. py:function::
    xuance.environment.vector_envs.env_utils.obs_n_space_info(obs_n_space)

    Handle a collection of observation spaces, where each element in the collection is treated as a separate observation space

    :param obs_n_space: an object representing nested observation spaces.
    :type obs_n_space: Space
    :return: a tuple containing information about the subspaces: keys, shapes, and data types.
    :rtype: tuple

.. py:function::
    xuance.environment.vector_envs.env_utils.clear_mpi_env_vars()

    Clear MPI environment variables temporarily.

.. py:function::
    xuance.environment.vector_envs.env_utils.flatten_list(l)

    Flatten a nested list or tuple into a single-level list.

    :param l: a nested structure.
    :type l: list
    :return: a flattened list containing all the elements from the nested structure.
    :rtype: list

.. py:function::
    xuance.environment.vector_envs.env_utils.flatten_obs(obs)

    Flatten a list or tuple of observations.

    :param obs: a list or tuple containing observations.
    :type obs: list, tuple
    :return: The flat data.
    :rtype: np.ndarray

.. py:class::
    xuance.environment.vector_envs.env_utils.CloudpickleWrapper(x)

    Use cloudpickle to serialize contents.

    :param x: the content that needs to be serialized using cloudpickle.

.. py:function::
    xuance.environment.vector_envs.env_utils.CloudpickleWrapper.__getstate__()

    Serialize the object's state using cloudpickle,
    ensuring that the object can be correctly transmitted between processes in a multiprocessing context.

    :return: the serialized state of the object.

.. py:function::
    xuance.environment.vector_envs.env_utils.CloudpickleWrapper.__setstate__(ob)

    deserialize the object's state, reconstructing the original object, and assigning it to the instance variable self.x

    :param ob: the serialized state of the object as a byte stream.

.. raw:: html

    <br><hr>

Source Code
---------------------------------------------

.. tabs::

    .. group-tab:: vector_env.py

        .. code-block:: python

            from abc import ABC, abstractmethod
            import numpy as np
            import cv2


            # referenced from openai/baselines
            class AlreadySteppingError(Exception):
                def __init__(self):
                    msg = 'already running an async step'
                    Exception.__init__(self, msg)


            class NotSteppingError(Exception):
                def __init__(self):
                    msg = 'not running an async step'
                    Exception.__init__(self, msg)


            def tile_images(images):
                image_nums = len(images)
                image_shape = images[0].shape
                image_height = image_shape[0]
                image_width = image_shape[1]
                rows = (image_nums - 1) // 4 + 1
                if image_nums >= 4:
                    cols = 4
                else:
                    cols = image_nums
                try:
                    big_img = np.zeros(
                        (rows * image_height + 10 * (rows - 1), cols * image_width + 10 * (cols - 1), image_shape[2]), np.uint8)
                except IndexError:
                    big_img = np.zeros((rows * image_height + 10 * (rows - 1), cols * image_width + 10 * (cols - 1)), np.uint8)
                for i in range(image_nums):
                    c = i % 4
                    r = i // 4
                    big_img[10 * r + image_height * r:10 * r + image_height * r + image_height,
                    10 * c + image_width * c:10 * c + image_width * c + image_width] = images[i]
                return big_img


            class VecEnv(ABC):
                def __init__(self, num_envs, observation_space, action_space):
                    self.num_envs = num_envs
                    self.observation_space = observation_space
                    self.action_space = action_space
                    self.closed = False

                @abstractmethod
                def reset(self):
                    """
                    Reset all the environments and return an array of
                    observations, or a dict of observation arrays.
                    If step_async is still doing work, that work will
                    be cancelled and step_wait() should not be called
                    until step_async() is invoked again.
                    """
                    pass

                @abstractmethod
                def step_async(self, actions):
                    """
                    Tell all the environments to start taking a step
                    with the given actions.
                    Call step_wait() to get the results of the step.
                    You should not call this if a step_async run is
                    already pending.
                    """
                    pass

                @abstractmethod
                def step_wait(self):
                    """
                    Wait for the step taken with step_async().
                    Returns (obs, rews, dones, infos):
                     - obs: an array of observations, or a dict of
                            arrays of observations.
                     - rews: an array of rewards
                     - dones: an array of "episode done" booleans
                     - infos: a sequence of info objects
                    """
                    pass

                @abstractmethod
                def close_extras(self):
                    """
                    Clean up the  extra resources, beyond what's in this base class.
                    Only runs when not self.closed.
                    """
                    pass

                def step(self, actions):
                    self.step_async(actions)
                    return self.step_wait()

                def render(self, mode):
                    raise NotImplementedError

                def close(self):
                    if self.closed == True:
                        return
                    self.close_extras()
                    self.closed = True


    .. group-tab:: subproc_vec_env.py

        .. code-block:: python

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

    .. group-tab:: env_utils.py

        .. code-block:: python

            import contextlib
            import os
            from collections import OrderedDict

            import gym
            import numpy as np


            def tile_images(images):
                image_nums = len(images)
                image_shape = images[0].shape
                image_height = image_shape[0]
                image_width = image_shape[1]

                rows = (image_nums - 1) // 4 + 1
                if image_nums >= 4:
                    cols = 4
                else:
                    cols = 0

                try:
                    big_img = np.zeros(
                        (rows * image_height + 10 * (rows - 1), cols * image_width + 10 * (cols - 1), image_shape[2]), np.uint8)
                except IndexError:
                    big_img = np.zeros((rows * image_height + 10 * (rows - 1), cols * image_width + 10 * (cols - 1)), np.uint8)

                for i in range(image_nums):
                    c = i % 4
                    r = i // 4
                    big_img[10 * r + image_height * r:10 * r + image_height * r + image_height,
                    10 * c + image_width * c:10 * c + image_width * c + image_width] = images[i]
                return big_img


            def copy_obs_dict(obs):
                return {k: np.copy(v) for k, v in obs.items()}


            def dict_to_obs(obs_dict):
                if set(obs_dict.keys()) == {None}:
                    return obs_dict[None]
                return obs_dict


            def obs_space_info(obs_space):
                if isinstance(obs_space, gym.spaces.Dict):
                    assert isinstance(obs_space.spaces, OrderedDict)
                    subspaces = obs_space.spaces
                elif isinstance(obs_space, gym.spaces.Tuple):
                    subspaces = {i: obs_space.spaces[i] for i in range(len(obs_space.spaces))}
                else:
                    subspaces = {None: obs_space}
                keys = []
                shapes = {}
                dtypes = {}
                for key, box in subspaces.items():
                    keys.append(key)
                    shapes[key] = box.shape
                    dtypes[key] = box.dtype
                return keys, shapes, dtypes


            # for multi-agent systems
            def obs_n_space_info(obs_n_space):
                if isinstance(obs_n_space, gym.spaces.Dict):
                    assert isinstance(obs_n_space.spaces, OrderedDict)
                    subspaces = obs_n_space.spaces
                elif isinstance(obs_n_space, gym.spaces.Tuple):
                    subspaces = {i: obs_n_space.spaces[i] for i in range(len(obs_n_space_info.spaces))}
                elif isinstance(obs_n_space, dict):
                    subspaces = {k: obs_n_space[k] for k in obs_n_space.keys()}
                else:
                    subspaces = {None: obs_n_space}
                keys = []
                shapes = {}
                dtypes = {}
                for key, box in subspaces.items():
                    keys.append(key)
                    shapes[key] = box.shape  # assume the obs_shapes are the same.
                    dtypes[key] = box.dtype
                return keys, shapes, dtypes


            @contextlib.contextmanager
            def clear_mpi_env_vars():
                """
                from mpi4py import MPI will call MPI_Init by default.  If the child process has MPI
                environment variables, MPI will think that the child process is an MPI process just
                like the parent and do bad things such as hang.
                This context manager is a hacky way to clear those environment variables temporarily
                such as when we are starting multiprocessing Processes.
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
