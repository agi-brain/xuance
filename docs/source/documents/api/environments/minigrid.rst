Minigrid
==============================================

.. image:: ../../../figures/minigrid/crossing.gif
    :height: 150px
.. image:: ../../../figures/minigrid/memory.gif
    :height: 150px
.. image:: ../../../figures/minigrid/lockedroom.gif
    :height: 150px
.. image:: ../../../figures/minigrid/playground.gif
    :height: 150px

.. raw:: html

    <br><hr>

minigrid_env.py
-------------------------------------------------

.. py:class::
    xuance.environment.minigrid.minigrid_env.MiniGridEnv(env_id: str, seed: int, render_mode: str, rgb_img_partial_obs_wrapper=False, img_obs_wrapper=False)

    This is a wrapper class for a MiniGrid environment.

    :param env_id: the environment id of minigrid.
    :type env_id: str
    :param seed: use to control randomness within the environment.
    :type seed: int
    :param render_mode: specifies how the environment should be rendered.
    :type render_mode: str
    :param rgb_img_partial_obs_wrapper: whether to apply the RGB image's partial observation wrapper.
    :type rgb_img_partial_obs_wrapper: bool
    :param img_obs_wrapper: whether to apply the image observation wrapper.
    :type img_obs_wrapper: bool

.. py:function::
    xuance.environment.minigrid.minigrid_env.MiniGridEnv.close()

    Close the environment.

.. py:function::
    xuance.environment.minigrid.minigrid_env.MiniGridEnv.render()

    Return the rendering result.

    :return: the rendering result.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.minigrid.minigrid_env.MiniGridEnv.reset()

    Reset the environment.

    :return: the initial flattened observation of the environment and additional information.
    :rtype: tuple

.. py:function::
    xuance.environment.minigrid.minigrid_env.MiniGridEnv.step(actions)

    Execute the actions and get next observations, rewards, and other information.

    :param actions: the executable actions for the environment.
    :type actions: np.ndarray
    :return: represent a stack of frames used as the initial observation for the environment, including rewards, terminated variables, truncated variables, and the other information.
    :rtype: tuple

.. py:function::
    xuance.environment.minigrid.minigrid_env.MiniGridEnv.flatten_obs(obs_raw)

    Convert image observation to vectors.

    :param obs_raw: the raw observation dictionary containing image and direction.
    :type obs_raw: dict
    :return: flattened observation vectors.
    :rtype: np.ndarray

.. raw:: html

    <br><hr>

minigrid_vec__env.py
-------------------------------------------------

.. py:class::
    xuance.environment.minigrid.minigrid_vec_env.SubprocVecEnv_MiniGrid(env_fns, context='spawn', in_series=1)

    Extend the functionality of a subprocess-based vectorized environment.

    :param env_fns: environment function.
    :param context:  the method used for creating and managing processes in a multiprocessing environment.
    :param in_series: specifies the number of environments to run in series.
    :type in_series: int

.. py:class::
    xuance.environment.minigrid.minigrid_vec_env.DummyVecEnv_MiniGrid(env_fns)

    Extends the functionality of a dummy vectorized environment

    :param env_fns: environment function.

.. raw:: html

    <br><hr>

Source Code
------------------------------------------------

.. tabs::

    .. group-tab:: minigrid_env.py

        .. code-block:: python

            import gymnasium as gym
            from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
            from gym.spaces import Box, Discrete
            import numpy as np


            class MiniGridEnv():
                """
                The wrapper of minigrid environment.

                Args:
                    env_id: The environment id of minigrid.
                    seed: random seed.
                    render_mode: "rgb_array", "human".
                    rgb_img_partial_wrapper: whether to apply the RGB image's partial observation wrapper.
                    img_obs_wrapper:  whether to apply the image observation wrapper.
                """
                def __init__(self, env_id: str, seed: int, render_mode: str,
                             rgb_img_partial_obs_wrapper=False,
                             img_obs_wrapper=False):
                    self.env = gym.make(env_id, render_mode=render_mode)
                    if rgb_img_partial_obs_wrapper:
                        self.env = RGBImgPartialObsWrapper(self.env)
                    if img_obs_wrapper:
                        self.env = ImgObsWrapper(self.env)

                    self.env_id = env_id
                    self.render_mode = render_mode
                    self._episode_step = 0
                    self._episode_score = 0.0
                    self.image_size = np.prod(self.env.observation_space['image'].shape)  # height * width * channels
                    self.dim_obs = self.image_size + 1  # direction
                    self.observation_space = Box(low=0, high=255, shape=[self.dim_obs, ], dtype=np.uint8, seed=seed)
                    self.action_space = self.env.action_space
                    self.max_episode_steps = self.env.env.env.max_steps

                def close(self):
                    """Close the environment."""
                    self.env.close()

                def render(self, *args):
                    """Return the rendering result"""
                    return self.env.render()

                def reset(self):
                    """Reset the environment."""
                    obs_raw, info = self.env.reset()
                    obs = self.flatten_obs(obs_raw)
                    self._episode_step = 0
                    self._episode_score = 0.0
                    info["episode_step"] = self._episode_step
                    return obs, info

                def step(self, actions):
                    """Execute the actions and get next observations, rewards, and other information."""
                    obs_raw, reward, terminated, truncated, info = self.env.step(actions)
                    observation = self.flatten_obs(obs_raw)

                    reward *= 10

                    self._episode_step += 1
                    self._episode_score += reward
                    info["episode_step"] = self._episode_step  # current episode step
                    info["episode_score"] = self._episode_score  # the accumulated rewards
                    return observation, reward, terminated, truncated, info

                def flatten_obs(self, obs_raw):
                    """Convert image observation to vectors"""
                    image = obs_raw['image']
                    direction = obs_raw['direction']
                    observations = np.append(image.reshape(-1), direction)
                    return observations


    .. group-tab:: minigrid_vec_env.py

        .. code-block:: python

            from xuance.environment.gym.gym_vec_env import SubprocVecEnv_Gym, DummyVecEnv_Gym, worker


            class SubprocVecEnv_MiniGrid(SubprocVecEnv_Gym):
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
                    super(SubprocVecEnv_MiniGrid, self).__init__(env_fns, context, in_series)


            class DummyVecEnv_MiniGrid(DummyVecEnv_Gym):
                """
                VecEnv that does runs multiple environments sequentially, that is,
                the step and reset commands are send to one environment at a time.
                Useful when debugging and when num_env == 1 (in the latter case,
                avoids communication overhead)
                """
                def __init__(self, env_fns):
                    super(DummyVecEnv_MiniGrid, self).__init__(env_fns)
