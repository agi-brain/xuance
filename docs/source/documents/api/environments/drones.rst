Minigrid
==============================================

.. raw:: html

    <br><hr>

drones_env.py
-------------------------------------------------

.. py:class::
    xuance.environment.drones.drones_env.Drones_Env(args)

    This is a wrapper class for a Drones_Env environment.

    :param args: An argument object that contains various settings and parameters for initializing the environment.
    :type args: object
    :param args.continuous: Determines whether the drone operates in a continuous control mode.
    :type args.continuous: bool
    :param args.env_id: Specifies the type of PyBullet Drones environment to instantiate.
    :type args.env_id: str
    :param args.render: Determines whether to render the environment with a graphical interface.
    :type args.render: bool
    :param args.record: Determines whether to record the environment's visual output.
    :type args.record: bool
    :param args.max_episode_steps: Maximum number of steps per episode for the environment.
    :type args.max_episode_steps: int

.. py:function::
    xuance.environment.drones.drones_env.Drones_Env.space_reshape(gym_space)

    Reshape the given Gym space into a new Box space with flattened boundaries.

    :param gym_space: The Gym space that needs to be reshaped.
    :type gym_space: gym.spaces.Space
    :return: A reshaped Box space with flattened boundaries.
    :rtype: gym.spaces.Box

.. py:function::
    xuance.environment.drones.drones_env.Drones_Env.close()

    Close the environment.

.. py:function::
    xuance.environment.drones.drones_env.Drones_Env.render()

    Return the rendering result.

    :return: the rendering result.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.drones.drones_env.Drones_Env.reset()

    Reset the environment.

    :return: The initial observation of the environment as a flattened 1-dimensional array and
             additional information regarding the environment's state.
    :rtype: tuple

.. py:function::
    xuance.environment.drones.drones_env.Drones_Env.step(actions)

    Execute the actions and get next observations, rewards, and other information.

    :param actions: Actions to be executed in the environment. The actions are reshaped to be compatible with the environment's expectations.
    :type actions: np.ndarray
    :return: A tuple containing the flattened initial observation of the environment, the received reward,
             a termination indicator, a truncation indicator, and additional environment-related information.
    :rtype: tuple

.. raw:: html

    <br><hr>

drones_vec__env.py
-------------------------------------------------

.. raw:: html

    <br><hr>

Source Code
------------------------------------------------

.. tabs::

    .. group-tab:: drones_env.py

        .. code-block:: python

            from gym.spaces import Box


            class Drones_Env():
                def __init__(self, args):
                    # import scenarios of gym-pybullet-drones
                    from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
                    from gym_pybullet_drones.envs.HoverAviary import HoverAviary
                    from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
                    REGISTRY = {
                        "CtrlAviary": CtrlAviary,
                        "HoverAviary": HoverAviary,
                        "VelocityAviary": VelocityAviary
                    }
                    continuous = args.continuous
                    self.env_id = args.env_id

                    from gym_pybullet_drones.utils.enums import DroneModel, Physics
                    self.env = REGISTRY[args.env_id](
                        drone_model=DroneModel.CF2X,
                        initial_xyzs=None,
                        initial_rpys=None,
                        physics=Physics.PYB,
                        pyb_freq=240,
                        ctrl_freq=240,
                        gui=args.render,
                        record=args.record
                    )
                    self._episode_step = 0
                    self._episode_score = 0.0
                    self.observation_space = self.space_reshape(self.env.observation_space)
                    self.action_space = self.space_reshape(self.env.action_space)
                    self.max_episode_steps = args.max_episode_steps

                def space_reshape(self, gym_space):
                    low = gym_space.low.reshape(-1)
                    high = gym_space.high.reshape(-1)
                    shape_obs = (gym_space.shape[-1], )
                    return Box(low=low, high=high, shape=shape_obs, dtype=gym_space.dtype)

                def close(self):
                    self.env.close()

                def render(self):
                    return self.env.render()

                def reset(self):
                    obs, info = self.env.reset()
                    self._episode_step = 0
                    self._episode_score = 0.0
                    info["episode_step"] = self._episode_step
                    return obs.reshape(-1), info

                def step(self, actions):
                    observation, reward, terminated, truncated, info = self.env.step(actions.reshape([1, -1]))

                    self._episode_step += 1
                    self._episode_score += reward
                    info["episode_step"] = self._episode_step  # current episode step
                    info["episode_score"] = self._episode_score  # the accumulated rewards

                    truncated = True if (self._episode_step >= self.max_episode_steps) else False

                    return observation.reshape(-1), reward, terminated, truncated, info


    .. group-tab:: drones_vec__env.py

        .. code-block:: python


