MetaDrive
=======================================

MetaDrive is an autonomous driving simulator that supports generating infinite scenes with various road maps and traffic settings for research of generalizable RL.

| **Official link**: `https://metadriverse.github.io/metadrive/ <https://metadriverse.github.io/metadrive/>`_
| **Paper link**: `https://arxiv.org/pdf/2109.12674.pdf <https://arxiv.org/pdf/2109.12674.pdf>`_

.. raw:: html

    <br><hr>

Installation
-----------------------------------------------

Open the terminal and create your conda environment.
Then, you can choose one of the listed methods to finish the installation of MetaDrive.

**Method 1**: From PyPI.

.. code-block:: bash

    pip install metadrive

**Method 2**: From GitHub.

.. code-block:: bash

    git clone https://github.com/metadriverse/metadrive.git
    cd metadrive
    pip install -e .

.. raw:: html

    <br><hr>

Try an Example
-----------------------------------------------

.. attention::

    Please note that each process should only have one single MetaDrive instance due to the limit of the underlying simulation engine.
    Thus the parallelization of training environment should be in process-level instead of thread-level.

Create a python file named, e.g., "demo_metadrive.py"

.. code-block:: python

    import argparse
    from xuance import get_runner

    def parse_args():
        parser = argparse.ArgumentParser("Run a demo.")
        parser.add_argument("--method", type=str, default="ppo")
        parser.add_argument("--env", type=str, default="metadrive")
        parser.add_argument("--env-id", type=str, default="your_map")
        parser.add_argument("--test", type=int, default=0)
        parser.add_argument("--device", type=str, default="cuda:0")
        parser.add_argument("--parallels", type=int, default=10)
        parser.add_argument("--benchmark", type=int, default=1)
        parser.add_argument("--test-episode", type=int, default=5)

        return parser.parse_args()


    if __name__ == '__main__':
        parser = parse_args()
        runner = get_runner(method=parser.method,
                            env=parser.env,
                            env_id=parser.env_id,
                            parser_args=parser,
                            is_test=parser.test)
        if parser.benchmark:
            runner.benchmark()
        else:
            runner.run()

Open the terminal the type the python command:

.. code-block:: bash

    python demo_metadrive.py

| Then, let your GPU and CPU work and wait for the training process to finish.
| Finally, you can test the trained model and view the effectiveness.

.. code-block:: bash

    python demo_metadrive.py --benchmark 0 --test 1

.. tip::

    When you successfully trained a model and visualize the MetaDrive simulator,
    you might find that the fps is too low to watch the effectiveness.

    **Solution**: You can hold on the F key to accelerate the simulation.

.. raw:: html

    <br><hr>

metadrive_env.py
-----------------------------------------------

.. py:class::
    xuance.environment.metadrive.MetaDrive_Env(args)

    This class is a custom wrapper for MetaDrive environments.

    :param env_id: the arguments for creating an environment.
    :type env_id: SimpleNamespace

.. py:function::
    xuance.environment.metadrive.MetaDrive_Env.close()

    Close the underlying MetaDrive environment.

.. py:function::
    xuance.environment.metadrive.MetaDrive_Env.render(mode)

    Get the rendered images of the environment.
    (In this environment, the render method is null.
    You can visualize the environment by setting the "use_render" config as True in the __init__() method.)

    :param mode: determine the rendering mode for the visualization
    :type mode: str
    :return: the rendered images from subprocesses.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.metadrive.MetaDrive_Env.reset()

    Reset the environment.

    :return: the reset observations, and the information.
    :rtype: tuple

.. py:function::
    xuance.environment.metadrive.MetaDrive_Env.step(actions)

    Take an action as input, perform a step in the underlying MetaDrive environment.

    :param actions: the executable actions for the environment.
    :type actions: np.ndarray
    :return: the next step data, including local observations, rewards, terminated variables, truncated variables, and the other information.
    :rtype: tuple

.. raw:: html

    <br><hr>

metadrive_vec_env.py
-----------------------------------------------

.. py:class::
    xuance.environment.metadrive.SubprocVecEnv_MetaDrive(env_fns, context='spawn', in_series=1)

    This class defines a vectorized environment for the metadrive environments.
    This class in derivated from the xuance.environment.gym.gym_vec_env.SubprocVecEnv_Gym.

    :param env_fns: environment function.
    :param context: the method used for creating and managing processes in a multiprocessing environment.
    :param in_series: specifies the number of environments to run in series.
    :type in_series: int


.. py:class::
    xuance.environment.metadrive.DummyVecEnv_MetaDrive(env_fns)

    A simplified vectorized environment that runs multiple environments sequentially,
    handling one environment at a time.
    This class in derivated from the xuance.environment.gym.gym_vec_env.DummyVecEnv_Gym.

    :param env_fns: environment function.

.. raw:: html

    <br><hr>

Source Code
------------------------------------------------

.. tabs::

    .. group-tab:: metadrive_env.py

        .. code-block:: python

            import numpy as np

            class MetaDrive_Env:
                def __init__(self, args):
                    self.env_id = args.env_id
                    from metadrive.envs.metadrive_env import MetaDriveEnv
                    self.env = MetaDriveEnv(config={"use_render": args.render})

                    self._episode_step = 0  # The count of steps for current episode.
                    self._episode_score = 0.0  # The cumulated rewards for current episode.
                    self.observation_space = self.env.observation_space
                    self.action_space = self.env.action_space
                    self.max_episode_steps = self.env.episode_lengths

                def close(self):
                    self.env.close()

                def render(self, *args, **kwargs):
                    return np.zeros([2, 2, 2])

                def reset(self):
                    obs, info = self.env.reset()
                    self._episode_step = 0  # The count of steps for current episode.
                    self._episode_score = 0.0  # The cumulated rewards for current episode.
                    info["episode_step"] = self._episode_step
                    return obs, info

                def step(self, actions):
                    observation, reward, terminated, truncated, info = self.env.step(actions)

                    self._episode_step += 1
                    self._episode_score += reward
                    info["episode_step"] = self._episode_step  # current episode step
                    info["episode_score"] = self._episode_score  # the accumulated rewards
                    return observation, reward, terminated, truncated, info

    .. group-tab:: metadrive_vec_env.py

        .. code-block:: python

            from xuance.environment.gym.gym_vec_env import SubprocVecEnv_Gym, DummyVecEnv_Gym, worker

            class SubprocVecEnv_MetaDrive(SubprocVecEnv_Gym):
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
                    super(SubprocVecEnv_MetaDrive, self).__init__(env_fns, context, in_series)


            class DummyVecEnv_MetaDrive(DummyVecEnv_Gym):
                """
                VecEnv that does runs multiple environments sequentially, that is,
                the step and reset commands are send to one environment at a time.
                Useful when debugging and when num_env == 1 (in the latter case,
                avoids communication overhead)
                """
                def __init__(self, env_fns):
                    super(DummyVecEnv_MetaDrive, self).__init__(env_fns)
